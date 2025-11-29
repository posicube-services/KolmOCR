import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import time

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_bbox_special_tokens(processor) -> bool:
    """Make sure bbox markers are known to the tokenizer."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return False

    required_tokens = ["[BBOX_BLK_END]"]
    vocab = tokenizer.get_vocab()
    missing_tokens = [token for token in required_tokens if token not in vocab]
    if not missing_tokens:
        return False

    tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
    logger.info("Registered bbox special tokens: %s", missing_tokens)
    return True


def build_inputs(processor, image: Image.Image, prompt: str, device: Optional[torch.device], move_to_device: bool) -> Dict[str, Any]:
    """Prepare chat-formatted inputs for Qwen2.5-VL."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    if move_to_device and device is not None:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs


def should_move_to_single_device(model) -> bool:
    """Determine if inputs should be moved to a single device.

    When `device_map` is set across multiple GPUs, leave inputs on CPU and let HF handle dispatch.
    """
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return True
    devices = {str(v) for v in device_map.values()}
    gpu_devices = {d for d in devices if d.startswith("cuda")}
    # If multiple CUDA devices are used, rely on HF dispatch instead of manual .to(device).
    return len(gpu_devices) <= 1


def _extract_images_from_markdown(md_path: Path) -> Iterable[Path]:
    """Find image references in a markdown file and yield resolved paths."""
    import re

    md_image = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    html_image = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']')
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    matches = md_image.findall(text) + html_image.findall(text)
    for raw in matches:
        # Strip surrounding whitespace or quotes and resolve relative to the md file location.
        cleaned = raw.strip().strip('"').strip("'")
        resolved = (md_path.parent / cleaned).resolve()
        yield resolved


def list_images_and_root(input_path: Path) -> Tuple[Iterable[Path], Path]:
    """Return iterable of images and the root used for relative output paths."""
    exts = {".png", ".jpg", ".jpeg"}
    if input_path.is_file() and input_path.suffix.lower() == ".md":
        return _extract_images_from_markdown(input_path), input_path.parent
    if input_path.is_file() and input_path.suffix.lower() in exts:
        return [input_path], input_path.parent
    # Default: directory traversal
    images = (p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in exts)
    return images, input_path


def _relative_to_root(path: Path, root: Path) -> Path:
    """Return path relative to root, falling back to basename if not possible."""
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _save_ground_truth_image(image: Image.Image, md_output_path: Path) -> Path:
    """Save the input image alongside the markdown with a `_gt.png` suffix."""
    gt_path = md_output_path.with_name(md_output_path.stem + "_gt.png")
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(gt_path, format="PNG")
    return gt_path


def _save_input_image(image: Image.Image, md_output_path: Path) -> Path:
    """Save a copy of the input image next to the markdown with a `_input.png` suffix."""
    input_path = md_output_path.with_name(md_output_path.stem + "_input.png")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(input_path, format="PNG")
    return input_path


def _decode_generation(processor, generated_ids: torch.Tensor, input_len: int) -> str:
    """Strip prompt tokens and assistant prefixes from generated output."""
    new_tokens = generated_ids[:, input_len:]
    if new_tokens.numel() == 0:
        return ""
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    prefixes = [
        "system\nYou are a helpful assistant.\nuser\n",
        "You are a helpful assistant.\nuser\n",
        "You are a helpful assistant.\n",
    ]
    for prefix in prefixes:
        if decoded.startswith(prefix):
            decoded = decoded[len(prefix) :]
            break
    for role in ("assistant\n", "assistant: ", "assistant "):
        if decoded.startswith(role):
            decoded = decoded[len(role) :]
            break
    return decoded.lstrip()


def _generate_markdown(
    processor,
    model,
    image: Image.Image,
    prompt: str,
    move_inputs: bool,
    device: Optional[torch.device],
    gen_kwargs: Dict[str, Any],
) -> tuple[str, float]:
    """Run model.generate for one image and return decoded markdown + elapsed seconds."""
    inputs = build_inputs(processor, image, prompt, device if move_inputs else None, move_to_device=move_inputs)
    start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.perf_counter() - start
    decoded = _decode_generation(processor, generated_ids, inputs["input_ids"].shape[-1])
    return decoded, elapsed


_WORKER_STATE: Dict[str, Any] = {}


def _init_worker(
    checkpoint: str,
    use_slow_processor: bool,
    device_str: Optional[str],
    dtype_str: str,
    device_map: Optional[Any],
    tokenizer: str,
    prompt: str,
    gen_kwargs: Dict[str, Any],
) -> None:
    """Load model/processor once per worker process."""
    global _WORKER_STATE

    device = torch.device(device_str) if device_str else None
    dtype = getattr(torch, dtype_str)

    processor = AutoProcessor.from_pretrained(
        tokenizer,
        trust_remote_code=True,
        use_fast=not use_slow_processor,
    )
    ensure_bbox_special_tokens(processor)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device is not None and device.type != "cuda":
        model.to(device)

    move_inputs = should_move_to_single_device(model)
    _WORKER_STATE = {
        "processor": processor,
        "model": model,
        "device": device,
        "move_inputs": move_inputs,
        "prompt": prompt,
        "gen_kwargs": gen_kwargs,
    }
    logger.info("Worker ready (move_inputs=%s, device=%s)", move_inputs, device)


def _process_image_task(task: tuple[Path, Path, Path]) -> Dict[str, Any]:
    """Worker entrypoint for multiprocessing."""
    global _WORKER_STATE
    img_path, input_root, output_root = task
    rel_path = _relative_to_root(img_path, input_root)
    out_path = output_root / rel_path.with_suffix(".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        image = Image.open(img_path).convert("RGB")
        decoded, elapsed = _generate_markdown(
            _WORKER_STATE["processor"],
            _WORKER_STATE["model"],
            image,
            _WORKER_STATE["prompt"],
            _WORKER_STATE["move_inputs"],
            _WORKER_STATE["device"],
            _WORKER_STATE["gen_kwargs"],
        )
        out_path.write_text(decoded, encoding="utf-8")
        _save_ground_truth_image(image, out_path)
        _save_input_image(image, out_path)
        return {"image": str(img_path), "output": str(out_path), "elapsed": elapsed, "error": None}
    except Exception as exc:  # pragma: no cover - pass through exceptions to main process cleanly
        logger.exception("Failed to process %s", img_path)
        return {"image": str(img_path), "output": str(out_path), "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference over all images in a directory with an OlmOCR Qwen2.5-VL checkpoint.")
    parser.add_argument("--config", default="configs/kolmocr_infer.yaml", help="YAML file with inference defaults.")
    parser.add_argument("--checkpoint", help="Path to checkpoint directory (with model + processor).")
    parser.add_argument("--tokenizer", help="Tokenizer/processor path (required; can differ from checkpoint).")
    parser.add_argument("--input-dir", help="Directory containing images (will scan recursively).")
    parser.add_argument("--output-dir", help="Directory to write generated markdown files (mirrors input layout).")
    parser.add_argument("--prompt", help="Instruction text for the model.")
    parser.add_argument("--max-new-tokens", type=int, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 for greedy).")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for nucleus sampling.")
    parser.add_argument("--use-slow-processor", action="store_true", help="Force slow processor to match older behavior.")
    parser.add_argument(
        "--device-map",
        default=None,
        help='HuggingFace device map (e.g., "auto", "balanced", "balanced_low_0"). Defaults to "auto" when CUDA is available.',
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of processes for inference. Each worker loads its own model (overrides config).",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    file_cfg = {}
    if config_path and config_path.exists():
        logger.info("Loading config from %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
    elif config_path:
        logger.warning("Config file %s not found. Falling back to CLI args/defaults.", config_path)

    def pick(key, cli_value, default=None):
        return cli_value if cli_value is not None else file_cfg.get(key, default)

    checkpoint = pick("checkpoint", args.checkpoint)
    tokenizer = pick("tokenizer", args.tokenizer)
    input_dir = pick("input_dir", args.input_dir)
    output_dir = pick("output_dir", args.output_dir)
    prompt = pick("prompt", args.prompt, "Read this document.")
    max_new_tokens = pick("max_new_tokens", args.max_new_tokens, 1024)
    device_map_arg = pick("device_map", args.device_map)
    num_workers = pick("num_workers", args.num_workers)
    if num_workers is None:
        num_workers = 1

    if not checkpoint:
        raise ValueError("checkpoint is required (set in --checkpoint or config).")
    if not tokenizer:
        raise ValueError("tokenizer is required (set in --tokenizer or config).")
    if not input_dir:
        raise ValueError("input_dir is required (set in --input-dir or config).")
    if not output_dir:
        raise ValueError("output_dir is required (set in --output-dir or config).")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error("Model checkpoint path not found: %s", checkpoint_path.resolve())
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not checkpoint_path.is_dir():
        logger.error("Model checkpoint path is not a directory: %s", checkpoint_path.resolve())
        raise NotADirectoryError(f"Checkpoint path is not a directory: {checkpoint_path}")

    if not input_dir.exists():
        logger.error("Input path not found: %s", input_dir.resolve())
        raise FileNotFoundError(f"Input path not found: {input_dir}")
    if not (input_dir.is_dir() or input_dir.is_file()):
        logger.error("Input path is neither file nor directory: %s", input_dir.resolve())
        raise FileNotFoundError(f"Input path must be a file or directory: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    dtype = torch.bfloat16 if has_cuda else torch.float32
    device_map = device_map_arg if device_map_arg is not None else ("auto" if has_cuda else None)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "temperature": args.temperature if args.temperature > 0.0 else None,
        "top_p": args.top_p,
    }
    # Drop None values to satisfy generate()
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    images_iter, rel_root = list_images_and_root(input_dir)
    images = list(images_iter)
    if not images:
        logger.warning("No images found under %s", input_dir)
        return
    total_images = len(images)
    total_start = time.perf_counter()

    if num_workers <= 1:
        proc_src = tokenizer
        logger.info("Loading processor from %s", proc_src)
        processor = AutoProcessor.from_pretrained(
            proc_src,
            trust_remote_code=True,
            use_fast=not args.use_slow_processor,
        )
        ensure_bbox_special_tokens(processor)

        logger.info("Loading model from %s (device=%s, dtype=%s)", checkpoint, device, dtype)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if not has_cuda:
            model.to(device)

        move_inputs = should_move_to_single_device(model)

        logger.info("Found %d images. Starting generation (single process)...", total_images)
        for idx, img_path in enumerate(images, start=1):
            rel_path = _relative_to_root(img_path, rel_root)
            out_path = output_dir / rel_path.with_suffix(".md")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Processing %s/%s: %s -> %s", idx, total_images, img_path, out_path)
            image = Image.open(img_path).convert("RGB")
            decoded, elapsed = _generate_markdown(
                processor, model, image, prompt, move_inputs, device if move_inputs else None, gen_kwargs
            )
            out_path.write_text(decoded, encoding="utf-8")
            _save_ground_truth_image(image, out_path)
            _save_input_image(image, out_path)
            logger.info("Finished %s/%s (%s) in %.2f s", idx, total_images, img_path.name, elapsed)
    else:
        if has_cuda:
            logger.warning("num_workers>1 with CUDA will load the model per process. Ensure you have enough GPU memory or pin workers to devices.")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set elsewhere

        logger.info("Found %d images. Starting generation with %d workers...", len(images), num_workers)
        device_str = str(device) if device is not None else None
        dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float32"

        with mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(
                str(checkpoint_path),
                args.use_slow_processor,
                device_str,
                dtype_str,
                device_map,
                str(tokenizer),
                prompt,
                gen_kwargs,
            ),
        ) as pool:
            completed = 0
            for result in pool.imap_unordered(_process_image_task, [(img, rel_root, output_dir) for img in images]):
                completed += 1
                if result["error"]:
                    logger.error("Failed %s -> %s: %s", result["image"], result["output"], result["error"])
                else:
                    logger.info(
                        "Finished %s/%s: %s -> %s (%.2f s)",
                        completed,
                        total_images,
                        result["image"],
                        result["output"],
                        result["elapsed"],
                    )

    total_elapsed = time.perf_counter() - total_start
    logger.info("Done. Processed %d images in %.2f s. Markdown files written to %s", total_images, total_elapsed, output_dir)


if __name__ == "__main__":
    main()
