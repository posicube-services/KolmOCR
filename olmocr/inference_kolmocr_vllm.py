import argparse
import atexit
import base64
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml
from openai import OpenAI
from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _extract_images_from_markdown(md_path: Path) -> Iterable[Path]:
    """Find image references in a markdown file and yield resolved paths."""
    import re

    md_image = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    html_image = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']')
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    matches = md_image.findall(text) + html_image.findall(text)
    for raw in matches:
        cleaned = raw.strip().strip('"').strip("'")
        yield (md_path.parent / cleaned).resolve()


def list_images_and_root(input_path: Path) -> Tuple[Iterable[Path], Path]:
    """Return iterable of images and the root used for relative output paths."""
    exts = {".png", ".jpg", ".jpeg"}
    if input_path.is_file() and input_path.suffix.lower() == ".md":
        return _extract_images_from_markdown(input_path), input_path.parent
    if input_path.is_file() and input_path.suffix.lower() in exts:
        return [input_path], input_path.parent
    images = (p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in exts)
    return images, input_path


def _relative_to_root(path: Path, root: Path) -> Path:
    """Return path relative to root, falling back to basename if not possible."""
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _image_to_data_url(image: Image.Image) -> str:
    """Encode an RGB PIL image to a base64 PNG data URL."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _build_messages(image: Image.Image, prompt: str) -> list[Dict[str, Any]]:
    """Create chat-formatted messages for OpenAI-compatible vLLM server."""
    data_url = _image_to_data_url(image)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _generate_markdown(
    client: OpenAI,
    model: str,
    image: Image.Image,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, float]:
    """Call the vLLM server for one image and return decoded markdown + elapsed seconds."""
    messages = _build_messages(image, prompt)
    start = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    elapsed = time.perf_counter() - start

    choice = completion.choices[0].message.content if completion and completion.choices else ""
    if isinstance(choice, list):
        generated_text = "".join(part.get("text", "") for part in choice if isinstance(part, dict))
    else:
        generated_text = choice or ""
    return generated_text, elapsed


def _save_input_image(image: Image.Image, md_output_path: Path) -> Path:
    """Save a copy of the input image next to the markdown with a `_input.png` suffix."""
    input_path = md_output_path.with_name(md_output_path.stem + "_input.png")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(input_path, format="PNG")
    return input_path


_WORKER_STATE: Dict[str, Any] = {}


def _init_worker(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    api_base: str,
    api_key: str,
) -> None:
    """Initialize OpenAI client once per worker process."""
    global _WORKER_STATE
    client = OpenAI(base_url=api_base, api_key=api_key)
    _WORKER_STATE = {
        "client": client,
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def _process_image_task(task: tuple[Path, Path, Path]) -> Dict[str, Any]:
    """Worker entrypoint for multiprocessing."""
    global _WORKER_STATE
    img_path, rel_root, output_root = task
    rel_path = _relative_to_root(img_path, rel_root)
    out_path = output_root / rel_path.with_suffix(".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        generated_text, elapsed = _generate_markdown(
            _WORKER_STATE["client"],
            _WORKER_STATE["model"],
            image,
            _WORKER_STATE["prompt"],
            _WORKER_STATE["max_tokens"],
            _WORKER_STATE["temperature"],
            _WORKER_STATE["top_p"],
        )
        out_path.write_text(generated_text, encoding="utf-8")
        _save_input_image(image, out_path)
        return {"image": str(img_path), "output": str(out_path), "elapsed": elapsed, "error": None}
    except Exception as exc:  # pragma: no cover - pass through exceptions to main process cleanly
        logger.exception("Failed to process %s", img_path)
        return {"image": str(img_path), "output": str(out_path), "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference against a vLLM OpenAI-compatible server for Qwen2.5-VL.")
    parser.add_argument("--config", default="configs/kolmocr_infer_vllm.yaml", help="YAML file with inference defaults.")
    parser.add_argument("--model", help="Model name served by vLLM (e.g., Qwen/Qwen2.5-VL-7B-Instruct).")
    parser.add_argument("--input-dir", help="Directory containing images (will scan recursively).")
    parser.add_argument("--output-dir", help="Directory to write generated markdown files (mirrors input layout).")
    parser.add_argument("--prompt", help="Instruction text for the model.")
    parser.add_argument("--max-new-tokens", type=int, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 for greedy).")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for nucleus sampling.")
    parser.add_argument("--api-base", default="http://localhost:8080/v1", help="OpenAI-compatible base URL of vLLM server.")
    parser.add_argument("--api-key", default="EMPTY", help="API key for server (use dummy if not required).")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of processes for inference (overrides config).")
    parser.add_argument("--launch-vllm", action="store_true", help="Launch a local vLLM server for the given model.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size when launching vLLM.")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    file_cfg: Dict[str, Any] = {}
    if config_path and config_path.exists():
        logger.info("Loading config from %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
    elif config_path:
        logger.warning("Config file %s not found. Falling back to CLI args/defaults.", config_path)

    def pick(key, cli_value, default=None):
        return cli_value if cli_value is not None else file_cfg.get(key, default)

    model = pick("model", args.model)
    input_dir = pick("input_dir", args.input_dir)
    output_dir = pick("output_dir", args.output_dir)
    prompt = pick("prompt", args.prompt, "Read this document.")
    max_new_tokens = pick("max_new_tokens", args.max_new_tokens, 1024)
    temperature = pick("temperature", args.temperature)
    top_p = pick("top_p", args.top_p)
    api_base = pick("api_base", args.api_base)
    api_key = pick("api_key", args.api_key)
    num_workers = pick("num_workers", args.num_workers, 1)
    api_base = api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"

    if not model:
        raise ValueError("model is required (set in --model or config).")
    if not input_dir:
        raise ValueError("input_dir is required (set in --input-dir or config).")
    if not output_dir:
        raise ValueError("output_dir is required (set in --output-dir or config).")

    def _wait_for_vllm(api_base_url: str, timeout: float = 120.0) -> None:
        """Poll the vLLM server until /models responds or timeout hits."""
        deadline = time.time() + timeout
        url = api_base_url.rstrip("/") + "/models"
        while time.time() < deadline:
            try:
                with urlopen(url, timeout=5):
                    return
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"vLLM server did not become ready within {timeout} seconds ({url}).")

    vllm_proc = None
    if args.launch_vllm:
        parsed = urlparse(api_base)
        if parsed.hostname not in {"localhost", "127.0.0.1", None}:
            raise ValueError("--launch-vllm is only supported for local api_base (localhost).")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
        ]
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
        logger.info("Launching vLLM server: %s", " ".join(cmd))
        vllm_proc = subprocess.Popen(cmd, env=env)
        atexit.register(lambda: vllm_proc.terminate() if vllm_proc and vllm_proc.poll() is None else None)
        _wait_for_vllm(api_base)
    else:
        _wait_for_vllm(api_base)

    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)

    if not input_dir_path.exists():
        logger.error("Input path not found: %s", input_dir_path.resolve())
        raise FileNotFoundError(f"Input path not found: {input_dir_path}")
    if not (input_dir_path.is_dir() or input_dir_path.is_file()):
        logger.error("Input path is neither file nor directory: %s", input_dir_path.resolve())
        raise FileNotFoundError(f"Input path must be a file or directory: {input_dir_path}")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=api_base, api_key=api_key)
    logger.info("Using vLLM server at %s with model=%s", api_base, model)

    images_iter, rel_root = list_images_and_root(input_dir_path)
    images = list(images_iter)
    if not images:
        logger.warning("No images found under %s", input_dir_path)
        return

    total_images = len(images)
    total_start = time.perf_counter()
    logger.info("Found %d images. Starting generation...", total_images)

    if num_workers <= 1:
        for idx, img_path in enumerate(images, start=1):
            rel_path = _relative_to_root(img_path, rel_root)
            out_path = output_dir_path / rel_path.with_suffix(".md")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                print(f"[progress] {idx}/{total_images} -> {rel_path}", flush=True)
                logger.info("Processing %s/%s: %s -> %s", idx, total_images, img_path, out_path)
                with Image.open(img_path) as img:
                    image = img.convert("RGB")
                generated_text, elapsed = _generate_markdown(
                    client, model, image, prompt, max_new_tokens, temperature, top_p
                )
                out_path.write_text(generated_text, encoding="utf-8")
                _save_input_image(image, out_path)
                logger.info("Finished %s/%s (%s) in %.2f s", idx, total_images, img_path.name, elapsed)
            except Exception:
                logger.exception("Failed to process %s", img_path)
    else:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set elsewhere

        logger.info("Starting multiprocessing with %d workers", num_workers)
        tasks = [(img, rel_root, output_dir_path) for img in images]
        with mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(model, prompt, max_new_tokens, temperature, top_p, api_base, api_key),
        ) as pool:
            completed = 0
            for result in pool.imap_unordered(_process_image_task, tasks):
                completed += 1
                if result["error"]:
                    logger.error(
                        "Failed %s/%s: %s -> %s (%s)", completed, total_images, result["image"], result["output"], result["error"]
                    )
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
    print(f"[progress] completed {total_images}/{total_images} files in {total_elapsed:.2f}s", flush=True)
    logger.info("Done. Processed %d images in %.2f s. Markdown files written to %s", total_images, total_elapsed, output_dir_path)


if __name__ == "__main__":
    main()
