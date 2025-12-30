import argparse
import atexit
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import time
import subprocess
import sys
from io import BytesIO
from urllib.request import urlopen
from urllib.parse import urlparse
import os
import shutil
import re
import json

from PIL import Image, ImageDraw
from openai import OpenAI
import yaml

from olmocr.prompts import PageResponse, build_no_anchoring_v4_yaml_prompt, build_no_anchoring_v4_yaml_prompt_with_bbox_wo_frontmatter, build_no_anchoring_v4_yaml_prompt_with_bbox_wo_frontmatter_for_qwen
from olmocr.train.dataloader import FrontMatterParser
from lab.utils.bbox.bbox_utils import scale_bboxes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
MAX_PAGE_RETRIES = 8
MODEL_MAX_CONTEXT = 16384


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
    images = (
        p
        for p in sorted(input_path.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in exts
        and not any(parent.name in {"images", "imgs"} for parent in p.parents)  # Skip nested images/ folders
    )
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


def _resize_longest_side(image: Image.Image, target_longest: Optional[int]) -> Image.Image:
    """Downscale image so its longest side matches target_longest (no upscaling)."""
    if not target_longest or target_longest <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= target_longest:
        return image
    scale = target_longest / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.LANCZOS)


def _rotate_image(image: Image.Image, rotation_degrees: int) -> Image.Image:
    if rotation_degrees == 0:
        return image
    if rotation_degrees == 90:
        transpose = Image.Transpose.ROTATE_90
    elif rotation_degrees == 180:
        transpose = Image.Transpose.ROTATE_180
    else:
        transpose = Image.Transpose.ROTATE_270
    return image.transpose(transpose)


def _copy_images_dir(src_path: Path, rel_root: Path, output_root: Path) -> None:
    """If an `images` subfolder exists next to the source, mirror-copy it to the output."""
    src_images = src_path.parent / "images"
    if not src_images.exists() or not src_images.is_dir():
        return
    dest_images = output_root / _relative_to_root(src_images, rel_root)
    shutil.copytree(src_images, dest_images, dirs_exist_ok=True)


def _strip_front_matter(text: str) -> str:
    """Remove YAML front-matter (--- ... ---) if present at the top of the document."""
    if not text.lstrip().startswith("---"):
        return text
    # Match leading block starting with --- and ending with --- on its own line.
    pattern = r"^---\s*\n.*?\n---\s*\n"
    cleaned = re.sub(pattern, "", text, count=1, flags=re.DOTALL)
    return cleaned.lstrip("\n")


def _normalize_bbox_markers(text: str) -> str:
    """Replace bbox HTML comments with canonical tokens so they stay atomic."""
    # Handle: <!-- id: b001 bbox: [x,y,w,h] --> → <!-- id: b001 bbox: <|box_start|> x,y,w,h <|box_end|> -->
    text = re.sub(
        r'(<!-- (?:id: \S+ )?bbox: )\[([^\]]+)\]( -->)',
        r'\1<|box_start|> \2 <|box_end|>\3',
        text
    )
    # Handle: <!-- bbox_blk_end --> → [BBOX_BLK_END]
    text = text.replace("<!-- bbox_blk_end -->", " [BBOX_BLK_END] ")
    return text


def _denormalize_bbox_markers(text: str) -> str:
    """Replace canonical tokens back to bbox HTML comments as requested by user."""
    # Handle: <!-- id: b001 bbox: <|box_start|> x,y,w,h <|box_end|> --> → <!-- id: b001 bbox: [x,y,w,h] -->
    text = re.sub(
        r'(<!-- (?:id: \S+ )?bbox: )<\|box_start\|> ([^<]+?) <\|box_end\|>( -->)',
        r'\1[\2]\3',
        text
    )
    # Handle: [BBOX_BLK_END] → <!-- bbox_blk_end -->
    text = text.replace(" [BBOX_BLK_END] ", "<!-- bbox_blk_end -->")
    return text


def _copy_ground_truth_md(img_path: Path, rel_root: Path, output_root: Path) -> Optional[Tuple[Path, str]]:
    """Copy existing GT .md/.html next to the image into output with `_gt.md` suffix."""
    gt_md_src = img_path.with_suffix(".md")
    gt_html_src = img_path.with_suffix(".html")
    if gt_md_src.exists():
        content = gt_md_src.read_text(encoding="utf-8", errors="ignore")
    elif gt_html_src.exists():
        content = gt_html_src.read_text(encoding="utf-8", errors="ignore")
    else:
        return None

    rel_path = _relative_to_root(img_path, rel_root)
    dest = output_root / rel_path.with_name(rel_path.stem + "_gt.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    stripped_content = _strip_front_matter(content)
    dest.write_text(stripped_content, encoding="utf-8")
    return dest, stripped_content


def _parse_bboxes_from_md(md_text: str) -> list[tuple[float, float, float, float]]:
    """Extract bbox tuples written as 'x0,y0,x1,y1'."""
    # 1. Match bboxes inside special tokens or HTML comments (anywhere in text)
    # Supports both <!-- bbox: [...] --> and <!-- [...] --> formats
    token_pattern = re.compile(r"(?:<\|box_start\|>|<!--\s*(?:bbox:\s*)?\[)\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:<\|box_end\|>|\]\s*-->)")
    # 2. Match bboxes on their own line (legacy/fallback)
    line_pattern = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*$", re.MULTILINE)
    
    bboxes: list[tuple[float, float, float, float]] = []
    # Use a set to avoid duplicates if both patterns match the same line
    seen = set()
    
    for match in token_pattern.finditer(md_text):
        bbox = tuple(float(match.group(i)) for i in range(1, 5))
        if bbox not in seen:
            bboxes.append(bbox)
            seen.add(bbox)
    
    for match in line_pattern.finditer(md_text):
        bbox = tuple(float(match.group(i)) for i in range(1, 5))
        if bbox not in seen:
            bboxes.append(bbox)
            seen.add(bbox)
            
    return bboxes


def _draw_bboxes_red(image: Image.Image, bboxes: Iterable[tuple[int, int, int, int]]) -> Image.Image:
    """Draw red bounding boxes on a copy of the image."""
    draw = ImageDraw.Draw(image)
    for box in bboxes:
        draw.rectangle(box, outline=(255, 0, 0), width=2)
    return image


def _save_bbox_visual(image: Image.Image, md_text: str, md_output_path: Path, already_normalized: bool = True) -> Optional[Path]:
    """Save an annotated PNG with bboxes drawn in red if any are found in markdown."""
    raw_bboxes = _parse_bboxes_from_md(md_text)
    if not raw_bboxes:
        return None
    scaled = scale_bboxes(raw_bboxes, image.size, already_normalized=already_normalized)
    annotated = _draw_bboxes_red(image.copy(), scaled)
    out_path = md_output_path.with_name(md_output_path.stem + "_bbox.png")
    annotated.save(out_path, format="PNG")
    return out_path


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


def _image_to_data_url(image: Image.Image) -> str:
    """Encode an RGB PIL image to a base64 PNG data URL."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    import base64  # local import to keep top-level deps minimal

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _build_messages_vllm(image: Image.Image, prompt: str) -> list[Dict[str, Any]]:
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


def _generate_markdown_vllm(
    client: OpenAI,
    model: str,
    image: Image.Image,
    prompt: str,
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    skip_special_tokens: bool = True,
) -> tuple[str, float, Optional[int], Optional[str]]:
    """Call the vLLM server for one image and return decoded markdown + elapsed seconds."""
    messages = _build_messages_vllm(image, prompt)
    temp_val = 0.0 if temperature is None else temperature
    top_p_val = 1.0 if top_p is None else top_p
    start = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temp_val,
        top_p=top_p_val,
        extra_body={"skip_special_tokens": skip_special_tokens},
    )
    elapsed = time.perf_counter() - start

    choice = completion.choices[0].message.content if completion and completion.choices else ""
    if isinstance(choice, list):
        generated_text = "".join(part.get("text", "") for part in choice if isinstance(part, dict))
    else:
        generated_text = choice or ""
    total_tokens = getattr(completion, "usage", None)
    total_tokens_val = total_tokens.total_tokens if total_tokens else None
    finish_reason = completion.choices[0].finish_reason if completion and completion.choices else None
    return generated_text, elapsed, total_tokens_val, finish_reason


def _load_transformers_model(
    checkpoint: str,
    processor_path: Optional[str],
    attn_implementation: Optional[str],
) -> tuple[Any, Any]:
    """Load HF model/processor for local inference with AutoProcessor normalization."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor_source = processor_path or checkpoint
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        **load_kwargs,
    )
    model.eval()
    return model, processor


def _generate_markdown_transformers(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    skip_special_tokens: bool = True,
) -> tuple[str, float, Optional[int], Optional[str]]:
    """Run local HF inference so AutoProcessor normalization is applied client-side."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_tokens}
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.perf_counter() - start

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, prompt_len:]
    generated_text = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=skip_special_tokens)[0]
    total_tokens = int(prompt_len + new_tokens.shape[1])
    finish_reason = "stop" if new_tokens.shape[1] < max_tokens else "length"
    return generated_text, elapsed, total_tokens, finish_reason


def _parse_page_response(markdown: str) -> PageResponse:
    parser = FrontMatterParser(front_matter_class=PageResponse)
    front_matter, text = parser._extract_front_matter_and_text(markdown)
    try:
        return parser._parse_front_matter(front_matter, text)
    except ValueError as e:
        logger.warning("Failed to parse front matter, using fallback: %s", e)
        # Fallback for custom prompts or models that don't output the expected YAML front matter
        return PageResponse(
            primary_language=front_matter.get("primary_language"),
            is_rotation_valid=front_matter.get("is_rotation_valid", True),
            rotation_correction=front_matter.get("rotation_correction", 0),
            is_table=front_matter.get("is_table", False),
            is_diagram=front_matter.get("is_diagram", False),
            natural_text=text,
        )


def _generate_with_retries(
    generate_fn,
    image: Image.Image,
    prompt: str,
    max_tokens: int,
    top_p: Optional[float],
) -> tuple[str, float]:
    attempt = 0
    exponential_backoffs = 0
    cumulative_rotation = 0
    total_elapsed = 0.0

    while attempt < MAX_PAGE_RETRIES:
        lookup_attempt = min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)
        temperature = TEMPERATURE_BY_ATTEMPT[lookup_attempt]
        rotated = _rotate_image(image, cumulative_rotation)

        try:
            decoded, elapsed, total_tokens, finish_reason = generate_fn(rotated, prompt, max_tokens, temperature, top_p)
            total_elapsed += elapsed

            if total_tokens is not None and total_tokens > MODEL_MAX_CONTEXT:
                raise ValueError(f"Response exceeded model_max_context of {MODEL_MAX_CONTEXT}")
            if finish_reason is not None and finish_reason != "stop":
                raise ValueError("Response did not finish with reason code 'stop'")

            page_response = _parse_page_response(decoded)
            if not page_response.is_rotation_valid and attempt < MAX_PAGE_RETRIES - 1:
                cumulative_rotation = (cumulative_rotation + page_response.rotation_correction) % 360
                raise ValueError("invalid_page rotation")

            return decoded, total_elapsed
        except (ConnectionError, OSError, TimeoutError) as exc:
            sleep_delay = 10 * (2**exponential_backoffs)
            exponential_backoffs += 1
            logger.warning("Client error on attempt %d: %s; sleeping %d seconds", attempt, exc, sleep_delay)
            time.sleep(sleep_delay)
        except Exception as exc:
            logger.warning("Retryable error on attempt %d: %s", attempt, exc)
            attempt += 1

    raise RuntimeError(f"Failed to process image after {MAX_PAGE_RETRIES} attempts.")


_WORKER_STATE: Dict[str, Any] = {}


def _process_single_image(
    img_path: Path,
    rel_root: Path,
    output_root: Path,
    generate_fn,
    prompt: str,
    max_tokens: int,
    top_p: Optional[float],
    target_longest_image_dim: Optional[int],
    skip_special_tokens: bool,
    replace_bbox_special: bool,
    is_normalized_bbox: bool,
) -> tuple[str, float]:
    """Process a single image: load, resize, generate, save outputs.

    Returns:
        Tuple of (output_md_path, elapsed_time)
    """
    rel_path = _relative_to_root(img_path, rel_root)
    out_path = output_root / rel_path.with_suffix(".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and resize image
    original = Image.open(img_path).convert("RGB")
    image = _resize_longest_side(original, target_longest_image_dim)
    
    # Generate markdown
    decoded, elapsed = _generate_with_retries(
        generate_fn,
        image,
        prompt,
        max_tokens,
        top_p,
    )
    
    # Process output text
    cleaned = _strip_front_matter(decoded)
    normalized = _normalize_bbox_markers(cleaned)
    
    final_text = normalized
    if not skip_special_tokens and replace_bbox_special:
        final_text = _denormalize_bbox_markers(normalized)
    
    # Save all outputs
    out_path.write_text(final_text, encoding="utf-8")
    _save_ground_truth_image(original, out_path)
    _save_input_image(image, out_path)
    _copy_images_dir(img_path, rel_root, output_root)
    
    # Process ground truth
    gt_info = _copy_ground_truth_md(img_path, rel_root, output_root)
    if gt_info:
        gt_path, gt_content = gt_info
        _save_bbox_visual(original, gt_content, gt_path, already_normalized=True)

    # Save prediction bbox visualization
    _save_bbox_visual(original, cleaned, out_path, already_normalized=is_normalized_bbox)

    return str(out_path), elapsed


def _init_worker(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    api_base: Optional[str],
    api_key: Optional[str],
    target_longest_image_dim: Optional[int],
    skip_special_tokens: bool,
    replace_bbox_special: bool,
    is_normalized_bbox: bool,
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
        "target_longest_image_dim": target_longest_image_dim,
        "skip_special_tokens": skip_special_tokens,
        "replace_bbox_special": replace_bbox_special,
        "is_normalized_bbox": is_normalized_bbox,
    }
    logger.info("Worker ready for vLLM (%s)", api_base)


def _process_image_task(task: tuple[Path, Path, Path]) -> Dict[str, Any]:
    """Worker entrypoint for multiprocessing."""
    global _WORKER_STATE
    img_path, input_root, output_root = task

    try:
        generate_fn = lambda img, prompt, max_tokens, temperature, top_p: _generate_markdown_vllm(
            _WORKER_STATE["client"],
            _WORKER_STATE["model"],
            img,
            prompt,
            max_tokens,
            temperature,
            top_p,
            _WORKER_STATE["skip_special_tokens"],
        )
        
        out_path, elapsed = _process_single_image(
            img_path,
            input_root,
            output_root,
            generate_fn,
            _WORKER_STATE["prompt"],
            _WORKER_STATE["max_tokens"],
            _WORKER_STATE["top_p"],
            _WORKER_STATE["target_longest_image_dim"],
            _WORKER_STATE["skip_special_tokens"],
            _WORKER_STATE["replace_bbox_special"],
            _WORKER_STATE["is_normalized_bbox"],
        )
        
        return {"image": str(img_path), "output": out_path, "elapsed": elapsed, "error": None}
    except Exception as exc:  # pragma: no cover - pass through exceptions to main process cleanly
        logger.exception("Failed to process %s", img_path)
        rel_path = _relative_to_root(img_path, input_root)
        out_path = output_root / rel_path.with_suffix(".md")
        return {"image": str(img_path), "output": str(out_path), "error": str(exc)}


def _get_available_prompt_functions() -> list[str]:
    """Get list of available prompt functions from olmocr.prompts."""
    from olmocr import prompts
    return [name for name in dir(prompts) if name.startswith('build_') and callable(getattr(prompts, name))]


def _load_prompt_function(function_name: str) -> str:
    """Dynamically load and call a prompt function from olmocr.prompts."""
    try:
        from olmocr import prompts

        if not hasattr(prompts, function_name):
            available_funcs = _get_available_prompt_functions()
            raise ValueError(
                f"Prompt function '{function_name}' not found in olmocr.prompts. "
                f"Available functions: {', '.join(available_funcs)}"
            )

        func = getattr(prompts, function_name)
        if not callable(func):
            raise ValueError(f"'{function_name}' is not a callable function")

        return func()
    except Exception as e:
        logger.error("Failed to load prompt function '%s': %s", function_name, e)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference over images using vLLM or local transformers.")
    parser.add_argument("--config", default="configs/inference/transformer.yaml", help="YAML file with inference defaults.")
    parser.add_argument("--checkpoint", help="Path to local model directory (HF format). If not provided, --hf-model is used.")
    parser.add_argument("--hf-model", dest="hf_model", help="Hugging Face model ID (e.g., 'Qwen/Qwen2.5-VL-7B-Instruct'). Used when --checkpoint is not provided.")
    parser.add_argument("--tokenizer", help="Tokenizer path (optional; defaults to checkpoint or hf_model).")
    parser.add_argument("--input-dir", help="Directory containing images (will scan recursively).")
    parser.add_argument("--output-dir", help="Directory to write generated markdown files (mirrors input layout).")
    parser.add_argument("--prompt", help="Instruction text for the model.")
    parser.add_argument(
        "--prompt-function",
        dest="prompt_function",
        help="Name of prompt function from olmocr.prompts (e.g., 'build_no_anchoring_v4_yaml_prompt_with_bbox_wo_frontmatter')"
    )
    parser.add_argument("--max-new-tokens", type=int, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (0 for greedy; defaults to 0.7).")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p for nucleus sampling (defaults to 0.9).")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of processes for inference. Each worker loads its own model (overrides config).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Limit the number of GPUs used for tensor-parallel dispatch (HF device_map auto).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of each GPU memory to allow when dispatching the model (0-1).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model sequence length (defaults to model's max).",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Attention backend for transformers (e.g., flash_attention_2).",
    )
    parser.add_argument(
        "--target-longest-image-dim",
        type=int,
        default=None,
        help="Resize input images so the longest side matches this value (no upscaling).",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        dest="skip_special_tokens",
        default=False,
        help="If set, skip_special_tokens will be set to True in vLLM (special tokens will not be generated).",
    )
    parser.add_argument(
        "--replace-bbox-special",
        action="store_false",
        dest="replace_bbox_special",
        default=True,
        help="If set, special tokens will NOT be converted back to HTML comments (keeps raw special tokens in output).",
    )
    parser.add_argument(
        "--is-normalized-bbox",
        action="store_true",
        dest="is_normalized_bbox",
        default=False,
        help="If set, bboxes in markdown are already normalized to image scale. Otherwise, normalize from 1000x1000.",
    )
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible base URL of vLLM server (defaults to http://localhost:8000/v1).")
    parser.add_argument("--api-key", default=None, help="API key for vLLM server (defaults to EMPTY).")
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers"],
        default=None,
        help="Inference backend: vllm (OpenAI-compatible server) or transformers (local HF).",
    )
    parser.add_argument("--launch-vllm", dest="launch_vllm", action="store_true", help="Launch a local vLLM server for the given model.")
    parser.add_argument("--no-launch-vllm", dest="launch_vllm", action="store_false", help="Do not launch vLLM even if config enables it.")
    parser.set_defaults(launch_vllm=None)
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
    hf_model = pick("hf_model", args.hf_model)
    tokenizer = pick("tokenizer", args.tokenizer)
    input_dir = pick("input_dir", args.input_dir)
    output_dir = pick("output_dir", args.output_dir)

    # Prompt selection: prompt_function is REQUIRED
    prompt_function = pick("prompt_function", args.prompt_function, None)
    if prompt_function is None:
        available_funcs = _get_available_prompt_functions()
        raise ValueError(
            "prompt_function is required but not specified in config or CLI.\n"
            f"Available prompt functions in olmocr/prompts/prompts.py:\n"
            f"{chr(10).join('  - ' + func for func in available_funcs)}\n"
            f"Please add 'prompt_function: <function_name>' to your YAML config."
        )

    logger.info("Loading prompt from function: %s", prompt_function)
    prompt = _load_prompt_function(prompt_function)

    max_new_tokens = pick("max_new_tokens", args.max_new_tokens, 1024)
    temperature = pick("temperature", args.temperature, 0.7)
    top_p = pick("top_p", args.top_p, 0.9)
    num_workers = pick("num_workers", args.num_workers)
    max_model_len = pick("max_model_len", args.max_model_len, None)
    tensor_parallel_size = pick("tensor_parallel_size", args.tensor_parallel_size, None)
    gpu_memory_utilization = pick("gpu_memory_utilization", args.gpu_memory_utilization, None)
    api_base = pick("api_base", args.api_base, "http://localhost:8000/v1")
    api_key = pick("api_key", args.api_key, "EMPTY")
    backend = pick("backend", args.backend, "vllm")
    attn_implementation = pick("attn_implementation", args.attn_implementation, None)
    launch_vllm = pick("launch_vllm", args.launch_vllm, False)
    target_longest_image_dim = pick("target_longest_image_dim", args.target_longest_image_dim, None)
    skip_special_tokens = pick("skip_special_tokens", args.skip_special_tokens, False)
    replace_bbox_special = pick("replace_bbox_special", args.replace_bbox_special, True)
    is_normalized_bbox = pick("is_normalized_bbox", args.is_normalized_bbox, False)
    if num_workers is None:
        num_workers = 1

    # Use checkpoint if available, otherwise use hf_model
    model_source = checkpoint if checkpoint else hf_model
    if not model_source:
        raise ValueError("Either checkpoint (local path) or hf_model (Hugging Face model ID) is required.")
    if not input_dir:
        raise ValueError("input_dir is required (set in --input-dir or config).")
    if not output_dir:
        raise ValueError("output_dir is required (set in --output-dir or config).")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Only validate local path if checkpoint is provided (local path)
    # If using hf_model (HuggingFace ID), skip path validation
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if backend == "transformers" or launch_vllm:
            if not checkpoint_path.exists():
                logger.error("Model checkpoint path not found: %s", checkpoint_path.resolve())
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            if not checkpoint_path.is_dir():
                logger.error("Model checkpoint path is not a directory: %s", checkpoint_path.resolve())
                raise NotADirectoryError(f"Checkpoint path is not a directory: {checkpoint_path}")
    else:
        # Using HF model ID, no path validation needed
        logger.info("Using Hugging Face model ID: %s", hf_model)

    if not input_dir.exists():
        logger.error("Input path not found: %s", input_dir.resolve())
        raise FileNotFoundError(f"Input path not found: {input_dir}")
    if not (input_dir.is_dir() or input_dir.is_file()):
        logger.error("Input path is neither file nor directory: %s", input_dir.resolve())
        raise FileNotFoundError(f"Input path must be a file or directory: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "vllm":
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"

    images_iter, rel_root = list_images_and_root(input_dir)
    images = list(images_iter)
    total_images = len(images)
    if not images:
        logger.warning("No images found under %s", input_dir)
        return
    
    logger.info("Found %d images under %s", total_images, input_dir)
    total_start = time.perf_counter()

    # Dictionary to store timing data per document
    timing_data: Dict[str, float] = {}

    # Only vLLM backend is supported
    if backend != "vllm":
        raise ValueError(f"Only vLLM backend is supported. Got backend='{backend}'. Please set backend='vllm' in config or use --backend vllm.")

    vllm_proc = None
    client = None
    if backend == "vllm":
        if launch_vllm:
            parsed = urlparse(api_base)
            if parsed.hostname not in {"localhost", "127.0.0.1", None}:
                raise ValueError("--launch-vllm is only supported for local api_base (localhost).")
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(model_source),
                "--port",
                str(port),
                "--tensor-parallel-size",
                str(tensor_parallel_size or 1),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization if gpu_memory_utilization is not None else 0.9),
            ]
            if tokenizer:
                cmd.extend(["--tokenizer", str(tokenizer)])
            if max_model_len is not None:
                cmd.extend(["--max-model-len", str(max_model_len)])
            cmd.append("--trust-remote-code")
            env = os.environ.copy()
            if "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = "0,1"
            
            vllm_log_path = "vllm_server.log"
            vllm_log_file = open(vllm_log_path, "w", encoding="utf-8")
            logger.info("Launching vLLM server: %s", " ".join(cmd))
            logger.info("Using CUDA_VISIBLE_DEVICES=%s", env.get("CUDA_VISIBLE_DEVICES"))
            logger.info("vLLM logs will be written to %s", vllm_log_path)
            
            vllm_proc = subprocess.Popen(cmd, env=env, stdout=vllm_log_file, stderr=subprocess.STDOUT)
            atexit.register(lambda: vllm_proc.terminate() if vllm_proc and vllm_proc.poll() is None else None)
            atexit.register(vllm_log_file.close)
            _wait_for_vllm(api_base)
        else:
            logger.info("Waiting for existing vLLM server at %s", api_base)
            _wait_for_vllm(api_base)

        client = OpenAI(base_url=api_base, api_key=api_key)
        logger.info("Using vLLM server at %s with model=%s", api_base, model_source)

    if num_workers <= 1:
        logger.info("Found %d images. Starting generation (single process)...", total_images)
        for idx, img_path in enumerate(images, start=1):
            logger.info("Processing %s/%s: %s", idx, total_images, img_path)

            generate_fn = lambda img, prompt, max_tokens, temperature, top_p: _generate_markdown_vllm(
                client, model_source, img, prompt, max_tokens, temperature, top_p, skip_special_tokens
            )

            out_path, elapsed = _process_single_image(
                img_path,
                rel_root,
                output_dir,
                generate_fn,
                prompt,
                max_new_tokens,
                top_p,
                target_longest_image_dim,
                skip_special_tokens,
                replace_bbox_special,
                is_normalized_bbox,
            )

            # Store timing data using relative path
            rel_img_path = _relative_to_root(img_path, rel_root)
            timing_data[str(rel_img_path)] = elapsed

            logger.info("Finished %s/%s (%s) -> %s in %.2f s", idx, total_images, img_path.name, out_path, elapsed)
    else:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set elsewhere

        logger.info("Found %d images. Starting generation with %d workers...", len(images), num_workers)
        with mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(
                model_source,
                prompt,
                max_new_tokens,
                temperature,
                top_p,
                api_base,
                api_key,
                target_longest_image_dim,
                skip_special_tokens,
                replace_bbox_special,
                is_normalized_bbox,
            ),
        ) as pool:
            completed = 0
            for result in pool.imap_unordered(_process_image_task, [(img, rel_root, output_dir) for img in images]):
                completed += 1
                if result["error"]:
                    logger.error("Failed %s -> %s: %s", result["image"], result["output"], result["error"])
                else:
                    # Store timing data using relative path
                    img_path = Path(result["image"])
                    rel_img_path = _relative_to_root(img_path, rel_root)
                    timing_data[str(rel_img_path)] = result["elapsed"]

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

    # Save timing data to JSON file
    timing_json_path = output_dir / "time_per_document.json"
    with open(timing_json_path, "w", encoding="utf-8") as f:
        json.dump(timing_data, f, indent=2, ensure_ascii=False)
    logger.info("Timing data saved to %s", timing_json_path)


if __name__ == "__main__":
    main()
