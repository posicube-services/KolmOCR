"""
Usage examples (run from repo root):
  python scripts/run_kolmocr_infer_and_eval.py  --config configs/pipeline/bench_table.yaml --cuda-visible-devices 0,1
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def format_value(value, run_id: str):
    if isinstance(value, str):
        return value.format(run_id=run_id)
    return value


def resolve_path(value: Optional[str], run_id: str) -> Optional[Path]:
    if value is None:
        return None
    formatted = format_value(value, run_id)
    path = Path(formatted)
    return path if path.is_absolute() else REPO_ROOT / path


def copy_md_to_md(md_source_root: Path, md_target_root: Path) -> int:
    """Copy all .md files from source to target directory."""
    copied = 0
    for md_path in md_source_root.rglob("*.md"):
        rel = md_path.relative_to(md_source_root)
        target_path = md_target_root / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
        copied += 1
    return copied


def run_cmd(cmd, env=None):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run KolmOCR inference + evaluation as a single pipeline.")
    parser.add_argument("--config", default="configs/pipeline/bench_table.yaml", help="Pipeline YAML path.")
    parser.add_argument("--python", help="Python executable override (defaults to config or current python).")
    parser.add_argument("--run-id", dest="run_id", help="Override run_id for output formatting.")
    parser.add_argument(
        "--cuda-visible-devices",
        dest="cuda_visible_devices",
        help="CUDA_VISIBLE_DEVICES to use (e.g., '0,1,2,3'). Overrides config.",
    )

    inf_group = parser.add_argument_group("Inference overrides")
    inf_group.add_argument("--inf-script", help="Inference script path.")
    inf_group.add_argument("--inf-config", help="YAML config to pass to the inference script.")
    inf_group.add_argument("--inf-checkpoint", help="Local checkpoint path (HF format).")
    inf_group.add_argument("--inf-tokenizer", help="Tokenizer/processor path (optional; defaults to checkpoint or hf_model).")
    inf_group.add_argument("--inf-hf-model", help="Hugging Face model ID (e.g., 'Qwen/Qwen2.5-VL-7B-Instruct').")
    inf_group.add_argument("--inf-input-dir", help="Input directory for inference.")
    inf_group.add_argument("--inf-output-dir", help="Output directory for inference results.")
    inf_group.add_argument("--inf-prompt", help="Prompt text for inference.")
    inf_group.add_argument("--inf-prompt-function", help="Name of prompt function from olmocr.prompts to use for inference.")
    inf_group.add_argument("--inf-max-new-tokens", type=int, help="Max new tokens for generation.")
    inf_group.add_argument("--inf-temperature", type=float, help="Sampling temperature.")
    inf_group.add_argument("--inf-top-p", type=float, help="Top-p nucleus sampling.")
    inf_group.add_argument("--inf-num-workers", type=int, help="Number of inference workers.")
    inf_group.add_argument("--inf-api-base", help="vLLM API base URL.")
    inf_group.add_argument("--inf-api-key", help="vLLM API key.")
    inf_group.add_argument("--inf-launch-vllm", dest="inf_launch_vllm", action="store_true", help="Launch vLLM server from the pipeline.")
    inf_group.add_argument("--no-inf-launch-vllm", dest="inf_launch_vllm", action="store_false", help="Do not launch vLLM server.")
    inf_group.add_argument("--inf-tensor-parallel-size", type=int, help="Tensor parallel size for vLLM launch.")
    inf_group.add_argument("--inf-gpu-memory-utilization", type=float, help="GPU memory utilization for vLLM launch.")
    inf_group.add_argument("--inf-max-model-len", type=int, help="Maximum model sequence length for vLLM launch.")
    inf_group.add_argument(
        "--inf-target-longest-image-dim",
        type=int,
        help="Resize input images so the longest side matches this value (no upscaling).",
    )
    inf_group.add_argument(
        "--inf-replace-bbox-special",
        dest="inf_replace_bbox_special",
        action="store_true",
        help="Convert special tokens back to HTML comments in final output (requires skip_special_tokens=False).",
    )
    inf_group.add_argument(
        "--inf-is-normalized-bbox",
        dest="inf_is_normalized_bbox",
        action="store_true",
        help="Bboxes in markdown are already normalized to image scale (no 1000x1000 normalization needed).",
    )

    eval_group = parser.add_argument_group("Evaluation overrides")
    eval_group.add_argument("--eval-script", help="Evaluation script path.")
    eval_group.add_argument("--eval-config", help="YAML config to pass to the evaluation script.")
    eval_group.add_argument("--eval-pred-dir", help="Directory containing prediction markdown files.")
    eval_group.add_argument("--eval-gt-dir", help="Directory containing GT markdown files.")
    eval_group.add_argument("--eval-gt-md-dir", help="Directory containing GT markdown files to copy.")
    eval_group.add_argument("--eval-output-dir", help="Directory to store evaluation outputs.")
    eval_group.add_argument("--eval-metrics", nargs="+", help="Metrics list for evaluation.")
    eval_group.add_argument("--eval-threshold-headings", type=int, help="Heading edit distance threshold.")
    eval_group.add_argument("--eval-threshold-table", type=float, help="Table similarity threshold.")
    eval_group.add_argument("--eval-version", help="Model version string passed to evaluator.")
    eval_group.add_argument(
        "--eval-text-f1",
        dest="eval_text_include_f1",
        action="store_true",
        help="Include heading F1 columns.",
    )
    eval_group.add_argument(
        "--eval-no-text-f1",
        dest="eval_text_include_f1",
        action="store_false",
        help="Exclude heading F1 columns.",
    )

    parser.set_defaults(inf_launch_vllm=None, eval_text_include_f1=None)
    args = parser.parse_args()

    cfg_path = REPO_ROOT / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    run_id = args.run_id or cfg.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    python_bin = args.python or cfg.get("python") or sys.executable
    cuda_visible_devices = args.cuda_visible_devices or cfg.get("cuda_visible_devices")

    def override(cfg_dict, key, value):
        if value is not None:
            cfg_dict[key] = value

    # Inference stage
    inf_cfg = dict(cfg.get("inference", {}))
    override(inf_cfg, "script", args.inf_script)
    override(inf_cfg, "config", args.inf_config)
    override(inf_cfg, "checkpoint", args.inf_checkpoint)
    override(inf_cfg, "tokenizer", args.inf_tokenizer)
    override(inf_cfg, "hf_model", args.inf_hf_model)
    override(inf_cfg, "input_dir", args.inf_input_dir)
    override(inf_cfg, "output_dir", args.inf_output_dir)
    override(inf_cfg, "prompt", args.inf_prompt)
    override(inf_cfg, "prompt_function", args.inf_prompt_function)
    override(inf_cfg, "max_new_tokens", args.inf_max_new_tokens)
    override(inf_cfg, "temperature", args.inf_temperature)
    override(inf_cfg, "top_p", args.inf_top_p)
    override(inf_cfg, "num_workers", args.inf_num_workers)
    override(inf_cfg, "api_base", args.inf_api_base)
    override(inf_cfg, "api_key", args.inf_api_key)
    override(inf_cfg, "launch_vllm", args.inf_launch_vllm)
    override(inf_cfg, "tensor_parallel_size", args.inf_tensor_parallel_size)
    override(inf_cfg, "gpu_memory_utilization", args.inf_gpu_memory_utilization)
    override(inf_cfg, "max_model_len", args.inf_max_model_len)
    override(inf_cfg, "target_longest_image_dim", args.inf_target_longest_image_dim)
    override(inf_cfg, "replace_bbox_special", args.inf_replace_bbox_special)
    override(inf_cfg, "is_normalized_bbox", args.inf_is_normalized_bbox)

    inf_script = resolve_path(inf_cfg.get("script", "olmocr/inference_kolmocr.py"), run_id)
    checkpoint = resolve_path(inf_cfg.get("checkpoint"), run_id)
    tokenizer = resolve_path(inf_cfg.get("tokenizer"), run_id)
    hf_model = inf_cfg.get("hf_model")  # Hugging Face model ID
    input_dir = resolve_path(inf_cfg.get("input_dir"), run_id)
    output_dir = resolve_path(inf_cfg.get("output_dir"), run_id)
    if not inf_script or not input_dir or not output_dir or (checkpoint is None and hf_model is None):
        raise ValueError("Inference config must include script, input_dir, output_dir, and checkpoint or hf_model.")
    output_dir.mkdir(parents=True, exist_ok=True)

    inf_cmd = [
        python_bin,
        str(inf_script),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]
    if checkpoint is not None:
        inf_cmd.extend(["--checkpoint", str(checkpoint)])
    if tokenizer is not None:
        inf_cmd.extend(["--tokenizer", str(tokenizer)])
    if hf_model is not None:
        inf_cmd.extend(["--hf-model", str(format_value(hf_model, run_id))])
    if inf_cfg.get("config"):
        inf_cmd.extend(["--config", str(resolve_path(inf_cfg["config"], run_id))])
    if inf_cfg.get("prompt"):
        inf_cmd.extend(["--prompt", str(format_value(inf_cfg["prompt"], run_id))])
    if inf_cfg.get("prompt_function"):
        inf_cmd.extend(["--prompt-function", str(inf_cfg["prompt_function"])])
    if inf_cfg.get("max_new_tokens") is not None:
        inf_cmd.extend(["--max-new-tokens", str(inf_cfg["max_new_tokens"])])
    if inf_cfg.get("temperature") is not None:
        inf_cmd.extend(["--temperature", str(inf_cfg["temperature"])])
    if inf_cfg.get("top_p") is not None:
        inf_cmd.extend(["--top-p", str(inf_cfg["top_p"])])
    if inf_cfg.get("num_workers") is not None:
        inf_cmd.extend(["--num-workers", str(inf_cfg["num_workers"])])
    if inf_cfg.get("api_base"):
        inf_cmd.extend(["--api-base", str(format_value(inf_cfg["api_base"], run_id))])
    if inf_cfg.get("api_key"):
        inf_cmd.extend(["--api-key", str(format_value(inf_cfg["api_key"], run_id))])
    if inf_cfg.get("launch_vllm"):
        inf_cmd.append("--launch-vllm")
    elif inf_cfg.get("launch_vllm") is False:
        inf_cmd.append("--no-launch-vllm")
    if inf_cfg.get("tensor_parallel_size"):
        inf_cmd.extend(["--tensor-parallel-size", str(inf_cfg["tensor_parallel_size"])])
    if inf_cfg.get("gpu_memory_utilization"):
        inf_cmd.extend(["--gpu-memory-utilization", str(inf_cfg["gpu_memory_utilization"])])
    if inf_cfg.get("max_model_len"):
        inf_cmd.extend(["--max-model-len", str(inf_cfg["max_model_len"])])
    if inf_cfg.get("target_longest_image_dim"):
        inf_cmd.extend(["--target-longest-image-dim", str(inf_cfg["target_longest_image_dim"])])
    if inf_cfg.get("replace_bbox_special"):
        inf_cmd.append("--replace-bbox-special")
    if inf_cfg.get("is_normalized_bbox"):
        inf_cmd.append("--is-normalized-bbox")

    print(f"[pipeline] Running inference -> {output_dir}")
    # Set CUDA_VISIBLE_DEVICES if specified
    inf_env = None
    if cuda_visible_devices:
        import os

        inf_env = os.environ.copy()
        inf_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(f"[pipeline] Using CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    run_cmd(inf_cmd, env=inf_env)

    # Prepare GT (.html -> .md) if requested
    eval_cfg = dict(cfg.get("evaluate", {}))
    override(eval_cfg, "script", args.eval_script)
    override(eval_cfg, "config", args.eval_config)
    override(eval_cfg, "pred_dir", args.eval_pred_dir)
    override(eval_cfg, "gt_dir", args.eval_gt_dir)
    override(eval_cfg, "gt_md_dir", args.eval_gt_md_dir)
    override(eval_cfg, "output_dir", args.eval_output_dir)
    override(eval_cfg, "metrics", args.eval_metrics)
    override(eval_cfg, "threshold_headings", args.eval_threshold_headings)
    override(eval_cfg, "threshold_table", args.eval_threshold_table)
    override(eval_cfg, "version", args.eval_version)
    override(eval_cfg, "text_include_f1", args.eval_text_include_f1)

    gt_dir = resolve_path(eval_cfg.get("gt_dir"), run_id)
    if gt_dir is None:
        raise ValueError("evaluate.gt_dir is required.")
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Copy markdown GT files if source directory is specified
    gt_md_dir = resolve_path(eval_cfg.get("gt_md_dir"), run_id)
    if gt_md_dir:
        if not gt_md_dir.exists():
            raise FileNotFoundError(f"GT markdown dir not found: {gt_md_dir}")
        copied = copy_md_to_md(gt_md_dir, gt_dir)
        print(f"[pipeline] Prepared GT md files: {copied} copied from {gt_md_dir} -> {gt_dir}")

    # Evaluation stage
    eval_script = resolve_path(eval_cfg.get("script", "olmocr/kolmocr_eval/scripts/evaluate.py"), run_id)
    pred_dir = resolve_path(eval_cfg.get("pred_dir"), run_id) or output_dir
    output_eval_root = resolve_path(eval_cfg.get("output_dir"), run_id)
    if not eval_script or not pred_dir or not output_eval_root:
        raise ValueError("Evaluation config must include script, pred_dir, and output_dir.")
    output_eval_root.mkdir(parents=True, exist_ok=True)

    eval_cmd = [
        python_bin,
        str(eval_script),
        "--pred_dir",
        str(pred_dir),
        "--gt_dir",
        str(gt_dir),
        "--output_dir",
        str(output_eval_root),
    ]

    if eval_cfg.get("config"):
        eval_cmd.extend(["--config", str(resolve_path(eval_cfg["config"], run_id))])
    metrics = eval_cfg.get("metrics")
    if metrics:
        eval_cmd.extend(["--metrics", *metrics])
    if eval_cfg.get("threshold_headings") is not None:
        eval_cmd.extend(["--threshold_headings", str(eval_cfg["threshold_headings"])])
    if eval_cfg.get("threshold_table") is not None:
        eval_cmd.extend(["--threshold_table", str(eval_cfg["threshold_table"])])
    if eval_cfg.get("version"):
        eval_cmd.extend(["--version", str(eval_cfg["version"])])
    if eval_cfg.get("text_include_f1") is False:
        eval_cmd.append("--no_text_f1")
    elif eval_cfg.get("text_include_f1") is True:
        eval_cmd.append("--text_include_f1")

    print(f"[pipeline] Running evaluation -> {output_eval_root}")
    run_cmd(eval_cmd)


if __name__ == "__main__":
    main()
