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


def copy_html_to_md(html_root: Path, md_root: Path) -> int:
    """Copy all .html files to .md so the evaluator can read them."""
    copied = 0
    for html_path in html_root.rglob("*.html"):
        rel = html_path.relative_to(html_root)
        md_path = md_root / rel.with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
        copied += 1
    return copied


def run_cmd(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run KolmOCR inference + evaluation as a single pipeline.")
    parser.add_argument("--config", default="configs/kolmocr_bench_pipeline.yaml", help="Pipeline YAML path.")
    parser.add_argument("--run-id", dest="run_id", help="Override run_id for output formatting.")
    args = parser.parse_args()

    cfg_path = REPO_ROOT / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    run_id = args.run_id or cfg.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    python_bin = cfg.get("python") or sys.executable

    # Inference stage
    inf_cfg = cfg.get("inference", {})
    inf_script = resolve_path(inf_cfg.get("script", "olmocr/inference_kolmocr_transformer.py"), run_id)
    checkpoint = resolve_path(inf_cfg.get("checkpoint"), run_id)
    input_dir = resolve_path(inf_cfg.get("input_dir"), run_id)
    output_dir = resolve_path(inf_cfg.get("output_dir"), run_id)
    if not inf_script or not checkpoint or not input_dir or not output_dir:
        raise ValueError("Inference config must include script, checkpoint, input_dir, and output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)

    inf_cmd = [
        python_bin,
        str(inf_script),
        "--checkpoint",
        str(checkpoint),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]
    if inf_cfg.get("config"):
        inf_cmd.extend(["--config", str(resolve_path(inf_cfg["config"], run_id))])
    if inf_cfg.get("prompt"):
        inf_cmd.extend(["--prompt", str(format_value(inf_cfg["prompt"], run_id))])
    if inf_cfg.get("max_new_tokens") is not None:
        inf_cmd.extend(["--max-new-tokens", str(inf_cfg["max_new_tokens"])])
    if inf_cfg.get("temperature") is not None:
        inf_cmd.extend(["--temperature", str(inf_cfg["temperature"])])
    if inf_cfg.get("top_p") is not None:
        inf_cmd.extend(["--top-p", str(inf_cfg["top_p"])])
    if inf_cfg.get("device_map"):
        inf_cmd.extend(["--device-map", str(inf_cfg["device_map"])])

    print(f"[pipeline] Running inference -> {output_dir}")
    run_cmd(inf_cmd)

    # Prepare GT (.html -> .md) if requested
    eval_cfg = cfg.get("evaluate", {})
    gt_dir = resolve_path(eval_cfg.get("gt_dir"), run_id)
    if gt_dir is None:
        raise ValueError("evaluate.gt_dir is required.")
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_html_dir = resolve_path(eval_cfg.get("gt_html_dir"), run_id)
    if gt_html_dir:
        if not gt_html_dir.exists():
            raise FileNotFoundError(f"GT html dir not found: {gt_html_dir}")
        copied = copy_html_to_md(gt_html_dir, gt_dir)
        print(f"[pipeline] Prepared GT md files: {copied} copied from {gt_html_dir} -> {gt_dir}")

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

    print(f"[pipeline] Running evaluation -> {output_eval_root}")
    run_cmd(eval_cmd)


if __name__ == "__main__":
    main()
