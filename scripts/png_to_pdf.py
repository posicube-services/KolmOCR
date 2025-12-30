"""Simple CLI to convert PNG images into PDF files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PNG files (or directories of PNGs) into PDF documents."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="PNG file(s) or directories containing PNG files.",
    )
    parser.add_argument(
        "--input-dir",
        "-d",
        action="append",
        type=Path,
        default=[],
        help="Directory to search recursively for PNG files. May be provided multiple times.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to write the PDF output (file or directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit directory where all PDFs should be stored (overrides --output when output is treated as dir).",
    )
    return parser.parse_args()


def collect_png_files(inputs: Iterable[Path]) -> List[Path]:
    png_paths: List[Path] = []
    for entry in inputs:
        resolved = entry.expanduser()
        if resolved.is_file():
            if resolved.suffix.lower() == ".png":
                png_paths.append(resolved)
        elif resolved.is_dir():
            png_paths.extend(sorted(resolved.rglob("*.png")))
        else:
            raise SystemExit(f"Input path not found: {entry}")

    if not png_paths:
        raise SystemExit("No PNG files found for conversion.")

    return png_paths


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def convert_png_to_pdf(png_path: Path, target_pdf: Path) -> None:
    target_pdf.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(png_path) as img:
        pdf_image = img.convert("RGB")
        pdf_image.save(target_pdf, "PDF")


def main() -> None:
    args = parse_args()
    inputs = args.inputs
    input_dirs = args.input_dir
    output_target = args.output
    explicit_output_dir = args.output_dir

    if not inputs and not input_dirs:
        raise SystemExit("Provide at least one PNG path or use --input-dir.")

    png_paths = collect_png_files(inputs + input_dirs)
    multi_input = len(png_paths) > 1

    output_dir: Optional[Path] = None
    output_file: Optional[Path] = None

    if explicit_output_dir:
        output_dir = ensure_output_dir(explicit_output_dir)
        if output_target and output_target.exists() and output_target.is_file():
            raise SystemExit(
                "Cannot combine --output-dir with --output when --output is expected to be a file."
            )
    elif output_target:
        if multi_input:
            if output_target.exists() and not output_target.is_dir():
                raise SystemExit("For multiple inputs the --output path must be a directory.")
            output_dir = ensure_output_dir(output_target)
        else:
            if output_target.exists():
                if output_target.is_dir():
                    output_dir = output_target.resolve()
                else:
                    output_file = output_target.resolve()
            else:
                if output_target.suffix:
                    output_target.parent.mkdir(parents=True, exist_ok=True)
                    output_file = output_target.resolve()
                else:
                    output_dir = ensure_output_dir(output_target)

    for png_path in png_paths:
        if output_file and not multi_input:
            target = output_file
        elif output_dir:
            target = output_dir / f"{png_path.stem}.pdf"
        else:
            target = png_path.with_suffix(".pdf")

        convert_png_to_pdf(png_path, target)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
