"""CLI helper to convert HTML into markdown using the existing formatter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from olmocr.bench.synth.mine_html_templates import html_to_markdown_with_frontmatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HTML files into markdown with frontmatter metadata."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="HTML file(s) or directories to convert. Directories are searched recursively for .html files.",
    )
    parser.add_argument(
        "--input-dir",
        "-d",
        action="append",
        type=Path,
        default=[],
        help="Directory to search recursively for HTML files. Can be provided multiple times.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to write markdown output. Can be a file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit directory to write markdown outputs (overrides --output for directory resolution).",
    )
    return parser.parse_args()


def collect_html_files(inputs: Iterable[Path]) -> List[Path]:
    """Resolve files and directories into a flat list of HTML file paths."""
    html_paths: List[Path] = []
    for path in inputs:
        resolved = path.expanduser()
        if resolved.is_file():
            html_paths.append(resolved)
        elif resolved.is_dir():
            html_paths.extend(sorted(resolved.rglob("*.html")))
        else:
            raise SystemExit(f"Input path not found: {path}")

    if not html_paths:
        raise SystemExit("No HTML files found for conversion.")

    return html_paths


def convert_html_file(input_path: Path) -> str:
    """Read HTML, convert to markdown, and return the result."""
    html_text = input_path.read_text(encoding="utf-8")
    return html_to_markdown_with_frontmatter(html_text)


def ensure_output_dir(path: Path) -> Path:
    """Create the directory if needed and return the resolved path."""
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def write_markdown(path: Path, markdown: str) -> None:
    """Persist markdown to disk with UTF-8 encoding."""
    path.write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    inputs = args.inputs
    input_dirs = args.input_dir
    output_target = args.output
    explicit_output_dir = args.output_dir

    if not inputs and not input_dirs:
        raise SystemExit("Provide at least one HTML path or --input-dir.")

    html_paths = collect_html_files(inputs + input_dirs)
    multi_input = len(html_paths) > 1

    output_dir: Optional[Path] = None
    output_file: Optional[Path] = None

    if multi_input and not output_target and not explicit_output_dir:
        raise SystemExit(
            "When providing multiple HTML files you must use --output to specify a directory."
        )

    if explicit_output_dir:
        output_dir = ensure_output_dir(explicit_output_dir)
        if output_target and output_target.exists() and output_target.is_file():
            raise SystemExit(
                "Cannot combine --output-dir with --output when --output is a file."
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

    for html_path in html_paths:
        markdown = convert_html_file(html_path)

        if output_file and not multi_input:
            write_markdown(output_file, markdown)
            print(f"Wrote markdown to {output_file}")
            continue

        if output_dir:
            target = output_dir / f"{html_path.stem}.md"
            write_markdown(target, markdown)
            print(f"Wrote markdown to {target}")
            continue

        sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
