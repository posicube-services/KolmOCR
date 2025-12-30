#!/usr/bin/env python3
"""Overlay bounding boxes from an HTML file onto an image.

Usage:
    python tests/visualize_bbox.py --input-img path/to/image.png --input-html path/to/doc.html [--output out.png]

Notes:
    - Expects data-bbox attributes on elements (e.g., <div data-bbox="x,y,w,h">) that were
      normalized to 1000x1000. This script rescales them to the input image size before drawing.
    - No external dependencies beyond Pillow and the standard library.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw


def _parse_bboxes_from_html(html_text: str) -> list[tuple[float, float, float, float]]:
    """Extract bbox tuples from data-bbox attributes in the HTML."""
    pattern = re.compile(r'data-bbox="([\d\.\-,]+)"')
    bboxes: list[tuple[float, float, float, float]] = []
    for match in pattern.finditer(html_text):
        raw = match.group(1)
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 4:
            continue
        try:
            x, y, w, h = (float(p) for p in parts)
            bboxes.append((x, y, w, h))
        except ValueError:
            continue
    return bboxes


def _scale_bboxes(
    bboxes: Iterable[tuple[float, float, float, float]],
    target_size: tuple[int, int],
    source_size: tuple[float, float] = (1000.0, 1000.0),
) -> list[tuple[int, int, int, int]]:
    """Scale normalized bboxes (default 1000x1000) to the target image size."""
    sx = target_size[0] / source_size[0]
    sy = target_size[1] / source_size[1]
    scaled: list[tuple[int, int, int, int]] = []
    for x, y, w, h in bboxes:
    #for x, y, x1, y1 in bboxes:
        left = int(round(x * sx))
        top = int(round(y * sy))
        right = int(round((x + w) * sx))
        bottom = int(round((y + h) * sy))
        # right = int(round(x1 * sx))
        # bottom = int(round(y1 * sy))
        scaled.append((left, top, right, bottom))
    return scaled


def _draw_bboxes(
    image: Image.Image,
    bboxes: Iterable[tuple[int, int, int, int]],
    colors: tuple[tuple[int, int, int], ...] | None = None,
) -> Image.Image:
    """Draw bounding boxes on a copy of the image."""
    if colors is None or len(colors) == 0:
        colors = (
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 165, 0),
            (128, 0, 128),
        )
    draw = ImageDraw.Draw(image)
    for idx, box in enumerate(bboxes):
        color = colors[idx % len(colors)]
        draw.rectangle(box, outline=color, width=2)
    return image


def visualize_bboxes(input_img: Path, input_html: Path, output: Path | None = None) -> Path:
    """Create an overlay image with bboxes drawn on top."""
    html_text = input_html.read_text(encoding="utf-8")
    raw_bboxes = _parse_bboxes_from_html(html_text)
    if not raw_bboxes:
        raise SystemExit(f"No data-bbox found in {input_html}")

    with Image.open(input_img) as img:
        scaled = _scale_bboxes(raw_bboxes, img.size)
        annotated = _draw_bboxes(img.copy(), scaled)
        if output is None:
            output = input_img.with_name(input_img.stem + "_bbox.png")
        annotated.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay normalized bboxes onto an image.")
    parser.add_argument("--input-img", required=True, type=Path, help="Path to PNG image.")
    parser.add_argument("--input-html", required=True, type=Path, help="Path to HTML with data-bbox attributes.")
    parser.add_argument("--output", type=Path, help="Output path for annotated PNG (default: *_bbox.png next to input image).")
    args = parser.parse_args()

    if not args.input_img.exists():
        raise SystemExit(f"Input image not found: {args.input_img}")
    if not args.input_html.exists():
        raise SystemExit(f"Input HTML not found: {args.input_html}")

    out_path = visualize_bboxes(args.input_img, args.input_html, args.output)
    print(f"Annotated image written to: {out_path}")


if __name__ == "__main__":
    main()