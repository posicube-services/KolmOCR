#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert bbox format from [x,y,w,h] to [x0,y0,x1,y1] in all kolmocr_bench files.
Uses actual PNG dimensions for proper denormalization/renormalization.

Usage:
    python scripts/convert_bbox_xywh_to_xyxy.py
"""

from pathlib import Path
import re
from typing import Tuple, Optional

try:
    from PIL import Image
except ImportError:
    print("Error: PIL (Pillow) is required. Install with: pip install pillow")
    exit(1)


def find_corresponding_png(file_path: Path) -> Optional[Path]:
    """Find corresponding PNG file for a md/html file."""
    # Try: same basename with .png extension
    png_path = file_path.with_suffix('.png')
    if png_path.exists():
        return png_path

    # Try: {basename}_gt.png (ground truth)
    gt_png_path = file_path.parent / f"{file_path.stem}_gt.png"
    if gt_png_path.exists():
        return gt_png_path

    return None


def get_image_size(png_path: Path) -> Tuple[int, int]:
    """Get image dimensions (width, height)."""
    try:
        with Image.open(png_path) as img:
            return img.size
    except Exception:
        return None, None


def convert_bbox_with_image(x: int, y: int, w: int, h: int,
                            img_width: int, img_height: int,
                            norm: int = 1000) -> Tuple[int, int, int, int]:
    """
    Convert bbox from [x,y,w,h] to [x0,y0,x1,y1] with proper denormalization.

    Process:
    1. Denormalize from 1000x1000 to actual image size
    2. Convert [x,y,w,h] to [x0,y0,x1,y1] in pixel space
    3. Renormalize back to 1000x1000
    """
    # Denormalize to pixel coordinates
    x_pixel = (x / float(norm)) * img_width
    y_pixel = (y / float(norm)) * img_height
    w_pixel = (w / float(norm)) * img_width
    h_pixel = (h / float(norm)) * img_height

    # Convert to x0,y0,x1,y1 in pixel space
    x0_pixel = x_pixel
    y0_pixel = y_pixel
    x1_pixel = x_pixel + w_pixel
    y1_pixel = y_pixel + h_pixel

    # Renormalize to 1000x1000
    x0 = int(round((x0_pixel / img_width) * norm))
    y0 = int(round((y0_pixel / img_height) * norm))
    x1 = int(round((x1_pixel / img_width) * norm))
    y1 = int(round((y1_pixel / img_height) * norm))

    return x0, y0, x1, y1


def convert_bbox_md(content: str, img_width: int, img_height: int) -> Tuple[str, int]:
    """
    Convert bbox format in markdown files.
    Pattern: <!-- bbox: [x,y,w,h] --> -> <!-- bbox: [x0,y0,x1,y1] -->

    Returns (new_content, count_of_replacements)
    """
    pattern = r'<!-- bbox: \[(\d+),(\d+),(\d+),(\d+)\] -->'
    replacements = 0

    def replacement(match):
        nonlocal replacements
        x = int(match.group(1))
        y = int(match.group(2))
        w = int(match.group(3))
        h = int(match.group(4))

        x0, y0, x1, y1 = convert_bbox_with_image(x, y, w, h, img_width, img_height)

        replacements += 1
        return f'<!-- bbox: [{x0},{y0},{x1},{y1}] -->'

    new_content = re.sub(pattern, replacement, content)
    return new_content, replacements


def convert_bbox_html(content: str, img_width: int, img_height: int) -> Tuple[str, int]:
    """
    Convert bbox format in HTML files.
    Pattern: data-bbox="x,y,w,h" -> data-bbox="x0,y0,x1,y1"

    Returns (new_content, count_of_replacements)
    """
    pattern = r'data-bbox="(\d+),(\d+),(\d+),(\d+)"'
    replacements = 0

    def replacement(match):
        nonlocal replacements
        x = int(match.group(1))
        y = int(match.group(2))
        w = int(match.group(3))
        h = int(match.group(4))

        x0, y0, x1, y1 = convert_bbox_with_image(x, y, w, h, img_width, img_height)

        replacements += 1
        return f'data-bbox="{x0},{y0},{x1},{y1}"'

    new_content = re.sub(pattern, replacement, content)
    return new_content, replacements


def process_file(file_path: Path) -> Tuple[bool, int]:
    """
    Process a single file and convert bbox format.

    Returns (changed, count) where changed is True if file was modified.
    """
    try:
        # Find corresponding PNG to get image dimensions
        png_path = find_corresponding_png(file_path)
        if not png_path:
            # Skip files without PNG (can't determine proper conversion)
            return False, 0

        img_width, img_height = get_image_size(png_path)
        if not img_width or not img_height:
            print(f"  Warning: Could not read image dimensions from {png_path}")
            return False, 0

        content = file_path.read_text(encoding='utf-8')

        if file_path.suffix == '.md':
            new_content, count = convert_bbox_md(content, img_width, img_height)
        elif file_path.suffix == '.html':
            new_content, count = convert_bbox_html(content, img_width, img_height)
        else:
            return False, 0

        if content != new_content and count > 0:
            file_path.write_text(new_content, encoding='utf-8')
            return True, count

        return False, 0

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return False, 0


def main(file_patterns: list = None):
    base_dir = Path(__file__).parent.parent / "kolmocr_bench"

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return

    print(f"Converting bbox format in: {base_dir}")
    if file_patterns:
        print(f"Filtering to specific files: {file_patterns}")
    print("=" * 80)

    # Find all .md and .html files
    md_files = list(base_dir.rglob("*.md"))
    html_files = list(base_dir.rglob("*.html"))

    all_files = md_files + html_files

    # Filter files if patterns provided
    if file_patterns:
        filtered_files = []
        for file_path in all_files:
            rel_path = str(file_path.relative_to(base_dir))
            # Check if any pattern matches this file
            for pattern in file_patterns:
                if pattern in rel_path:
                    filtered_files.append(file_path)
                    break
        all_files = filtered_files
        print(f"Filtered to {len(all_files)} files matching patterns\n")
    else:
        print(f"Found {len(md_files)} .md files and {len(html_files)} .html files\n")

    # Process each file
    total_changed = 0
    total_replacements = 0

    for file_path in all_files:
        changed, count = process_file(file_path)

        if changed:
            total_changed += 1
            total_replacements += count
            rel_path = file_path.relative_to(base_dir.parent)
            print(f"✓ {rel_path} ({count} bboxes)")

    print("\n" + "=" * 80)
    print(f"✅ Conversion complete!")
    print(f"   Files modified: {total_changed}")
    print(f"   Total bboxes converted: {total_replacements}")
    print(f"   Format: [x,y,w,h] → [x0,y0,x1,y1] where x1=x+w, y1=y+h")


if __name__ == "__main__":
    main()
