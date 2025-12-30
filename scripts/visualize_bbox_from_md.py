#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create bbox visualization images from markdown files.
Reads bbox coordinates from .md files and draws them on corresponding .png images.

IMPORTANT: Bbox coordinates in md files are normalized to 1000x1000 space.
They must be denormalized to actual PNG dimensions before drawing.

Usage:
    # Basic usage (images in same directory as markdown files)
    python scripts/visualize_bbox_from_md.py \\
        --input-md-dir kolmocr_bench \\
        --output-dir output/visualized_kolmocr_bench

    # With separate image directory
    python scripts/visualize_bbox_from_md.py \\
        --input-md-dir kolmocr_bench \\
        --input-image-dir kolmocr_bench \\
        --output-dir output/visualized_kolmocr_bench
"""

import argparse
from pathlib import Path
import re
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


def parse_bboxes_from_md(md_path: Path) -> List[Tuple[int, int, int, int, int]]:
    """
    Parse bbox coordinates from markdown file.
    Returns list of (bbox_num, x0, y0, x1, y1) tuples.
    Coordinates are in normalized 1000x1000 space.
    """
    try:
        content = md_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  Error reading {md_path}: {e}")
        return []

    # Pattern: <!-- bbox: [x0,y0,x1,y1] -->
    pattern = r'<!-- bbox: \[(\d+),(\d+),(\d+),(\d+)\] -->'
    matches = re.findall(pattern, content)

    bboxes = []
    for i, match in enumerate(matches, start=1):
        x0 = int(match[0])
        y0 = int(match[1])
        x1 = int(match[2])
        y1 = int(match[3])
        bboxes.append((i, x0, y0, x1, y1))

    return bboxes


def denormalize_bbox(
    x0: int, y0: int, x1: int, y1: int,
    img_width: int, img_height: int,
    norm: int = 1000
) -> Tuple[float, float, float, float]:
    """
    Denormalize bbox coordinates from 1000x1000 space to actual image dimensions.

    Args:
        x0, y0, x1, y1: Normalized coordinates (0-1000 range)
        img_width, img_height: Actual image dimensions in pixels
        norm: Normalization size (default 1000)

    Returns:
        (x0_pixel, y0_pixel, x1_pixel, y1_pixel) in actual image coordinates
    """
    x0_pixel = (x0 / float(norm)) * img_width
    y0_pixel = (y0 / float(norm)) * img_height
    x1_pixel = (x1 / float(norm)) * img_width
    y1_pixel = (y1 / float(norm)) * img_height

    return x0_pixel, y0_pixel, x1_pixel, y1_pixel


def draw_bboxes_on_image(
    img_path: Path,
    bboxes: List[Tuple[int, int, int, int, int]],
    output_path: Path
) -> bool:
    """
    Draw bbox rectangles on image and save to output path.

    Args:
        img_path: Path to source PNG image
        bboxes: List of (bbox_num, x0, y0, x1, y1) tuples in normalized coordinates
        output_path: Path to save visualization

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image
        img = Image.open(img_path).convert("RGBA")
        img_width, img_height = img.size
        draw = ImageDraw.Draw(img)

        # Try to load font, fall back to default if unavailable
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        # Draw each bbox
        for bbox_num, x0_norm, y0_norm, x1_norm, y1_norm in bboxes:
            # Denormalize coordinates
            x0, y0, x1, y1 = denormalize_bbox(
                x0_norm, y0_norm, x1_norm, y1_norm,
                img_width, img_height
            )

            # Convert to int for drawing
            x0, y0 = int(round(x0)), int(round(y0))
            x1, y1 = int(round(x1)), int(round(y1))

            # Draw rectangle outline (magenta)
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=(255, 0, 255, 255),
                width=3
            )

            # Draw label with background
            label = f"{bbox_num}"
            bbox_text = draw.textbbox((0, 0), label, font=font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            pad = 3

            # Label background (semi-transparent magenta)
            label_y = max(0, y0 - th - 2 * pad)
            draw.rectangle(
                [x0, label_y, x0 + tw + 2 * pad, y0],
                fill=(255, 0, 255, 160)
            )

            # Label text (black)
            draw.text(
                (x0 + pad, max(0, y0 - th - pad)),
                label,
                font=font,
                fill=(0, 0, 0, 255)
            )

        # Save output image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return True

    except Exception as e:
        print(f"  Error processing image {img_path}: {e}")
        return False


def find_corresponding_png(
    md_path: Path,
    md_base_dir: Path,
    image_base_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Find corresponding PNG file for a markdown file.

    Args:
        md_path: Path to markdown file
        md_base_dir: Base directory for markdown files
        image_base_dir: Base directory for images (if None, uses same dir as md_path)

    Returns:
        Path to PNG if found, None otherwise
    """
    if image_base_dir is None:
        # Original behavior: look in same directory as MD file
        search_dir = md_path.parent
    else:
        # Parallel structure: get relative path from md_base, apply to image_base
        rel_path = md_path.parent.relative_to(md_base_dir)
        search_dir = image_base_dir / rel_path

    # Try: same basename with .png extension
    png_path = search_dir / f"{md_path.stem}.png"
    if png_path.exists():
        return png_path

    # Try: {basename}_gt.png (ground truth)
    gt_png_path = search_dir / f"{md_path.stem}_gt.png"
    if gt_png_path.exists():
        return gt_png_path

    return None


def main(
    input_md_dir: Path,
    input_image_dir: Optional[Path] = None,
    output_dir: Path = None,
    file_patterns: list = None
):
    """
    Create bbox visualizations from markdown files.

    Args:
        input_md_dir: Directory containing markdown files with bbox annotations
        input_image_dir: Directory containing corresponding PNG images (if None, uses same dir as md files)
        output_dir: Directory to save visualization images
        file_patterns: Optional list of file patterns to filter
    """
    base_dir = input_md_dir
    image_base_dir = input_image_dir
    output_base = output_dir

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return

    if image_base_dir and not image_base_dir.exists():
        print(f"Error: {image_base_dir} does not exist")
        return

    print(f"Creating bbox visualizations from: {base_dir}")
    if image_base_dir:
        print(f"Image directory: {image_base_dir}")
    else:
        print(f"Image directory: Same as markdown files")
    print(f"Output directory: {output_base}")
    if file_patterns:
        print(f"Filtering to specific files: {file_patterns}")
    print("=" * 80)

    # Find all .md files
    md_files = list(base_dir.rglob("*.md"))

    # Filter files if patterns provided
    if file_patterns:
        filtered_files = []
        for file_path in md_files:
            rel_path = str(file_path.relative_to(base_dir))
            # Check if any pattern matches this file
            for pattern in file_patterns:
                if pattern in rel_path:
                    filtered_files.append(file_path)
                    break
        md_files = filtered_files
        print(f"Filtered to {len(md_files)} files matching patterns\n")
    else:
        print(f"Found {len(md_files)} .md files\n")

    # Process each md file
    total_visualized = 0
    total_bboxes = 0
    total_skipped = 0

    for md_path in md_files:
        # Parse bboxes from md
        bboxes = parse_bboxes_from_md(md_path)

        if not bboxes:
            continue

        # Find corresponding PNG
        png_path = find_corresponding_png(md_path, base_dir, image_base_dir)

        if not png_path:
            total_skipped += 1
            rel_path = md_path.relative_to(base_dir)
            print(f"⚠ {rel_path}: No PNG found, skipping")
            continue

        # Determine output path (preserve directory structure)
        rel_path = md_path.relative_to(base_dir)
        output_path = output_base / rel_path.parent / f"{md_path.stem}_visualized.png"

        # Draw bboxes
        success = draw_bboxes_on_image(png_path, bboxes, output_path)

        if success:
            total_visualized += 1
            total_bboxes += len(bboxes)
            print(f"✓ {rel_path} ({len(bboxes)} bboxes) → {output_path.relative_to(output_base.parent)}")

    print("\n" + "=" * 80)
    print(f"✅ Visualization complete!")
    print(f"   Images created: {total_visualized}")
    print(f"   Total bboxes drawn: {total_bboxes}")
    print(f"   MD files skipped (no PNG): {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create bbox visualization images from markdown files."
    )
    parser.add_argument(
        "--input-md-dir",
        type=Path,
        required=True,
        help="Directory containing markdown files with bbox annotations"
    )
    parser.add_argument(
        "--input-image-dir",
        type=Path,
        default=None,
        help="Directory containing corresponding PNG images. If not specified, looks in same directory as markdown files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save visualization images"
    )

    args = parser.parse_args()

    main(
        input_md_dir=args.input_md_dir,
        input_image_dir=args.input_image_dir,
        output_dir=args.output_dir
    )
