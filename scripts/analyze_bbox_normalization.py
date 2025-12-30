#!/usr/bin/env python3
"""
Analyze bbox normalization in HTML files
"""
import re
from pathlib import Path
import subprocess

def get_image_dimensions(img_path):
    """Get image dimensions using identify command."""
    try:
        result = subprocess.run(
            ['identify', str(img_path)],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output: filename format WxH ...
        parts = result.stdout.split()
        dims = parts[2]  # WxH format
        width, height = dims.split('x')
        return int(width), int(height)
    except:
        return None, None

def get_max_bbox_coords(html_path):
    """Get maximum bbox coordinates from HTML file."""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all bbox values
    bbox_pattern = r'data-bbox="([^"]+)"'
    matches = re.findall(bbox_pattern, content)

    max_x = 0
    max_y = 0

    for bbox_str in matches:
        coords = [int(c) for c in bbox_str.split(',')]
        if len(coords) == 4:
            x0, y0, x1, y1 = coords
            max_x = max(max_x, x0, x1)
            max_y = max(max_y, y0, y1)

    return max_x, max_y

def main():
    graphic_dir = Path('kolmocr_bench/graphic')

    print(f"{'Folder':<20} {'Image Size':<15} {'Max bbox X':<12} {'Max bbox Y':<12} {'Ratio'}")
    print("=" * 80)

    for subdir in sorted(graphic_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Find HTML file
        html_files = list(subdir.glob('*.html'))
        if not html_files:
            continue
        html_file = html_files[0]

        # Find first image
        img_dir = subdir / 'imgs'
        if not img_dir.exists():
            continue

        img_files = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
        if not img_files:
            continue
        img_file = img_files[0]

        # Get dimensions
        img_w, img_h = get_image_dimensions(img_file)
        if img_w is None:
            continue

        # Get max bbox coords
        max_x, max_y = get_max_bbox_coords(html_file)

        # Calculate ratio
        ratio_x = max_x / img_w if img_w > 0 else 0
        ratio_y = max_y / img_h if img_h > 0 else 0

        print(f"{subdir.name:<20} {img_w}x{img_h:<10} {max_x:<12} {max_y:<12} x:{ratio_x:.3f} y:{ratio_y:.3f}")

if __name__ == "__main__":
    main()
