#!/usr/bin/env python3
"""
Analyze bbox coordinates in detail
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
        parts = result.stdout.split()
        dims = parts[2]
        width, height = dims.split('x')
        return int(width), int(height)
    except:
        return None, None

def get_document_width(html_path):
    """Try to extract document width from HTML."""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Look for document width
    width_match = re.search(r'\.document.*?width:\s*(\d+)px', content, re.DOTALL)
    if width_match:
        return int(width_match.group(1))

    # Look for page-w in mm
    page_w_match = re.search(r'--page-w:\s*(\d+)mm', content)
    if page_w_match:
        mm = int(page_w_match.group(1))
        # Convert mm to pixels at different DPI
        return f"{mm}mm (~{mm*3.78:.0f}px@96dpi, {mm*7.56:.0f}px@192dpi)"

    return "unknown"

def get_max_bbox_coords(html_path):
    """Get maximum bbox coordinates from HTML file."""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

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

    print(f"{'Folder':<20} {'Doc Width':<25} {'Image Size':<15} {'Max bbox':<15}")
    print("=" * 80)

    for subdir in sorted(graphic_dir.iterdir()):
        if not subdir.is_dir():
            continue

        html_files = list(subdir.glob('*.html'))
        if not html_files:
            continue
        html_file = html_files[0]

        img_dir = subdir / 'imgs'
        if not img_dir.exists():
            continue

        img_files = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
        if not img_files:
            continue
        img_file = img_files[0]

        img_w, img_h = get_image_dimensions(img_file)
        if img_w is None:
            continue

        max_x, max_y = get_max_bbox_coords(html_file)
        doc_width = get_document_width(html_file)

        print(f"{subdir.name:<20} {str(doc_width):<25} {img_w}x{img_h:<10} {max_x}x{max_y}")

if __name__ == "__main__":
    main()
