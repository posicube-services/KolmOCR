#!/usr/bin/env python3
"""
Reverse engineer the normalization parameters used
"""
import re
from pathlib import Path

def analyze_bbox_normalization(html_path):
    """Analyze if bbox was normalized to 1000x1000."""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all bbox values
    bbox_pattern = r'data-bbox="([^"]+)"'
    matches = re.findall(bbox_pattern, content)

    if not matches:
        return None

    max_x = 0
    max_y = 0

    for bbox_str in matches:
        coords = [int(c) for c in bbox_str.split(',')]
        if len(coords) == 4:
            x, y, w, h = coords
            # bbox is in x,y,w,h format
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

    return {
        'file': html_path.name,
        'max_x': max_x,
        'max_y': max_y,
        'normalized_1000x1000': max_x <= 1000 and max_y <= 1000,
        'aspect_ratio': max_x / max_y if max_y > 0 else 0,
    }

def main():
    graphic_dir = Path('kolmocr_bench/graphic')

    print(f"{'File':<25} {'Max X':<10} {'Max Y':<10} {'1000x1000?':<15} {'Aspect'}")
    print("=" * 80)

    for html_file in sorted(graphic_dir.glob('*/*.html')):
        result = analyze_bbox_normalization(html_file)
        if result:
            normalized_str = "✓ YES" if result['normalized_1000x1000'] else "✗ NO"
            print(f"{result['file']:<25} {result['max_x']:<10} {result['max_y']:<10} {normalized_str:<15} {result['aspect_ratio']:.3f}")

if __name__ == "__main__":
    main()
