"""
Convert bbox format from [x,y,w,h] to [x0,y0,x1,y1] in graphic markdown files.

Process:
1. Unnormalize with 1000x1000: [x*1000, y*1000, w*1000, h*1000]
2. Convert to x0,y0,x1,y1: [x, y, x+w, y+h]
3. Normalize with 1000x1000: [x0/1000, y0/1000, x1/1000, y1/1000]
"""

import re
from pathlib import Path


def convert_bbox_xywh_to_xyxy(bbox_str):
    """Convert bbox from [x,y,w,h] to [x0,y0,x1,y1] format."""
    # Parse bbox string: "[22,10,643,11]"
    bbox_match = re.match(r'\[(\d+),(\d+),(\d+),(\d+)\]', bbox_str)
    if not bbox_match:
        return None

    x, y, w, h = map(int, bbox_match.groups())

    # Step 1: Unnormalize (assuming already in 1000x1000 scale)
    # The values are already in pixel coordinates (0-1000)

    # Step 2: Convert to x0,y0,x1,y1
    x0 = x
    y0 = y
    x1 = x + w
    y1 = y + h

    # Step 3: Keep in 1000x1000 scale (no need to normalize/unnormalize)
    # Return as normalized values if needed, but the values are already 0-1000

    return f"[{x0},{y0},{x1},{y1}]"


def process_markdown_file(md_path):
    """Process a single markdown file to convert all bbox annotations."""
    print(f"\nProcessing: {md_path}")

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all bbox patterns: <!-- bbox: [x,y,w,h] -->
    bbox_pattern = r'<!-- bbox: (\[\d+,\d+,\d+,\d+\]) -->'

    matches = list(re.finditer(bbox_pattern, content))
    print(f"Found {len(matches)} bbox annotations")

    if not matches:
        print("No bbox found, skipping...")
        return 0

    # Convert all bboxes
    converted_count = 0
    new_content = content

    for match in reversed(matches):  # Process in reverse to maintain positions
        old_bbox = match.group(1)
        new_bbox = convert_bbox_xywh_to_xyxy(old_bbox)

        if new_bbox:
            old_text = f"<!-- bbox: {old_bbox} -->"
            new_text = f"<!-- bbox: {new_bbox} -->"

            # Replace in content
            start, end = match.span()
            new_content = new_content[:start] + new_text + new_content[end:]

            print(f"  {old_bbox} → {new_bbox}")
            converted_count += 1

    # Write back to file
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Converted {converted_count} bboxes in {md_path.name}")
    return converted_count


def main():
    """Process all markdown files in kolmocr_bench/graphic/."""
    graphic_dir = Path(__file__).parent.parent / 'kolmocr_bench' / 'graphic'

    if not graphic_dir.exists():
        print(f"Error: Directory not found: {graphic_dir}")
        return

    # Find all markdown files
    md_files = list(graphic_dir.glob('**/*.md'))
    print(f"Found {len(md_files)} markdown files in {graphic_dir}")

    total_converted = 0
    for md_file in sorted(md_files):
        converted = process_markdown_file(md_file)
        total_converted += converted

    print(f"\n{'='*60}")
    print(f"Total: Converted {total_converted} bboxes in {len(md_files)} files")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
