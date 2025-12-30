#!/usr/bin/env python3
"""
Convert graphic folder bbox format from <!-- bbox: [...] --> to <!-- id: ... bbox: [...] -->
to match code_block folder format.
"""

from pathlib import Path
import re


def convert_bbox_format(file_path: Path) -> tuple[bool, int]:
    """
    Convert bbox format in a file.
    Returns (changed, count) where changed is True if file was modified.
    """
    content = file_path.read_text(encoding='utf-8')

    # Pattern to match: <!-- bbox: [x,y,w,h] -->
    # We need to add an id before bbox
    pattern = r'<!-- bbox: \[([^\]]+)\] -->'

    block_counter = 1
    replacements = 0

    def replacement(match):
        nonlocal block_counter, replacements
        coords = match.group(1)
        new_format = f'<!-- id: b{block_counter:03d} bbox: [{coords}] -->'
        block_counter += 1
        replacements += 1
        return new_format

    new_content = re.sub(pattern, replacement, content)

    if content != new_content:
        file_path.write_text(new_content, encoding='utf-8')
        return True, replacements
    return False, 0


def main():
    base_dir = Path(__file__).parent.parent / "kolmocr_bench" / "graphic"

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return

    total_files = 0
    total_replacements = 0

    # Find all .md files recursively
    md_files = list(base_dir.rglob("*.md"))

    for md_file in md_files:
        changed, count = convert_bbox_format(md_file)
        if changed:
            total_files += 1
            total_replacements += count
            print(f"✓ {md_file.relative_to(base_dir.parent)} ({count} bboxes)")

    print(f"\n✅ Converted {total_replacements} bboxes in {total_files} files")


if __name__ == "__main__":
    main()
