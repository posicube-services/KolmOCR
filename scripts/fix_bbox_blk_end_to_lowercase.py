#!/usr/bin/env python3
"""
Convert <!-- BBOX_BLK_END --> to <!-- bbox_blk_end --> in benchmark markdown files.
"""

from pathlib import Path


def convert_bbox_blk_end_to_lowercase(file_path: Path) -> bool:
    """
    Replace <!-- BBOX_BLK_END --> with <!-- bbox_blk_end --> in a file.

    Returns True if any replacements were made.
    """
    content = file_path.read_text(encoding='utf-8')
    new_content = content.replace('<!-- BBOX_BLK_END -->', '<!-- bbox_blk_end -->')

    if content != new_content:
        file_path.write_text(new_content, encoding='utf-8')
        return True
    return False


def main():
    base_dir = Path(__file__).parent.parent / "kolmocr_bench"

    # Process both code_block and graphic directories
    directories = [
        base_dir / "code_block",
        base_dir / "graphic"
    ]

    total_files = 0
    total_replacements = 0

    for directory in directories:
        if not directory.exists():
            print(f"Warning: {directory} does not exist")
            continue

        # Find all .md files recursively
        md_files = list(directory.rglob("*.md"))

        for md_file in md_files:
            if convert_bbox_blk_end_to_lowercase(md_file):
                count = md_file.read_text(encoding='utf-8').count('<!-- bbox_blk_end -->')
                total_replacements += count
                total_files += 1
                print(f"✓ {md_file.relative_to(base_dir)} ({count} replacements)")

    print(f"\n✅ Converted {total_replacements} occurrences in {total_files} files")


if __name__ == "__main__":
    main()
