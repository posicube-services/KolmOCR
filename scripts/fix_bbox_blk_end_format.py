#!/usr/bin/env python3
"""
Replace [BBOX_BLK_END] with <!-- BBOX_BLK_END --> in markdown files.
"""
from pathlib import Path
import argparse


def process_markdown_file(file_path: Path) -> bool:
    """
    Process a single markdown file to replace BBOX_BLK_END format.
    Returns True if file was modified, False otherwise.
    """
    content = file_path.read_text(encoding='utf-8')
    original_content = content

    # Replace [BBOX_BLK_END] with <!-- BBOX_BLK_END -->
    new_content = content.replace('[BBOX_BLK_END]', '<!-- BBOX_BLK_END -->')

    if new_content != original_content:
        file_path.write_text(new_content, encoding='utf-8')
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Replace [BBOX_BLK_END] with <!-- BBOX_BLK_END -->')
    parser.add_argument('directories', nargs='+', type=Path, help='Directories to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    args = parser.parse_args()

    all_md_files = []
    for directory in args.directories:
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist")
            continue
        all_md_files.extend(directory.rglob('*.md'))

    if not all_md_files:
        print("No markdown files found")
        return 0

    print(f"Found {len(all_md_files)} markdown files")
    modified_count = 0

    for md_file in all_md_files:
        if args.dry_run:
            content = md_file.read_text(encoding='utf-8')
            count = content.count('[BBOX_BLK_END]')
            if count > 0:
                print(f"Would modify: {md_file} ({count} replacements)")
                modified_count += 1
        else:
            if process_markdown_file(md_file):
                content = md_file.read_text(encoding='utf-8')
                count = content.count('<!-- BBOX_BLK_END -->')
                print(f"Modified: {md_file} ({count} replacements)")
                modified_count += 1

    if args.dry_run:
        print(f"\nWould modify {modified_count} files (dry run)")
    else:
        print(f"\nModified {modified_count} files")

    return 0


if __name__ == '__main__':
    exit(main())
