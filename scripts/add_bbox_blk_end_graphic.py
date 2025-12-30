#!/usr/bin/env python3
"""
Add <!-- bbox_blk_end --> tags after bbox blocks in markdown files.
Handles both formats:
- <!-- id: ... bbox: [...] --> (code_block format)
- <!-- bbox: [...] --> (graphic/eval format)
"""
import re
from pathlib import Path
import argparse


def process_markdown_file(file_path: Path) -> bool:
    """
    Process a single markdown file to add bbox_blk_end tags.
    Returns True if file was modified, False otherwise.
    """
    content = file_path.read_text(encoding='utf-8')
    original_content = content

    # Split content by lines and process
    lines = content.split('\n')
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this line is a bbox comment (either format)
        # Format 1: <!-- id: ... bbox: [...] -->
        # Format 2: <!-- bbox: [...] -->
        is_bbox_comment = (
            re.match(r'<!-- id: [^\s]+ bbox: \[[^\]]+\] -->', line.strip()) or
            re.match(r'<!-- bbox: \[[^\]]+\] -->', line.strip())
        )

        if is_bbox_comment:
            result_lines.append(line)
            i += 1

            # Collect content until we hit another bbox comment or end of file
            block_content = []
            while i < len(lines):
                next_line = lines[i]
                # Stop if we hit another bbox comment
                if (re.match(r'<!-- id: [^\s]+ bbox: \[[^\]]+\] -->', next_line.strip()) or
                    re.match(r'<!-- bbox: \[[^\]]+\] -->', next_line.strip())):
                    break
                # Stop if we hit bbox_blk_end (already processed)
                if '<!-- bbox_blk_end -->' in next_line:
                    break
                block_content.append(next_line)
                i += 1

            # Add the block content
            result_lines.extend(block_content)

            # Add bbox_blk_end if we collected any content and it's not already there
            if block_content:
                # Check if last non-empty line already has bbox_blk_end
                last_non_empty = None
                for line in reversed(block_content):
                    if line.strip():
                        last_non_empty = line
                        break

                if last_non_empty is None or '<!-- bbox_blk_end -->' not in last_non_empty:
                    # Remove trailing empty lines
                    while result_lines and not result_lines[-1].strip():
                        result_lines.pop()
                    result_lines.append('<!-- bbox_blk_end -->')
                    result_lines.append('')
        else:
            result_lines.append(line)
            i += 1

    new_content = '\n'.join(result_lines)

    # Remove any trailing newlines and add exactly one
    new_content = new_content.rstrip('\n') + '\n'

    if new_content != original_content:
        file_path.write_text(new_content, encoding='utf-8')
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Add bbox_blk_end tags to markdown files')
    parser.add_argument('directory', type=Path, help='Directory to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory {args.directory} does not exist")
        return 1

    md_files = list(args.directory.rglob('*.md'))
    if not md_files:
        print(f"No markdown files found in {args.directory}")
        return 0

    print(f"Found {len(md_files)} markdown files")
    modified_count = 0

    for md_file in md_files:
        if args.dry_run:
            content = md_file.read_text(encoding='utf-8')
            # Count both bbox formats
            bbox_count = (
                len(re.findall(r'<!-- id: [^\s]+ bbox: \[[^\]]+\] -->', content)) +
                len(re.findall(r'<!-- bbox: \[[^\]]+\] -->', content))
            )
            blk_end_count = content.count('<!-- bbox_blk_end -->')
            if bbox_count > blk_end_count:
                print(f"Would modify: {md_file} ({bbox_count} bbox blocks, {blk_end_count} bbox_blk_end tags)")
                modified_count += 1
        else:
            if process_markdown_file(md_file):
                print(f"Modified: {md_file}")
                modified_count += 1

    if args.dry_run:
        print(f"\nWould modify {modified_count} files (dry run)")
    else:
        print(f"\nModified {modified_count} files")

    return 0


if __name__ == '__main__':
    exit(main())
