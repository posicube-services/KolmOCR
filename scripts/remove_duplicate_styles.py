#!/usr/bin/env python3
"""
Remove duplicate CtxtMenu style blocks from HTML files.
These styles are MathJax context menu styles that get duplicated during processing.
"""

from pathlib import Path
import re


def remove_duplicate_ctxtmenu_styles(html_path: Path) -> bool:
    """
    Remove all but the first occurrence of CtxtMenu style blocks.
    Returns True if file was modified.
    """
    content = html_path.read_text(encoding='utf-8')
    original = content

    # Pattern to match CtxtMenu style blocks
    # These blocks start with <style type="text/css">.CtxtMenu_
    pattern = r'<style type="text/css">\.CtxtMenu_[^<]*</style>'

    # Find all matches
    matches = list(re.finditer(pattern, content))

    if len(matches) <= 1:
        # No duplicates found
        return False

    print(f"  Found {len(matches)} CtxtMenu style blocks in {html_path.name}")

    # Remove all but the first occurrence
    # We go backwards to preserve indices
    for match in reversed(matches[1:]):
        content = content[:match.start()] + content[match.end():]

    html_path.write_text(content, encoding='utf-8')
    return True


def main():
    graphic_dir = Path("kolmocr_bench/graphic")

    if not graphic_dir.exists():
        print(f"Directory not found: {graphic_dir}")
        return

    html_files = list(graphic_dir.rglob("*.html"))
    print(f"Found {len(html_files)} HTML files\n")

    modified_count = 0
    for html_path in html_files:
        if remove_duplicate_ctxtmenu_styles(html_path):
            modified_count += 1
            print(f"  ✓ Cleaned {html_path.relative_to(graphic_dir)}")

    print(f"\n{'='*60}")
    print(f"Modified {modified_count}/{len(html_files)} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
