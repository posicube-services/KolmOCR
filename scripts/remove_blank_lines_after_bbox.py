#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remove blank lines between bbox comments and code blocks in markdown files.

Pattern to fix:
<!-- id: b004 bbox: [61, 195, 877, 59] -->

```c
...
```

Should become:
<!-- id: b004 bbox: [61, 195, 877, 59] -->
```c
...
```
"""

from pathlib import Path
import re


def remove_blank_lines_after_bbox(content: str) -> tuple[str, int]:
    """
    Remove blank lines after bbox comments.

    Returns (new_content, count_of_removals)
    """
    # Pattern 1: bbox comment followed by blank line(s) and then code block
    # Match: <!-- id: ... bbox: ... -->\n\n```
    pattern1 = r'(<!-- (?:id: [^\s>]+ )?bbox: \[[^\]]+\] -->)\n\n+(```)'

    # Pattern 2: bbox comment followed by blank line(s) and then any content
    # Match: <!-- bbox: ... -->\n\n(any non-blank content)
    pattern2 = r'(<!-- bbox: \[[^\]]+\] -->)\n\n+(?=\S)'

    replacements = 0

    def replacement1(match):
        nonlocal replacements
        replacements += 1
        # Return bbox comment + single newline + code block marker
        return f'{match.group(1)}\n{match.group(2)}'

    def replacement2(match):
        nonlocal replacements
        replacements += 1
        # Return bbox comment + single newline (content follows)
        return f'{match.group(1)}\n'

    # Apply both patterns
    new_content = re.sub(pattern1, replacement1, content)
    new_content = re.sub(pattern2, replacement2, new_content)

    return new_content, replacements


def process_md_file(file_path: Path) -> tuple[bool, int]:
    """
    Process a single markdown file.

    Returns (changed, count) where changed is True if file was modified.
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content, count = remove_blank_lines_after_bbox(content)

        if content != new_content and count > 0:
            file_path.write_text(new_content, encoding='utf-8')
            return True, count

        return False, 0

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return False, 0


def main():
    base_dir = Path(__file__).parent.parent / "kolmocr_bench"

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return

    print(f"Removing blank lines after bbox comments in: {base_dir}")
    print("=" * 80)

    # Find all .md files
    md_files = list(base_dir.rglob("*.md"))
    print(f"Found {len(md_files)} .md files\n")

    # Process each file
    total_changed = 0
    total_removals = 0

    for file_path in md_files:
        changed, count = process_md_file(file_path)

        if changed:
            total_changed += 1
            total_removals += count
            rel_path = file_path.relative_to(base_dir.parent)
            print(f"✓ {rel_path} ({count} blank lines removed)")

    print("\n" + "=" * 80)
    print(f"✅ Processing complete!")
    print(f"   Files modified: {total_changed}")
    print(f"   Total blank lines removed: {total_removals}")


if __name__ == "__main__":
    main()
