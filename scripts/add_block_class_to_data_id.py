#!/usr/bin/env python3
"""
Add .block class to all elements with data-id attribute in HTML files.
"""

from pathlib import Path
from bs4 import BeautifulSoup


def add_block_class_to_data_id_elements(html_path: Path) -> bool:
    """Add .block class to all elements with data-id attribute."""
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Find all elements with data-id attribute
    elements_with_data_id = soup.find_all(attrs={'data-id': True})

    modified = False
    for element in elements_with_data_id:
        # Check if element already has 'block' class
        if element.get('class'):
            classes = element.get('class')
            if 'block' not in classes:
                classes.append('block')
                element['class'] = classes
                modified = True
        else:
            # No class attribute, add 'block'
            element['class'] = ['block']
            modified = True

    if modified:
        # Write back to file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(str(soup))
        return True

    return False


def main():
    """Process all HTML files in kolmocr_bench/graphic."""
    graphic_dir = Path(__file__).parent.parent / 'kolmocr_bench' / 'graphic'

    if not graphic_dir.exists():
        print(f"Error: Directory not found: {graphic_dir}")
        return

    # Find all HTML files
    html_files = sorted(list(graphic_dir.rglob('*.html')))
    print(f"Found {len(html_files)} HTML files")

    modified_count = 0
    for html_file in html_files:
        try:
            if add_block_class_to_data_id_elements(html_file):
                print(f"[MODIFIED] {html_file.relative_to(graphic_dir)}")
                modified_count += 1
            else:
                print(f"[SKIPPED] {html_file.relative_to(graphic_dir)} (already has .block class)")
        except Exception as e:
            print(f"[ERROR] {html_file.relative_to(graphic_dir)}: {e}")

    print(f"\n{'='*60}")
    print(f"Modified {modified_count}/{len(html_files)} files")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
