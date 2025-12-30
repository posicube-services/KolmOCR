#!/usr/bin/env python3
"""
Add data-id and .block class to all <figure> elements in HTML files.
"""

from pathlib import Path
from bs4 import BeautifulSoup


def add_bbox_to_figures(html_path: Path) -> bool:
    """Add data-id and .block class to all <figure> elements."""
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Find all figure elements
    figures = soup.find_all('figure')

    if not figures:
        return False

    modified = False
    fig_counter = 1

    for figure in figures:
        # Check if figure already has data-id
        if not figure.get('data-id'):
            # Generate unique figure ID
            while True:
                fig_id = f"fig-{fig_counter:03d}"
                # Check if ID already exists in document
                if not soup.find(attrs={'data-id': fig_id}):
                    break
                fig_counter += 1

            figure['data-id'] = fig_id
            modified = True
            fig_counter += 1

        # Check if figure already has 'block' class
        if figure.get('class'):
            classes = figure.get('class')
            if 'block' not in classes:
                classes.append('block')
                figure['class'] = classes
                modified = True
        else:
            # No class attribute, add 'block'
            figure['class'] = ['block']
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
            if add_bbox_to_figures(html_file):
                print(f"[MODIFIED] {html_file.relative_to(graphic_dir)}")
                modified_count += 1
            else:
                print(f"[SKIPPED] {html_file.relative_to(graphic_dir)} (no figures or already has data-id & .block)")
        except Exception as e:
            print(f"[ERROR] {html_file.relative_to(graphic_dir)}: {e}")

    print(f"\n{'='*60}")
    print(f"Modified {modified_count}/{len(html_files)} files")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
