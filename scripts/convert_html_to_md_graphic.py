#!/usr/bin/env python3
"""
Convert HTML files in kolmocr_bench/graphic to Markdown format with bbox annotations.
"""

import re
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString, Tag


def extract_text_content(element):
    """Extract text content from an element, preserving some formatting."""
    if isinstance(element, NavigableString):
        return str(element).strip()

    # For elements with children, concatenate their text
    text_parts = []
    for child in element.children:
        if isinstance(child, NavigableString):
            text_parts.append(str(child))
        elif child.name == 'br':
            text_parts.append('\n')
        elif child.name == 'strong' or child.name == 'b':
            text_parts.append(f"**{child.get_text()}**")
        elif child.name == 'em' or child.name == 'i':
            text_parts.append(f"*{child.get_text()}*")
        else:
            text_parts.append(extract_text_content(child))

    return ''.join(text_parts).strip()


def convert_element_to_markdown(element, depth=0, parent_bbox=''):
    """Convert an HTML element to Markdown with bbox annotation."""
    if not isinstance(element, Tag):
        return ""

    # Get bbox from current element or use parent's bbox
    bbox = element.get('data-bbox', '') or parent_bbox
    data_id = element.get('data-id', '')

    result = []

    # Handle different element types
    if element.name == 'h1':
        text = extract_text_content(element)
        if bbox:
            result.append(f'<!-- bbox: [{bbox}] -->')
        result.append(f'# {text}')
        if bbox:
            result.append('<!-- bbox_blk_end -->')

    elif element.name == 'h2':
        text = extract_text_content(element)
        if bbox:
            result.append(f'<!-- bbox: [{bbox}] -->')
        result.append(f'## {text}')
        if bbox:
            result.append('<!-- bbox_blk_end -->')

    elif element.name == 'h3':
        text = extract_text_content(element)
        if bbox:
            result.append(f'<!-- bbox: [{bbox}] -->')
        result.append(f'### {text}')
        if bbox:
            result.append('<!-- bbox_blk_end -->')

    elif element.name == 'h4':
        text = extract_text_content(element)
        if bbox:
            result.append(f'<!-- bbox: [{bbox}] -->')
        result.append(f'#### {text}')
        if bbox:
            result.append('<!-- bbox_blk_end -->')

    elif element.name == 'p':
        text = extract_text_content(element)
        if text:  # Only add if there's actual content
            if bbox:
                result.append(f'<!-- bbox: [{bbox}] -->')
            result.append(text)
            if bbox:
                result.append('<!-- bbox_blk_end -->')

    elif element.name == 'li':
        text = extract_text_content(element)
        if text:
            if bbox:
                result.append(f'<!-- bbox: [{bbox}] -->')
            result.append(f'- {text}')
            if bbox:
                result.append('<!-- bbox_blk_end -->')

    elif element.name == 'img':
        alt = element.get('alt', '')
        src = element.get('src', '')
        if bbox:
            result.append(f'<!-- bbox: [{bbox}] -->')
        result.append(f'![{alt}]({src})')
        if bbox:
            result.append('<!-- bbox_blk_end -->')

    elif element.name == 'figcaption':
        text = extract_text_content(element)
        if text:
            if bbox:
                result.append(f'<!-- bbox: [{bbox}] -->')
            result.append(f'*{text}*')
            if bbox:
                result.append('<!-- bbox_blk_end -->')

    elif element.name == 'table':
        # Keep tables as HTML
        if bbox:
            result.append(f'<!-- bbox: [{bbox}] -->')
        # Get the table HTML as string
        table_html = str(element)
        # Clean up the HTML a bit
        table_html = re.sub(r'\n\s*\n', '\n', table_html)
        result.append(table_html)
        if bbox:
            result.append('<!-- bbox_blk_end -->')

    elif element.name in ['div', 'figure', 'ul', 'ol']:
        # For divs with bbox, pass bbox to children
        # But don't create separate blocks for the div itself
        current_bbox = element.get('data-bbox', '')

        # Process children recursively, passing down the bbox if this element has one
        for child in element.children:
            if isinstance(child, Tag):
                # Pass current bbox to child if child doesn't have its own bbox
                child_md = convert_element_to_markdown(child, depth + 1, current_bbox or parent_bbox)
                if child_md:
                    result.append(child_md)

    return '\n'.join(filter(None, result))


def convert_html_to_markdown(html_path: Path) -> str:
    """Convert an HTML file to Markdown format."""
    print(f"Converting {html_path}...")

    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the main document div
    document_div = soup.find('div', class_='document') or soup.find('div', class_='page')

    if not document_div:
        print(f"  Warning: No document div found in {html_path}")
        return ""

    # Find all relevant elements (blocks + headings + tables + images)
    # Process elements with class="block" AND headings/tables that might not have the class
    processed_elements = set()
    markdown_parts = []

    # First, process all elements with class="block"
    blocks = document_div.find_all(class_='block')
    for block in blocks:
        md = convert_element_to_markdown(block)
        if md:
            markdown_parts.append(md)
            processed_elements.add(id(block))

    # Then, find headings and tables that weren't processed (don't have class="block")
    for element in document_div.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table']):
        if id(element) not in processed_elements:
            # Check if parent div has bbox we can use
            parent_bbox = ''
            parent = element.find_parent(['div'])
            if parent:
                parent_bbox = parent.get('data-bbox', '')

            md = convert_element_to_markdown(element, parent_bbox=parent_bbox)
            if md:
                markdown_parts.append(md)
                processed_elements.add(id(element))

    return '\n\n'.join(markdown_parts)


def main():
    """Convert all HTML files in kolmocr_bench/graphic to Markdown."""
    graphic_dir = Path('kolmocr_bench/graphic')

    if not graphic_dir.exists():
        print(f"Error: {graphic_dir} does not exist")
        return

    # Find all HTML files
    html_files = list(graphic_dir.glob('*/*.html'))
    print(f"Found {len(html_files)} HTML files to convert\n")

    for html_file in sorted(html_files):
        try:
            # Convert to markdown
            markdown_content = convert_html_to_markdown(html_file)

            if markdown_content:
                # Write to .md file (same name, different extension)
                md_file = html_file.with_suffix('.md')
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

                print(f"  ✓ Created {md_file.relative_to('kolmocr_bench')}")
            else:
                print(f"  ✗ No content generated for {html_file.name}")

        except Exception as e:
            print(f"  ✗ Error converting {html_file.name}: {e}")

    print(f"\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
