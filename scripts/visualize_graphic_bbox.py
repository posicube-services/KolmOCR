"""
Visualize bboxes on graphic images.
Reads bbox annotations from markdown files and draws them on corresponding PNG images.
"""

import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def parse_bboxes_from_md(md_path):
    """Extract all bbox annotations from markdown file."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all bbox patterns: <!-- bbox: [x0,y0,x1,y1] -->
    bbox_pattern = r'<!-- bbox: \[(\d+),(\d+),(\d+),(\d+)\] -->'
    matches = re.findall(bbox_pattern, content)

    bboxes = []
    for match in matches:
        x0, y0, x1, y1 = map(int, match)
        bboxes.append((x0, y0, x1, y1))

    return bboxes


def draw_bboxes_on_image(image_path, bboxes, output_path):
    """Draw bounding boxes on image and save."""
    # Open image
    img = Image.open(image_path)
    width, height = img.size

    print(f"  Image size: {width}x{height}")
    print(f"  Drawing {len(bboxes)} bboxes...")

    # Create drawing context
    draw = ImageDraw.Draw(img)

    # Draw each bbox
    for idx, (x0, y0, x1, y1) in enumerate(bboxes):
        # Scale coordinates from 1000x1000 to actual image size
        scaled_x0 = x0 * width / 1000
        scaled_y0 = y0 * height / 1000
        scaled_x1 = x1 * width / 1000
        scaled_y1 = y1 * height / 1000

        # Draw rectangle
        draw.rectangle(
            [scaled_x0, scaled_y0, scaled_x1, scaled_y1],
            outline='red',
            width=3
        )

        # Draw bbox number
        try:
            # Try to use a font, fallback to default if not available
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()

        text = str(idx + 1)
        # Draw text background
        bbox_text = draw.textbbox((scaled_x0, scaled_y0), text, font=font)
        draw.rectangle(bbox_text, fill='red')
        draw.text((scaled_x0, scaled_y0), text, fill='white', font=font)

    # Save image
    img.save(output_path)
    print(f"  Saved visualization to: {output_path}")


def process_graphic_folder(folder_path):
    """Process a single graphic folder."""
    folder_name = folder_path.name
    print(f"\nProcessing: {folder_name}")

    # Find markdown file
    md_file = folder_path / f"{folder_name}.md"
    if not md_file.exists():
        print(f"  Warning: Markdown file not found: {md_file}")
        return False

    # Find PNG file
    png_file = folder_path / f"{folder_name}.png"
    if not png_file.exists():
        print(f"  Warning: PNG file not found: {png_file}")
        return False

    # Parse bboxes
    bboxes = parse_bboxes_from_md(md_file)
    if not bboxes:
        print(f"  No bboxes found in {md_file.name}")
        return False

    print(f"  Found {len(bboxes)} bboxes in {md_file.name}")

    # Create output path
    output_file = folder_path / f"{folder_name}_bbox_vis.png"

    # Draw and save
    draw_bboxes_on_image(png_file, bboxes, output_file)

    return True


def main():
    """Process all graphic folders."""
    graphic_dir = Path(__file__).parent.parent / 'kolmocr_bench' / 'graphic'

    if not graphic_dir.exists():
        print(f"Error: Directory not found: {graphic_dir}")
        return

    # Find all subdirectories
    subdirs = [d for d in graphic_dir.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} graphic folders in {graphic_dir}")

    success_count = 0
    for subdir in sorted(subdirs):
        try:
            if process_graphic_folder(subdir):
                success_count += 1
        except Exception as e:
            print(f"  Error processing {subdir.name}: {e}")

    print(f"\n{'='*60}")
    print(f"Successfully visualized {success_count}/{len(subdirs)} folders")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
