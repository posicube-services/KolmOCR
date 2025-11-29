import re
import os
from typing import Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageChops


def read_md(file_path: str) -> str:
    """Read markdown file content."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def remove_images_from_md(md: str) -> str:
    """Strip markdown image blocks."""
    cleaned_text = re.sub(r"!\[.*?\]\(.*?\)", "", md)
    return cleaned_text.strip()


def remove_tables_from_md(md: str, ver: str) -> str:
    """
    Remove table expressions by version.
    v1.9 : html table
    v1.10 : custom tag
    """
    if ver == "1.9":
        cleaned_text = re.sub(r"<style[\s\S]*?</style>", "", md)
        cleaned_text = re.sub(r"<table[\s\S]*?</table>", "", cleaned_text)
    elif ver == "1.10":
        cleaned_text = re.sub(r"<POSICUBE_TABLE_KV>[\s\S]*?</POSICUBE_TABLE_KV>", "", md)
    else:
        cleaned_text = md
    return cleaned_text.strip()


def remove_heading_symbols_from_md(markdown_text: str) -> str:
    """Remove heading markers (#) from markdown."""
    cleaned_text = re.sub(r"^(#{1,6})\s*", "", markdown_text, flags=re.MULTILINE)
    return cleaned_text


def remove_errors_from_md(md: str) -> str:
    """
    Remove strings categorized as errors.
    Error types: "plaintext", "<p>|-..-|</p>"
    """
    cleaned_text = re.sub("plaintext", "", md)
    cleaned_text = re.sub(r"<p>\|.*?\|</p>", "", cleaned_text)
    return cleaned_text


def clean_text(text: str) -> str:
    """Normalize whitespace for evaluation."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def extract_headings(md: str) -> list[dict]:
    """Extract headings as list of dict(text, level, type)."""
    headings = re.findall(r"^(#{1,6})\s*(.+)", md, re.MULTILINE)
    return [{"text": h[1].strip(), "level": len(h[0]), "type": "dummy"} for h in headings]


def extract_lists(md: str) -> list[str]:
    """Extract list item texts (unordered lists)."""
    items = re.findall(r"^[ \t]*[-*+]\s+(.*)", md, re.MULTILINE)
    return [i.strip() for i in items if i.strip()]


def extract_single_number(filename, path_):
    numbers = re.findall(r"\d+", filename)

    if len(numbers) == 0:
        raise ValueError(f"{path_}/{filename}: 숫자가 포함되어 있지 않습니다.")
    elif len(numbers) > 1:
        raise ValueError(f"{path_}/{filename} 에서 여러 개의 숫자가 발견되었습니다: {numbers}")

    return numbers[0]


def get_index2file(path_):
    index = {}
    for f in os.listdir(path_):
        if f.endswith(".md"):
            file_path = os.path.basename(f)
            num = extract_single_number(file_path, path_)
            index[num] = f
    return index


def list_md_files(base_dir):
    """
    base_dir 하위의 모든 .md 파일을 재귀적으로 찾아 상대 경로 리스트로 반환.
    """
    md_paths = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".md"):
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, base_dir)
                md_paths.append(rel_path)
    return sorted(md_paths)


def render_csv_table_image(
    csv_path: str,
    image_path: Optional[str] = None,
    header_fill: str = "#f5f5f5",
    dpi: int = 600,
    font_path: Optional[str] = None,
) -> str:
    """
    csv_path의 테이블을 pandas + matplotlib를 사용하여 고화질 이미지(PNG)로 렌더링하고 저장.
    dpi: 저장 시 적용할 해상도(Dots Per Inch). 기본값 600.
    font_path: 사용할 폰트 경로 (미지정 시 matplotlib 기본 폰트 사용).
    저장 후 여백을 최소화하기 위해 흰색 배경을 기준으로 크롭합니다.
    """
    # matplotlib 백엔드 설정 (GUI 없이 작동)
    mpl.use('Agg')
    
    df = pd.read_csv(csv_path)
    if df.empty:
        df = pd.DataFrame([{"Element": "N/A", "F1-score": "N/A"}])
    if image_path is None:
        image_path = os.path.splitext(csv_path)[0] + ".png"

    # 폰트 설정
    if font_path:
        mpl.rcParams['font.family'] = font_path
    
    # 그림과 축 생성
    fig, ax = plt.subplots(figsize=(8, 3), dpi=dpi)
    ax.axis('tight')
    ax.axis('off')

    # 테이블 생성
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25] * len(df.columns),
    )

    # 테이블 스타일 설정
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 헤더 셀 스타일 (배경색 설정)
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor(header_fill)
        cell.set_text_props(weight='bold')

    # 데이터 셀 스타일
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            cell.set_facecolor('white')

    # 이미지로 저장 (고화질)
    plt.savefig(image_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    # 흰색 배경 기준으로 여백 크롭
    try:
        img = Image.open(image_path).convert("RGB")
        bg = Image.new("RGB", img.size, "white")
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            img.crop(bbox).save(image_path)
    except Exception:
        pass

    return image_path
