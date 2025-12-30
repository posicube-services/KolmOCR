import re
from typing import Dict, List, Optional

from olmocr.kolmocr_eval.utils.data_io import read_md

# Table 표현 타입:
# - posicube : <POSICUBE_TABLE_KV> ... </POSICUBE_TABLE_KV>
# - html     : <table> ... </table>
# - markdown : | a | b | 형태의 파이프 테이블
TABLE_PATTERNS = {
    "posicube": re.compile(r"<POSICUBE_TABLE_KV>[\s\S]*?</POSICUBE_TABLE_KV>", re.MULTILINE),
    "html": re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE | re.MULTILINE),
    # 연속된 파이프 테이블 블록 전체를 감지
    "markdown": re.compile(r"(?:^\s*\|.*\|\s*$\n?){2,}", re.MULTILINE),
}

# 이미지 패턴: ![alt](url)
IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((.*?)\)")
# 이미지 bbox 주석: <!-- bbox: [x0,y0,w,h] --> 또는 <!-- [x0,y0,w,h] -->
BBOX_PATTERN = re.compile(
    r"<!--\s*(?:bbox:\s*)?\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*]\s*-->",
    re.IGNORECASE,
)
# 수식 패턴
MATH_BLOCK_PATTERN = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
MATH_INLINE_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)

# Markdown 테이블 감지/파싱용 패턴
_MD_TABLE_HEADER = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{3,}\s*(\|\s*:?-{3,}\s*)+\|?\s*$")


def detect_table_type(md: str) -> Optional[str]:
    """문서에서 감지되는 테이블 표현 타입을 반환."""
    for name, pattern in TABLE_PATTERNS.items():
        if pattern.search(md):
            return name
    return None


def _find_markdown_tables(md: str) -> List[List[str]]:
    """
    Markdown 파이프 테이블 블록을 라인 리스트 형태로 추출.
    헤더-구분선 쌍을 찾고, 이후 파이프가 포함된 라인들을 테이블로 묶는다.
    """
    lines = md.splitlines()
    tables: List[List[str]] = []
    i = 0
    n = len(lines)
    while i + 1 < n:
        header_line = lines[i]
        divider_line = lines[i + 1]
        if not (_MD_TABLE_HEADER.match(header_line) and _MD_TABLE_DIVIDER.match(divider_line)):
            i += 1
            continue
        block = [header_line, divider_line]
        j = i + 2
        while j < n and "|" in lines[j]:
            block.append(lines[j])
            j += 1
        tables.append(block)
        i = j
    return tables


def _md_table_to_html(table_lines: List[str]) -> str:
    """
    간단한 Markdown 파이프 테이블을 HTML 테이블 문자열로 변환.
    헤더/바디만 지원하며 colspan/rowspan은 고려하지 않는다.
    """
    if len(table_lines) < 2:
        return ""
    header = table_lines[0]
    data_lines = table_lines[2:]

    def _split_row(row: str) -> List[str]:
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [cell.strip() for cell in row.split("|")]

    headers = _split_row(header)
    rows = [_split_row(r) for r in data_lines]

    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{h}</th>" for h in headers)
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for r in rows:
        parts.append("<tr>")
        for cell in r:
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody>")
    parts.append("</table>")
    return "".join(parts)


def extract_tables(md: str, table_type: Optional[str] = None) -> List[str]:
    """테이블 블록을 리스트로 추출."""
    detected = table_type or detect_table_type(md)
    if detected is None:
        return []
    if detected == "markdown":
        blocks = _find_markdown_tables(md)
        return [_md_table_to_html(b) for b in blocks if _md_table_to_html(b)]
    pattern = TABLE_PATTERNS.get(detected)
    if pattern is None:
        return []
    return pattern.findall(md)


def extract_images(md: str) -> List[Dict[str, str]]:
    """이미지 마크다운을 추출하고 alt/url/원문을 반환."""
    images = []
    for match in IMAGE_PATTERN.finditer(md):
        alt_text, url = match.group(1), match.group(2)
        images.append({"alt": alt_text, "url": url, "raw": match.group(0)})
    return images


def extract_image_bboxes(md: str) -> List[List[float]]:
    """이미지 bbox 주석을 추출하여 [x0, y0, x1, y1] 리스트를 반환.

    bbox 주석 바로 다음 줄에 이미지 마크다운(![alt](url))이 있는 경우에만 추출한다.
    """
    bboxes: List[List[float]] = []
    lines = md.split('\n')

    for i, line in enumerate(lines):
        # Check if current line is a bbox comment
        bbox_match = BBOX_PATTERN.search(line)
        if not bbox_match:
            continue

        # Check if next line exists and is an image
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if IMAGE_PATTERN.search(next_line):
                # Extract bbox coordinates
                x0, y0, x1, y1 = bbox_match.groups()
                bboxes.append([float(x0), float(y0), float(x1), float(y1)])

    return bboxes


def extract_formulas(md: str) -> List[str]:
    """
    마크다운에서 LaTeX 수식을 추출한다.
    $$...$$ 블록 수식을 먼저 찾고, 남은 영역에서 $...$ 인라인 수식을 찾는다.
    """
    formulas = []

    # 블록 수식 추출
    blocks = MATH_BLOCK_PATTERN.findall(md)
    formulas.extend([b.strip() for b in blocks if b.strip()])

    # 블록 수식을 제거한 뒤 인라인 수식 추출
    md_wo_blocks = MATH_BLOCK_PATTERN.sub("", md)
    inline = MATH_INLINE_PATTERN.findall(md_wo_blocks)
    formulas.extend([i.strip() for i in inline if i.strip()])

    return formulas


def _strip_patterns(md: str, patterns: List[re.Pattern]) -> str:
    cleaned = md
    for p in patterns:
        cleaned = p.sub("", cleaned)
    return cleaned


def extract_text(md: str, table_type: Optional[str] = None) -> str:
    """테이블/이미지를 제거한 순수 텍스트를 반환."""
    detected = table_type or detect_table_type(md)
    patterns = [IMAGE_PATTERN]
    if detected:
        patterns.append(TABLE_PATTERNS[detected])
    cleaned = _strip_patterns(md, patterns)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def parse_md(md: str, table_type: Optional[str] = None) -> Dict[str, object]:
    """md 문자열에서 텍스트/테이블/이미지를 분리."""
    detected = table_type or detect_table_type(md)
    return {
        "text": extract_text(md, detected),
        "tables": extract_tables(md, detected),
        "images": extract_images(md),
        "image_bboxes": extract_image_bboxes(md),
        "formulas": extract_formulas(md),
        # markdown은 HTML로 변환하므로 downstream에서는 html로 취급
        "table_type": "html" if detected == "markdown" else detected,
    }


def parse_md_file(path: str, table_type: Optional[str] = None) -> Dict[str, object]:
    """md 파일 경로를 입력받아 파싱 결과 반환."""
    content = read_md(path)
    return parse_md(content, table_type)
