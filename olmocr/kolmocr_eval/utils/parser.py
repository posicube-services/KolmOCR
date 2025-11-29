import re
from typing import Dict, List, Optional

from olmocr.kolmocr_eval.utils.data_io import read_md

# Table 표현 타입:
# - posicube : <POSICUBE_TABLE_KV> ... </POSICUBE_TABLE_KV>
# - html     : <table> ... </table>
TABLE_PATTERNS = {
    "posicube": re.compile(r"<POSICUBE_TABLE_KV>[\s\S]*?</POSICUBE_TABLE_KV>", re.MULTILINE),
    "html": re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE | re.MULTILINE),
}

# 이미지 패턴: ![alt](url)
IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((.*?)\)")
# 이미지 bbox 주석: <!-- bbox: [x0,y0,w,h] -->
BBOX_PATTERN = re.compile(
    r"<!--\s*bbox:\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*]\s*-->",
    re.IGNORECASE,
)
# 수식 패턴
MATH_BLOCK_PATTERN = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
MATH_INLINE_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)


def detect_table_type(md: str) -> Optional[str]:
    """문서에서 감지되는 테이블 표현 타입을 반환."""
    for name, pattern in TABLE_PATTERNS.items():
        if pattern.search(md):
            return name
    return None


def extract_tables(md: str, table_type: Optional[str] = None) -> List[str]:
    """테이블 블록을 리스트로 추출."""
    detected = table_type or detect_table_type(md)
    if detected is None:
        return []
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
    """이미지 bbox 주석을 추출하여 [x0, y0, w, h] 리스트를 반환."""
    bboxes: List[List[float]] = []
    for match in BBOX_PATTERN.finditer(md):
        x0, y0, w, h = match.groups()
        bboxes.append([float(x0), float(y0), float(w), float(h)])
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
        "table_type": detected,
    }


def parse_md_file(path: str, table_type: Optional[str] = None) -> Dict[str, object]:
    """md 파일 경로를 입력받아 파싱 결과 반환."""
    content = read_md(path)
    return parse_md(content, table_type)
