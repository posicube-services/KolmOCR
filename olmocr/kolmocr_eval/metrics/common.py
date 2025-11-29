import re
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from Levenshtein import distance as edit_distance

from olmocr.kolmocr_eval.utils.data_io import clean_text
from olmocr.kolmocr_eval.utils.parser import extract_formulas
from olmocr.kolmocr_eval.utils.matching import f1_score
from olmocr.kolmocr_eval.utils.tree import Node, tree_edit_distance, tree_size
import pandas as pd


def normalized_edit_distance(pred: str, gt: str) -> float:
    """d(pred, gt) / max(|gt|, 1). 0이 가장 좋다."""
    if not pred and not gt:
        return 0.0
    return edit_distance(pred, gt) / max(len(gt), 1)


def similarity_from_distance(distance: float) -> float:
    """Edit distance 비율을 유사도로 변환 (0~1 사이로 clamp)."""
    return max(0.0, 1.0 - distance)


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def round_numeric(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round numeric columns to the given decimals."""
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].round(decimals)
    return df


# ---------- Table parsing ----------
def _parse_html_table(table_html: str) -> List[List[Dict[str, object]]]:
    """HTML table을 행/셀 단위로 단순 파싱."""
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    target = table if table else soup
    rows: List[List[Dict[str, object]]] = []
    for tr in target.find_all("tr"):
        parsed_row = []
        for cell in tr.find_all(["th", "td"]):
            text = " ".join(cell.get_text(" ", strip=True).split())
            parsed_row.append(
                {
                    "tag": cell.name.lower(),
                    "text": text,
                    "rowspan": int(cell.get("rowspan", 1)),
                    "colspan": int(cell.get("colspan", 1)),
                }
            )
        if parsed_row:
            rows.append(parsed_row)
    return rows


def _parse_posicube_table(table_block: str) -> List[List[Dict[str, object]]]:
    """<POSICUBE_TABLE_KV> 포맷을 행/셀 단위로 단순 파싱."""
    content = re.sub(r"</?POSICUBE_TABLE_KV>", "", table_block, flags=re.IGNORECASE)
    rows = []
    for line in content.splitlines():
        tokens = [tok.strip() for tok in re.split(r"\s*\|\s*", line) if tok.strip()]
        if not tokens:
            continue
        rows.append([{"text": tok, "rowspan": 1, "colspan": 1} for tok in tokens])
    return rows


def _serialize_table(rows: List[List[Dict[str, object]]], include_text: bool) -> str:
    parts = []
    for row in rows:
        cells = []
        for cell in row:
            base = f"r{cell.get('rowspan', 1)}c{cell.get('colspan', 1)}"
            if include_text:
                text = cell.get("text", "")
                if text:
                    base = f"{base}:{text}"
            cells.append(base)
        parts.append("|".join(cells))
    return "\n".join(parts) if parts else ""


def compute_table_scores(pred_tables: List[str], gt_tables: List[str], table_type: Optional[str]) -> Dict[str, float]:
    """
    테이블 블록 리스트를 받아 논문 Table-TEDS(Tree Edit Distance Similarity) 방식으로 점수를 계산.
    - 삽입/삭제 비용: 1
    - 대체 비용: 태그가 같으면 텍스트 normalized edit distance(텍스트 없으면 0), 다르면 1
    - 정규화: 1 - TED / max(|pred|, |gt|)
    """

    def parse_block(block: str) -> List[List[Dict[str, object]]]:
        if not block:
            return []
        if table_type == "posicube":
            return _parse_posicube_table(block)
        return _parse_html_table(block)

    structure_scores = []
    semantic_scores = []
    for gt_block, pred_block in zip_longest(gt_tables, pred_tables, fillvalue=""):
        gt_parsed = parse_block(gt_block)
        pred_parsed = parse_block(pred_block)
        structure_scores.append(_table_teds_similarity(pred_parsed, gt_parsed, include_text=False))
        semantic_scores.append(_table_teds_similarity(pred_parsed, gt_parsed, include_text=True))

    return {
        "table_teds": average(structure_scores) if structure_scores else 1.0,
        "table_teds_s": average(semantic_scores) if semantic_scores else 1.0,
    }


def _table_similarity(
    pred_table: Optional[List[List[Dict[str, object]]]],
    gt_table: Optional[List[List[Dict[str, object]]]],
    include_text: bool,
) -> float:
    if not pred_table and not gt_table:
        return 1.0
    pred_serial = _serialize_table(pred_table or [], include_text)
    gt_serial = _serialize_table(gt_table or [], include_text)
    distance = edit_distance(pred_serial, gt_serial)
    denom = max(len(pred_serial), len(gt_serial), 1)
    return similarity_from_distance(distance / denom)


def _normalize_cell_text(text: str) -> str:
    return " ".join(text.split())


def _build_table_tree(rows: List[List[Dict[str, object]]], include_text: bool) -> Node:
    """Build ordered table tree: table -> tr -> cell[tag+span] -> text(optional)."""
    root = Node("table")
    for row in rows:
        row_node = Node("tr")
        for cell in row:
            tag = cell.get("tag", "td")
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            cell_node = Node(f"{tag}:r{rowspan}c{colspan}")
            if include_text:
                text = _normalize_cell_text(cell.get("text", ""))
                if text:
                    cell_node.add(Node(f"text:{text}"))
            row_node.add(cell_node)
        root.add(row_node)
    return root


def _table_replace_cost(a: Node, b: Node) -> float:
    """논문 TEDS 규칙에 맞춘 대체 비용."""
    is_text_a = a.label.startswith("text:")
    is_text_b = b.label.startswith("text:")
    if is_text_a and is_text_b:
        text_a = a.label.split(":", 1)[1]
        text_b = b.label.split(":", 1)[1]
        return normalized_edit_distance(text_a, text_b)
    if is_text_a != is_text_b:
        return 1.0
    return 0.0 if a.label == b.label else 1.0


def _table_teds_similarity(
    pred_table: Optional[List[List[Dict[str, object]]]],
    gt_table: Optional[List[List[Dict[str, object]]]],
    include_text: bool,
) -> float:
    """
    논문 Table-TEDS 유사도: 1 - TED / max(|pred|, |gt|),
    삽입/삭제 비용 1, 대체 비용은 태그 동일 시 텍스트 NED(없으면 0), 태그 불일치 시 1.
    """
    if not pred_table and not gt_table:
        return 1.0
    pred_tree = _build_table_tree(pred_table or [], include_text)
    gt_tree = _build_table_tree(gt_table or [], include_text)

    # 삽입/삭제 비용을 1로 맞추기 위해 size_fn을 상수 반환으로 사용
    unit_size = lambda _: 1  # noqa: E731
    distance = tree_edit_distance(pred_tree, gt_tree, replace_cost_fn=_table_replace_cost, size_fn=unit_size)
    denom = max(tree_size(pred_tree), tree_size(gt_tree), 1)
    return similarity_from_distance(distance / denom)


def compute_table_f1_scores(
    pred_tables: List[str],
    gt_tables: List[str],
    table_type: Optional[str],
    threshold: float = 0.5,
    include_text: bool = True,
) -> Dict[str, float]:
    """
    테이블 블록 리스트를 받아 greedy matching으로 TP/FP/FN을 계산한 뒤 precision/recall/F1을 반환.
    include_text=False이면 구조만 사용, True면 셀 텍스트까지 포함해 매칭한다.
    """

    def parse_block(block: str) -> List[List[Dict[str, object]]]:
        if not block:
            return []
        if table_type == "posicube":
            return _parse_posicube_table(block)
        return _parse_html_table(block)

    pred_parsed = [parse_block(b) for b in pred_tables]
    gt_parsed = [parse_block(b) for b in gt_tables]

    if not gt_parsed:
        raise ValueError("GT tables are missing; cannot compute table F1 without ground truth tables.")
    if not pred_parsed:
        print("[Warning] No predicted tables; returning zero scores.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0, "avg_similarity": 0.0}

    num_pred = len(pred_tables)
    num_gt = len(gt_tables)
    matched = 0
    sims: List[float] = [0.0] * len(gt_parsed)  # 매칭되지 않은 GT는 0으로 채워 평균 계산

    # 모든 조합을 유사도 내림차순으로 정렬 후, 중복 없이 greedy 매칭
    combos: List[Tuple[float, int, int]] = []
    for gi, gt_tab in enumerate(gt_parsed):
        for pi, pred_tab in enumerate(pred_parsed):
            combos.append((_table_similarity(pred_tab, gt_tab, include_text=include_text), gi, pi))
    combos.sort(key=lambda x: x[0], reverse=True)

    used_gt = set()
    used_pred = set()
    for sim, gi, pi in combos:
        if gi in used_gt or pi in used_pred:
            continue
        sims[gi] = sim
        if sim >= threshold:
            matched += 1
        used_gt.add(gi)
        used_pred.add(pi)
        if len(used_gt) == len(gt_parsed) or len(used_pred) == len(pred_parsed):
            break

    precision = matched / num_pred if num_pred > 0 else 0.0
    recall = matched / num_gt if num_gt > 0 else 0.0
    if num_pred == 0 and num_gt == 0:
        precision = recall = 1.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "matched": matched,
        "avg_similarity": sum(sims) / len(sims) if sims else 0.0,
    }


# ---------- Formula ----------
def compute_formula_cdm(pred_md: str, gt_md: str) -> float:
    """수식 문자열의 문자 매칭 비율로 CDM을 근사."""
    from itertools import zip_longest

    pred_formulas = extract_formulas(pred_md)
    gt_formulas = extract_formulas(gt_md)
    total_chars = 0
    matched_chars = 0

    for gt_formula, pred_formula in zip_longest(gt_formulas, pred_formulas, fillvalue=""):
        gt_len = len(gt_formula)
        total_chars += gt_len
        if gt_len == 0:
            continue
        dist = edit_distance(pred_formula, gt_formula)
        matched_chars += max(gt_len - dist, 0)

    if total_chars == 0:
        return 1.0
    return matched_chars / total_chars


# ---------- Text / Reading order ----------
def compute_text_reading_scores(text_pred: str, text_gt: str) -> Dict[str, float]:
    """텍스트/읽기순서 normalized edit distance와 유사도 계산."""
    # 텍스트는 줄 전체를 대상으로, 읽기순서는 줄 단위 시퀀스를 비교
    text_pred_clean = clean_text(text_pred)
    text_gt_clean = clean_text(text_gt)
    text_ned = normalized_edit_distance(text_pred_clean, text_gt_clean)
    text_score = similarity_from_distance(text_ned)

    reading_pred = "\n".join([ln.strip() for ln in text_pred.splitlines() if ln.strip()])
    reading_gt = "\n".join([ln.strip() for ln in text_gt.splitlines() if ln.strip()])
    reading_ned = normalized_edit_distance(reading_pred, reading_gt)
    reading_score = similarity_from_distance(reading_ned)

    return {
        "text_edit_ned": text_ned,
        "text_edit": text_score,
        "reading_order_ned": reading_ned,
        "reading_order": reading_score,
    }
