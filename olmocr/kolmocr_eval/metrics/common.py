import re
import math
from functools import lru_cache
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from bs4 import BeautifulSoup
from Levenshtein import distance as edit_distance
import unicodedata
import html

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


def align_sequences(pred_items: List[object], gt_items: List[object], sim_fn) -> List[Tuple[int, int]]:
    """
    Adjacency-style sequence 매칭: global alignment (Needleman-Wunsch)으로
    총 유사도를 극대화하며 GT/Pred 순서를 유지한 매칭 쌍 리스트를 반환.
    gap penalty는 0으로 두어 unmatched는 유사도 0으로 취급.
    """
    m, n = len(gt_items), len(pred_items)
    if m == 0 or n == 0:
        return []

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    # fill DP backwards
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            match = sim_fn(pred_items[j], gt_items[i]) + dp[i + 1][j + 1]
            skip_gt = dp[i + 1][j]
            skip_pred = dp[i][j + 1]
            dp[i][j] = max(match, skip_gt, skip_pred)

    # backtrack to collect pairs
    pairs: List[Tuple[int, int]] = []
    i = j = 0
    while i < m and j < n:
        score = dp[i][j]
        match = sim_fn(pred_items[j], gt_items[i]) + dp[i + 1][j + 1]
        if abs(score - match) < 1e-12:
            pairs.append((i, j))
            i += 1
            j += 1
            continue
        if abs(score - dp[i + 1][j]) < 1e-12:
            i += 1
        else:
            j += 1
    return pairs


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


def _normalize_html_table(table_html: str) -> str:
    """OmniDocBench식 HTML 테이블 정리: th->td, thead/span 제거, math alttext 삽입, style/attr 제거."""
    if not table_html:
        return ""
    soup = BeautifulSoup(table_html, "html.parser")
    for th in soup.find_all("th"):
        th.name = "td"
    for thead in soup.find_all("thead"):
        thead.unwrap()
    for math_tag in soup.find_all("math"):
        alttext = math_tag.get("alttext", "")
        alttext = f"${alttext}$" if alttext else ""
        math_tag.replace_with(alttext)
    for span in soup.find_all("span"):
        span.unwrap()

    # drop common style/size attrs
    for tag in soup.find_all():
        for attr in ["style", "height", "width", "align", "class"]:
            if attr in tag.attrs:
                tag.attrs.pop(attr, None)

    html_str = html.unescape(str(soup))
    html_str = unicodedata.normalize("NFKC", html_str).strip()
    return html_str


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
    테이블 블록 리스트를 받아 OmniDocBench 스타일 Table-TEDS를 계산.
    - 테이블을 단순 tree(table->tr->cell)로 변환
    - 삽입/삭제 비용: 서브트리 크기
    - 치환 비용: 태그 불일치(1) + (옵션) 셀 텍스트 NED
    - 유사도: exp(-TED / |GT|)
    """

    structure_sims = [0.0] * len(gt_tables)
    semantic_sims = [0.0] * len(gt_tables)

    def _sim_struct(pred_block, gt_block):
        return _table_teds_similarity_omni(pred_block, gt_block, table_type, include_text=False)

    def _sim_sem(pred_block, gt_block):
        return _table_teds_similarity_omni(pred_block, gt_block, table_type, include_text=True)

    pairs = align_sequences(pred_tables, gt_tables, _sim_struct)
    for gi, pi in pairs:
        structure_sims[gi] = _sim_struct(pred_tables[pi], gt_tables[gi])
        semantic_sims[gi] = _sim_sem(pred_tables[pi], gt_tables[gi])

    return {
        "table_teds": average(structure_sims) if structure_sims else 1.0,
        "table_teds_s": average(semantic_sims) if semantic_sims else 1.0,
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


# ---------- OmniDocBench Table TEDS ----------
@dataclass
class _TableNode:
    tag: str
    text: str = ""
    children: List["_TableNode"] = field(default_factory=list)

    def add(self, child: "_TableNode") -> None:
        self.children.append(child)


_WS_RE = re.compile(r"\s+")


def _normalize_cell_text(text: str) -> str:
    if text is None:
        return ""
    return _WS_RE.sub(" ", text.strip())


def _table_block_to_tree(table_block: str, table_type: Optional[str], include_text: bool) -> Optional[_TableNode]:
    """
    Convert a table block into a simple ordered tree (table->tr->cell).
    """
    if not table_block:
        return None

    # posicube 포맷을 HTML 테이블처럼 취급
    if table_type == "posicube":
        rows = _parse_posicube_table(table_block)
    else:
        rows = _parse_html_table(_normalize_html_table(table_block))

    if not rows:
        return None

    root = _TableNode("table")
    for row in rows:
        row_node = _TableNode("tr")
        for cell in row:
            txt = _normalize_cell_text(cell.get("text", "")) if include_text else ""
            row_node.add(_TableNode("cell", txt))
        root.add(row_node)
    return root


def _table_tree_size(node: Optional[_TableNode]) -> int:
    if node is None:
        return 0
    return 1 + sum(_table_tree_size(ch) for ch in node.children)


def _table_ted_distance(a: Optional[_TableNode], b: Optional[_TableNode], include_text: bool) -> float:
    """
    Ordered tree edit distance where ins/del cost = subtree size,
    sub cost = tag mismatch + optional text NED.
    """
    id_map_a: Dict[int, _TableNode] = {}
    id_map_b: Dict[int, _TableNode] = {}
    children_a: Dict[int, List[int]] = {}
    children_b: Dict[int, List[int]] = {}
    size_a: Dict[int, int] = {}
    size_b: Dict[int, int] = {}

    def build_maps(root: Optional[_TableNode], id_map, child_map, size_map):
        def dfs(node: _TableNode) -> int:
            nid = id(node)
            id_map[nid] = node
            child_ids = []
            total = 1
            for ch in node.children:
                cid = dfs(ch)
                child_ids.append(cid)
                total += size_map[cid]
            child_map[nid] = child_ids
            size_map[nid] = total
            return nid

        if root is None:
            return None
        return dfs(root)

    root_a_id = build_maps(a, id_map_a, children_a, size_a)
    root_b_id = build_maps(b, id_map_b, children_b, size_b)

    @lru_cache(maxsize=None)
    def ted(nid_a: Optional[int], nid_b: Optional[int]) -> float:
        if nid_a is None and nid_b is None:
            return 0.0
        if nid_a is None:
            return float(size_b[nid_b])
        if nid_b is None:
            return float(size_a[nid_a])

        na = id_map_a[nid_a]
        nb = id_map_b[nid_b]
        tag_cost = 0.0 if na.tag == nb.tag else 1.0
        text_cost = normalized_edit_distance(na.text, nb.text) if include_text else 0.0
        node_sub_cost = tag_cost + text_cost

        ca = children_a[nid_a]
        cb = children_b[nid_b]
        m, n = len(ca), len(cb)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + size_a[ca[i - 1]]
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + size_b[cb[j - 1]]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                del_cost = size_a[ca[i - 1]]
                ins_cost = size_b[cb[j - 1]]
                sub_cost = ted(ca[i - 1], cb[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + del_cost,
                    dp[i][j - 1] + ins_cost,
                    dp[i - 1][j - 1] + sub_cost,
                )
        return node_sub_cost + dp[m][n]

    return ted(root_a_id, root_b_id)


def _table_teds_similarity_omni(pred_block: str, gt_block: str, table_type: Optional[str], include_text: bool) -> float:
    """
    OmniDocBench Table-TEDS 유사도: exp(-TED / |GT|)
    """
    pred_tree = _table_block_to_tree(pred_block, table_type, include_text)
    gt_tree = _table_block_to_tree(gt_block, table_type, include_text)

    if pred_tree is None and gt_tree is None:
        return 1.0
    if pred_tree is None or gt_tree is None:
        return 0.0

    ted_val = _table_ted_distance(pred_tree, gt_tree, include_text=include_text)
    gt_size = float(_table_tree_size(gt_tree))
    if gt_size == 0:
        return 1.0 if ted_val == 0 else 0.0
    return math.exp(-ted_val / gt_size)


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

    pairs = align_sequences(pred_parsed, gt_parsed, lambda p, g: _table_similarity(p, g, include_text=include_text))
    for gi, pi in pairs:
        sim = _table_similarity(pred_parsed[pi], gt_parsed[gi], include_text=include_text)
        sims[gi] = sim
        if sim >= threshold:
            matched += 1

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
