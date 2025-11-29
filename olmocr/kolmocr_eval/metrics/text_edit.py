import json
import os
import re
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple

import pandas as pd
from Levenshtein import distance as edit_distance

from olmocr.kolmocr_eval.utils.data_io import (
    read_md,
    remove_images_from_md,
    remove_tables_from_md,
    extract_headings,
    extract_lists,
)
from olmocr.kolmocr_eval.utils.matching import match_texts, f1_score, compute_text_similarity
from olmocr.kolmocr_eval.metrics.base import Metric
from olmocr.kolmocr_eval.metrics.common import round_numeric


INLINE_MATH_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
INLINE_LATEX_PATTERN = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
BLOCK_MATH_PATTERN = re.compile(r"\$\$[\s\S]*?\$\$", re.MULTILINE)
BLOCK_MATH_BRACKET_PATTERN = re.compile(r"\\\[[\s\S]*?\\\]", re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~", re.MULTILINE)
HTML_TABLE_PATTERN = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE | re.MULTILINE)
POSICUBE_TABLE_PATTERN = re.compile(r"<POSICUBE_TABLE_KV>[\s\S]*?</POSICUBE_TABLE_KV>", re.IGNORECASE | re.MULTILINE)
YAML_FRONT_MATTER_PATTERN = re.compile(r"^---[\\s\\S]*?---\\s*", re.MULTILINE)


def normalized_levenshtein(a: str, b: str) -> float:
    """Edit distance / max length (0 best, 1 worst)."""
    if not a and not b:
        return 0.0
    return edit_distance(a, b) / max(len(a), len(b), 1)


def _collapse_repeated_symbols(text: str) -> str:
    """Limit long runs of divider-like symbols to length 3."""
    return re.sub(r"([=_~\\-]{3,})", lambda m: m.group(1)[0] * 3, text)


def _remove_markdown_tables(text: str) -> str:
    """Remove GitHub-style markdown tables."""
    lines = text.splitlines()
    cleaned: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        sep_match = (
            re.match(r"^\s*\|.*\|\s*$", line)
            and i + 1 < len(lines)
            and re.match(r"^\s*\|?\s*[:-]+[-| :]*\|?\s*$", lines[i + 1])
        )
        if sep_match:
            i += 2
            while i < len(lines) and re.match(r"^\s*\|.*\|\s*$", lines[i]):
                i += 1
            continue
        cleaned.append(line)
        i += 1
    return "\n".join(cleaned)


def _strip_inline_math(text: str) -> str:
    """Replace inline math markers with their content."""
    text = INLINE_MATH_PATTERN.sub(lambda m: m.group(1).strip(), text)
    text = INLINE_LATEX_PATTERN.sub(lambda m: m.group(1).strip(), text)
    return text


def _strip_markdown_syntax(text: str) -> str:
    """Remove heading/list/blockquote markers and link wrappers."""
    stripped: List[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*>+\s*", "", line)
        line = re.sub(r"^\s*#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        stripped.append(line)
    text = "\n".join(stripped)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _prepare_text_for_text_edit(md: str, table_version: str = "") -> str:
    """
    PaddleOCR-VL / OmniDocBench 방식에 맞춰 순수 텍스트만 남긴다.
    - 이미지/테이블/코드블록/블록수식 제거
    - 인라인 수식 기호 제거 후 본문에 포함
    - 마크다운 헤더/리스트/링크 문법 제거
    - 반복 심볼/공백 정규화
    """
    text = remove_images_from_md(md)
    text = remove_tables_from_md(text, table_version)
    text = HTML_TABLE_PATTERN.sub(" ", text)
    text = POSICUBE_TABLE_PATTERN.sub(" ", text)
    text = _remove_markdown_tables(text)
    text = CODE_BLOCK_PATTERN.sub(" ", text)
    text = BLOCK_MATH_PATTERN.sub(" ", text)
    text = BLOCK_MATH_BRACKET_PATTERN.sub(" ", text)
    text = YAML_FRONT_MATTER_PATTERN.sub("", text)
    text = _collapse_repeated_symbols(text)
    text = _strip_inline_math(text)
    text = _strip_markdown_syntax(text)
    text = _normalize_whitespace(text)
    return text


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs using blank lines; fallback to line split."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return paragraphs


def adjacency_normalized_edit_distance(
    pred_blocks: List[str], gt_blocks: List[str], max_merge: int = 3
) -> Tuple[float, List[float]]:
    """
    Adjacency search style 매칭: 인접 문단을 병합해 더 낮은 NED 조합을 찾는다.
    반환값: (평균 NED, 매칭별 NED 목록)
    """
    if not pred_blocks and not gt_blocks:
        return 0.0, []

    i = j = 0
    distances: List[float] = []
    while i < len(pred_blocks) and j < len(gt_blocks):
        best_dist = normalized_levenshtein(pred_blocks[i], gt_blocks[j])
        best_p, best_g = 1, 1
        for p_span in range(1, max_merge + 1):
            for g_span in range(1, max_merge + 1):
                if i + p_span > len(pred_blocks) or j + g_span > len(gt_blocks):
                    continue
                merged_pred = " ".join(pred_blocks[i : i + p_span])
                merged_gt = " ".join(gt_blocks[j : j + g_span])
                cand = normalized_levenshtein(merged_pred, merged_gt)
                if cand + 1e-12 < best_dist:
                    best_dist = cand
                    best_p, best_g = p_span, g_span
        distances.append(best_dist)
        i += best_p
        j += best_g

    while i < len(pred_blocks):
        distances.append(1.0)
        i += 1
    while j < len(gt_blocks):
        distances.append(1.0)
        j += 1

    avg_dist = sum(distances) / len(distances) if distances else 0.0
    return avg_dist, distances


class AverageMeter_PrecisionRecall:
    def __init__(self, name):
        self.name = name
        self.num_correct = 0
        self.num_pred = 0
        self.num_total = 0

    def update(self, num_correct, num_pred, num_batch):
        self.num_correct += num_correct
        self.num_pred += num_pred
        self.num_total += num_batch
        precision = 0 if num_pred == 0 else num_correct / num_pred
        recall = 0 if num_batch == 0 else num_correct / num_batch
        return precision, recall

    def get_avg_pr(self):
        precision = 0 if self.num_pred == 0 else self.num_correct / self.num_pred
        recall = 0 if self.num_total == 0 else self.num_correct / self.num_total
        if self.num_total == 0:
            precision, recall = -1, -1
        return precision, recall


class Evaluator_Headings:
    def __init__(self, threshold: int):
        self.heading_text_meter = AverageMeter_PrecisionRecall("headings_text")
        self.heading_level_meter = AverageMeter_PrecisionRecall("headings_level")
        self.threshold = threshold
        self.result_total = []

    def eval(self, pred, target, debug=False):
        if len(pred) == 0 or len(target) == 0:
            self.heading_text_meter.update(0, len(pred), len(target))
            self.heading_level_meter.update(0, len(pred), len(target))
            precision_text, recall_text, precision_level, recall_level = 0, 0, 0, 0
            if len(pred) == 0 and len(target) == 0:
                precision_text, recall_text, precision_level, recall_level = 1, 1, 1, 1
            self.result_total.append([precision_text, recall_text, precision_level, recall_level])
        else:
            pred_text_list = []
            target_text_list = []
            for p in pred:
                pred_text_list.append(p["text"])
            for t in target:
                target_text_list.append(t["text"])
            p2t_matching_dict, t2p_matching_dict = match_texts(pred_text_list, target_text_list, self.threshold)
            precision_text, recall_text = self.heading_text_meter.update(
                (len(p2t_matching_dict) + len(t2p_matching_dict)) / 2, len(pred), len(target)
            )

            p2t_num_correct_level = 0
            for k, v in p2t_matching_dict.items():
                if pred[k]["level"] == target[v]["level"]:
                    p2t_num_correct_level += 1

            t2p_num_correct_level = 0
            for k, v in t2p_matching_dict.items():
                if target[k]["level"] == pred[v]["level"]:
                    t2p_num_correct_level += 1

            precision_level, recall_level = self.heading_level_meter.update(
                (p2t_num_correct_level + t2p_num_correct_level) / 2, len(pred), len(target)
            )
            self.result_total.append([precision_text, recall_text, precision_level, recall_level])

        if debug:
            if len(pred) == 0 or len(target) == 0:
                return precision_text, recall_text, precision_level, recall_level, {}, {}
            else:
                return precision_text, recall_text, precision_level, recall_level, p2t_matching_dict, t2p_matching_dict
        else:
            return precision_text, recall_text, precision_level, recall_level

    def get_summary(self):
        p_text, r_text = self.heading_text_meter.get_avg_pr()
        f1_text = f1_score(p_text, r_text)

        p_level, r_level = self.heading_level_meter.get_avg_pr()
        f1_level = f1_score(p_level, r_level)

        summary = {
            "heading_structure_p": p_text,
            "heading_structure_r": r_text,
            "heading_structure_f1": f1_text,
            "level_p": p_level,
            "level_r": r_level,
            "level_f1": f1_level,
        }

        return summary


class Evaluator(Metric):
    name = "text_edit"

    def __init__(self, threshold_heading, version, include_f1: bool = True):
        self.evaluator_heading = Evaluator_Headings(threshold_heading)
        self.evaluator_lists = Evaluator_Headings(threshold_heading)
        self.version = version
        self.include_f1 = include_f1

    def eval(self, pred_dir, gt_dir):
        md_pred_raw = read_md(pred_dir)
        md_gt_raw = read_md(gt_dir)

        table_version = self.version if self.version else ""
        md_pred = remove_images_from_md(md_pred_raw)
        md_pred = remove_tables_from_md(md_pred, table_version)
        md_gt = remove_images_from_md(md_gt_raw)
        md_gt = remove_tables_from_md(md_gt, table_version)

        hd_pred = extract_headings(md_pred)
        hd_gt = extract_headings(md_gt)
        p_t, r_t, p_l, r_l, d1, d2 = self.evaluator_heading.eval(hd_pred, hd_gt, True)

        # eval lists
        li_pred = [{"text": t, "level": 1, "type": "list"} for t in extract_lists(md_pred)]
        li_gt = [{"text": t, "level": 1, "type": "list"} for t in extract_lists(md_gt)]
        p_li, r_li, _, _, _, _ = self.evaluator_lists.eval(li_pred, li_gt, True)

        da = {}
        for i in range(len(hd_pred)):
            v = d1.get(i, "-")
            key_heading = "#" * hd_pred[i]["level"] + " " + hd_pred[i]["text"]
            if v == "-":
                value_heading = "-"
            else:
                value_heading = "#" * hd_gt[v]["level"] + " " + hd_gt[v]["text"]
            da[key_heading] = value_heading

        db = {}
        for i in range(len(hd_gt)):
            v = d2.get(i, "-")
            key_heading = "#" * hd_gt[i]["level"] + " " + hd_gt[i]["text"]
            if v == "-":
                value_heading = "-"
            else:
                value_heading = "#" * hd_pred[v]["level"] + " " + hd_pred[v]["text"]
            db[key_heading] = value_heading

        d = {
            "pred_to_target": da,
            "target_to_pred": db,
        }

        text_pred = _prepare_text_for_text_edit(md_pred_raw, table_version)
        text_gt = _prepare_text_for_text_edit(md_gt_raw, table_version)
        if text_pred.strip() == "" and text_gt.strip() == "":
            return None
        pred_paragraphs = _split_paragraphs(text_pred)
        gt_paragraphs = _split_paragraphs(text_gt)
        text_edit_ned, paragraph_neds = adjacency_normalized_edit_distance(pred_paragraphs, gt_paragraphs)
        text_edit_sim = max(0.0, 1.0 - text_edit_ned)
        flattened_pred = "\n\n".join(pred_paragraphs)
        flattened_gt = "\n\n".join(gt_paragraphs)
        if flattened_pred.strip() == "" and flattened_gt.strip() == "":
            contents_sim = 1.0
        elif flattened_pred.strip() == "" or flattened_gt.strip() == "":
            contents_sim = 0.0
        else:
            contents_sim = compute_text_similarity(flattened_pred, flattened_gt)

        result_dict = {
            "heading_structure_p": p_t,
            "heading_structure_r": r_t,
            "heading_structure_f1": f1_score(p_t, r_t),
            "level_p": p_l,
            "level_r": r_l,
            "level_f1": f1_score(p_l, r_l),
            "list_p": p_li,
            "list_r": p_li,
            "list_f1": f1_score(p_li, p_li),
            "text_edit": text_edit_ned,
            "text_edit_sim": text_edit_sim,
            "paragraph_pairs": len(paragraph_neds),
            "contents_sim": contents_sim,
            "has_heading_gt": len(hd_gt) > 0,
            "has_list_gt": len(li_gt) > 0,
        }

        return result_dict

    def eval_batch(self, list_pred_dir, list_gt_dir, sample_ids):
        result_dict = defaultdict(list)

        kept_sample_ids = []
        for pred_dir, gt_dir, sid in zip(list_pred_dir, list_gt_dir, sample_ids):
            r = self.eval(pred_dir, gt_dir)
            if r is None:
                print(f"[text_edit] Skipped {sid} (no text).")
                continue
            for k, v in r.items():
                result_dict[k].append(v)
            kept_sample_ids.append(sid)

        ordered_cols = [
            "filename",
            "heading_structure_p",
            "heading_structure_r",
            "heading_structure_f1",
            "level_p",
            "level_r",
            "level_f1",
            "list_p",
            "list_r",
            "list_f1",
            "text_edit",
            "text_edit_sim",
            "paragraph_pairs",
            "contents_sim",
        ]

        if len(kept_sample_ids) == 0:
            return pd.DataFrame(columns=ordered_cols)

        result_csv = pd.DataFrame(data=result_dict)
        result_csv["filename"] = kept_sample_ids
        result_csv = result_csv.sort_values(by="filename")

        # 평균 계산: GT에 요소가 있는 샘플만 해당 요소 평균에 포함
        avg = {"filename": "average"}
        mask_heading = result_csv["has_heading_gt"] == True if "has_heading_gt" in result_csv else result_csv.index == result_csv.index
        mask_list = result_csv["has_list_gt"] == True if "has_list_gt" in result_csv else result_csv.index == result_csv.index
        subset_h = result_csv[mask_heading] if mask_heading.any() else result_csv
        subset_l = result_csv[mask_list] if mask_list.any() else result_csv

        for col in ["heading_structure_p", "heading_structure_r", "heading_structure_f1", "level_p", "level_r", "level_f1"]:
            if col in result_csv:
                avg[col] = subset_h[col].mean()
        for col in ["list_p", "list_r", "list_f1"]:
            if col in result_csv:
                avg[col] = subset_l[col].mean()

        for col in ["text_edit", "text_edit_sim", "paragraph_pairs"]:
            if col in result_csv:
                avg[col] = result_csv[col].mean()
        avg["contents_sim"] = result_csv["contents_sim"].mean() if "contents_sim" in result_csv else -1

        df_avg = pd.DataFrame(avg, index=[0])
        # average 추가 후 플래그 제거
        result_csv = pd.concat([df_avg, result_csv], ignore_index=True)
        for flag_col in ["has_heading_gt", "has_list_gt"]:
            if flag_col in result_csv:
                result_csv = result_csv.drop(columns=[flag_col])
        result_csv.insert(0, "filename", result_csv.pop("filename"))

        if not self.include_f1:
            result_csv["heading_structure_f1"] = "N/A"
            result_csv["level_f1"] = "N/A"
            result_csv["list_f1"] = "N/A"

        result_csv = result_csv[[c for c in ordered_cols if c in result_csv.columns]]

        return round_numeric(result_csv)

    def run(self, args):
        from olmocr.kolmocr_eval.utils.data_io import list_md_files

        gt_rel_paths = list_md_files(args.gt_dir)
        list_pred_dir = []
        list_gt_dir = []
        sample_ids = []
        for rel in gt_rel_paths:
            gt_path = os.path.join(args.gt_dir, rel)
            pred_path = os.path.join(args.pred_dir, rel)
            if not os.path.exists(pred_path):
                print(f"[Warning] pred missing for {rel}, skipping.")
                continue
            list_gt_dir.append(gt_path)
            list_pred_dir.append(pred_path)
            sample_ids.append(rel)

        r_df = self.eval_batch(list_pred_dir, list_gt_dir, sample_ids)
        date_dir = getattr(args, "run_dir", os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
        os.makedirs(date_dir, exist_ok=True)
        output_dir = os.path.join(date_dir, "text_edit.csv")
        r_df.to_csv(output_dir, index=False)
        print(f"[text_edit] Saved to {output_dir}")


def run_text_edit_eval(args):
    evaluator = Evaluator(args.threshold_headings, args.version, include_f1=args.text_include_f1)
    evaluator.run(args)
