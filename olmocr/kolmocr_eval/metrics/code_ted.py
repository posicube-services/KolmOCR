import os
from datetime import datetime
from typing import List

import pandas as pd

from olmocr.kolmocr_eval.metrics.base import Metric
from olmocr.kolmocr_eval.metrics.common import round_numeric
from olmocr.kolmocr_eval.utils.structure import build_code_tree, _build_code_block_tree, CODE_FENCE_PATTERN, CODE_FENCE_LANG_PATTERN
from olmocr.kolmocr_eval.utils.tree import tree_edit_distance, tree_size, Node
from olmocr.kolmocr_eval.utils.data_io import list_md_files, read_md


def _extract_code_blocks(md: str) -> list[Node]:
    """
    md에서 코드 블록들을 개별 Node 리스트로 추출.
    언어 미표기/기타는 code:unknown으로 처리한다.
    """
    allowed = {"python", "py", "c", "c++", "cpp", "java"}
    blocks: list[Node] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        m = CODE_FENCE_LANG_PATTERN.match(lines[i])
        if m:
            lang = (m.group(1) or "").lower()
            body = []
            i += 1
            while i < len(lines) and not CODE_FENCE_PATTERN.match(lines[i]):
                body.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # closing fence
            label = f"code:{lang}" if lang and lang in allowed else "code:unknown"
            node = Node(label)
            for child in _build_code_block_tree(lang, body):
                node.add(child)
            blocks.append(node)
            continue
        i += 1
    return blocks


class CodeBlockTEDEvaluator(Metric):
    """
    코드 블록만 추출해 트리를 만들고 Tree Edit Distance로 유사도를 측정.
    지원 언어: python, c, c++, cpp, java (code fence 언어로 판별).
    """

    name = "code_TED"

    def run(self, args):
        gt_rel_paths = list_md_files(args.gt_dir)

        records: List[dict] = []
        for rel in gt_rel_paths:
            gt_dir = os.path.join(args.gt_dir, rel)
            pred_dir = os.path.join(args.pred_dir, rel)
            if not os.path.exists(pred_dir):
                print(f"[Warning] pred missing for {rel}, skipping.")
                continue

            md_pred = read_md(pred_dir)
            md_gt = read_md(gt_dir)

            # 코드 블록을 개별적으로 추출해 매칭
            gt_blocks = _extract_code_blocks(md_gt)
            pred_blocks = _extract_code_blocks(md_pred)
            has_code_gt = len(gt_blocks) > 0
            if not has_code_gt:
                # GT에 코드 블록이 없으면 제외
                continue

            # GT 수만큼의 최고 유사도 매칭 (중복 없이 greedy)
            combos = []
            for gi, g in enumerate(gt_blocks):
                for pi, p in enumerate(pred_blocks):
                    dist = tree_edit_distance(p, g)
                    denom = max(tree_size(p), tree_size(g), 1)
                    sim = max(0.0, 1.0 - dist / denom)
                    combos.append((sim, dist, gi, pi))
            combos.sort(key=lambda x: x[0], reverse=True)

            sims = [0.0] * len(gt_blocks)  # 매칭되지 않은 GT는 0으로 처리
            dists = [tree_size(g) for g in gt_blocks]  # unmatched는 GT 크기만큼 거리로 둔다
            used_gt = set()
            used_pred = set()
            for sim, dist, gi, pi in combos:
                if gi in used_gt or pi in used_pred:
                    continue
                sims[gi] = sim
                dists[gi] = dist
                used_gt.add(gi)
                used_pred.add(pi)
                if len(used_gt) == len(gt_blocks) or len(used_pred) == len(pred_blocks):
                    break

            sim_avg = sum(sims) / len(gt_blocks) if gt_blocks else 1.0
            dist_avg = sum(dists) / len(gt_blocks) if gt_blocks else 0.0
            nodes_pred = sum(tree_size(p) for p in pred_blocks) or 0
            nodes_gt = sum(tree_size(g) for g in gt_blocks) or 0

            records.append(
                {
                    "filename": rel,
                    "code_ted": sim_avg,
                    "code_ted_distance": dist_avg,
                    "code_nodes_pred": nodes_pred,
                    "code_nodes_gt": nodes_gt,
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame([{"filename": "average"}])
        else:
            df = df.sort_values(by="filename")
            avg_row = {"filename": "average"}
            for col in df.columns:
                if col == "filename":
                    continue
                avg_row[col] = df[col].mean()
            df = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)

        date_dir = getattr(args, "run_dir", os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
        os.makedirs(date_dir, exist_ok=True)
        output_dir = os.path.join(date_dir, "code_TED.csv")
        round_numeric(df).to_csv(output_dir, index=False)
        print(f"[code_TED] Saved to {output_dir}")


def run_code_ted_eval(args):
    evaluator = CodeBlockTEDEvaluator()
    evaluator.run(args)
