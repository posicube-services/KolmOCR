import os
from datetime import datetime
from typing import List

import pandas as pd

from olmocr.kolmocr_eval.metrics.base import Metric
from olmocr.kolmocr_eval.metrics.common import round_numeric, align_sequences
from olmocr.kolmocr_eval.utils.structure import _build_code_block_tree, parse_code_blocks
from olmocr.kolmocr_eval.utils.tree import tree_edit_distance, tree_size, Node
from olmocr.kolmocr_eval.utils.data_io import list_md_files, read_md


def _extract_code_blocks(md: str) -> list[Node]:
    """
    md에서 코드 블록들을 개별 Node 리스트로 추출.
    fenced + 들여쓰기 코드 블록을 모두 처리하고, Paddle/OmniDocBench처럼
    코드 내부는 정규화 후 트리로 변환한다.
    """
    allowed = {"python", "py", "c", "c++", "cpp", "java"}
    blocks: list[Node] = []
    for lang, body in parse_code_blocks(md):
        label = f"code:{lang}" if lang and lang in allowed else "code:unknown"
        node = Node(label)
        for child in _build_code_block_tree(lang, body):
            node.add(child)
        blocks.append(node)
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

            sims = [0.0] * len(gt_blocks)  # 매칭되지 않은 GT는 0으로 처리
            dists = [tree_size(g) for g in gt_blocks]  # unmatched는 GT 크기만큼 거리로 둔다

            def _sim_block(p, g):
                dist = tree_edit_distance(p, g)
                denom = max(tree_size(p), tree_size(g), 1)
                return max(0.0, 1.0 - dist / denom)

            pairs = align_sequences(pred_blocks, gt_blocks, _sim_block)
            for gi, pi in pairs:
                dist = tree_edit_distance(pred_blocks[pi], gt_blocks[gi])
                dists[gi] = dist
                denom = max(tree_size(pred_blocks[pi]), tree_size(gt_blocks[gi]), 1)
                sims[gi] = max(0.0, 1.0 - dist / denom)

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
