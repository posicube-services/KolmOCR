import os
from datetime import datetime
from typing import List

import pandas as pd

from olmocr.kolmocr_eval.metrics.base import Metric
from olmocr.kolmocr_eval.metrics.common import normalized_edit_distance, similarity_from_distance, round_numeric, align_sequences, average
from olmocr.kolmocr_eval.utils.data_io import list_md_files, read_md
from olmocr.kolmocr_eval.metrics.text_edit import _prepare_text_for_text_edit, _split_paragraphs


class ReadingOrderEvaluator(Metric):
    name = "reading_order"

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

            # 문단 단위 블록을 추출해 순서를 비교 (Adjacency alignment)
            text_pred = _prepare_text_for_text_edit(md_pred)
            text_gt = _prepare_text_for_text_edit(md_gt)
            blocks_pred = _split_paragraphs(text_pred)
            blocks_gt = _split_paragraphs(text_gt)

            if not blocks_gt:
                continue

            sims = [0.0] * len(blocks_gt)

            def _sim(p, g):
                return similarity_from_distance(normalized_edit_distance(p, g))

            pairs = align_sequences(blocks_pred, blocks_gt, _sim)
            for gi, pi in pairs:
                sims[gi] = _sim(blocks_pred[pi], blocks_gt[gi])

            reading_score = average(sims) if sims else 1.0
            reading_ned = 1.0 - reading_score

            records.append(
                {
                    "filename": rel,
                    "reading_order": reading_score,
                    "reading_order_ned": reading_ned,
                }
            )

        if not records:
            avg_row = {"filename": "average", "reading_order": 0.0, "reading_order_ned": 0.0}
            df = pd.DataFrame([avg_row])
        else:
            df = pd.DataFrame(records).sort_values(by="filename")
            avg_row = {
                "filename": "average",
                "reading_order": df["reading_order"].mean(),
                "reading_order_ned": df["reading_order_ned"].mean(),
            }
            df = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)

        date_dir = getattr(args, "run_dir", os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
        os.makedirs(date_dir, exist_ok=True)
        output_dir = os.path.join(date_dir, "reading_order.csv")
        round_numeric(df).to_csv(output_dir, index=False)
        print(f"[reading_order] Saved to {output_dir}")


def run_reading_order_eval(args):
    evaluator = ReadingOrderEvaluator()
    evaluator.run(args)
