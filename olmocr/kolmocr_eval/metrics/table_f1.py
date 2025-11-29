import os
from datetime import datetime
from typing import List

import pandas as pd

from olmocr.kolmocr_eval.metrics.base import Metric
from olmocr.kolmocr_eval.metrics.common import compute_table_f1_scores, round_numeric
from olmocr.kolmocr_eval.utils.data_io import list_md_files, read_md
from olmocr.kolmocr_eval.utils.parser import parse_md


class TableF1Evaluator(Metric):
    name = "table_f1"

    def run(self, args):
        gt_rel_paths = list_md_files(args.gt_dir)
        threshold = getattr(args, "threshold_table")

        records: List[dict] = []
        for rel in gt_rel_paths:
            gt_dir = os.path.join(args.gt_dir, rel)
            pred_dir = os.path.join(args.pred_dir, rel)
            if not os.path.exists(pred_dir):
                print(f"[Warning] pred missing for {rel}, skipping.")
                continue

            md_pred = read_md(pred_dir)
            md_gt = read_md(gt_dir)
            parsed_pred = parse_md(md_pred)
            parsed_gt = parse_md(md_gt)
            table_type = parsed_gt["table_type"] or parsed_pred["table_type"]
            has_table_gt = len(parsed_gt["tables"]) > 0
            if not has_table_gt:
                # GT에 테이블이 없으면 table_f1 계산에서 제외
                continue

            struct_scores = compute_table_f1_scores(
                parsed_pred["tables"], parsed_gt["tables"], table_type, threshold=threshold, include_text=False
            )
            semantic_scores = compute_table_f1_scores(
                parsed_pred["tables"], parsed_gt["tables"], table_type, threshold=threshold, include_text=True
            )

            records.append(
                {
                    "filename": rel,
                    "table_struct_p": struct_scores["precision"],
                    "table_struct_r": struct_scores["recall"],
                    "table_struct_f1": struct_scores["f1"],
                    "table_struct_avg_sim": struct_scores["avg_similarity"],
                    "table_semantic_p": semantic_scores["precision"],
                    "table_semantic_r": semantic_scores["recall"],
                    "table_semantic_f1": semantic_scores["f1"],
                    "table_semantic_avg_sim": semantic_scores["avg_similarity"],
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
        output_dir = os.path.join(date_dir, "table_f1.csv")
        round_numeric(df).to_csv(output_dir, index=False)
        print(f"[table_f1] Saved to {output_dir}")


def run_table_f1_eval(args):
    evaluator = TableF1Evaluator()
    evaluator.run(args)
