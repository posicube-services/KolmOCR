import os
from datetime import datetime
from typing import List

import pandas as pd

from olmocr.kolmocr_eval.metrics.base import Metric
from olmocr.kolmocr_eval.metrics.common import compute_table_scores, round_numeric
from olmocr.kolmocr_eval.utils.data_io import list_md_files, read_md
from olmocr.kolmocr_eval.utils.parser import parse_md


class TableTEDSEvaluator(Metric):
    name = "table_TEDS"

    def run(self, args):
        gt_rel_paths = list_md_files(args.gt_dir)

        records: List[dict] = []
        for rel in gt_rel_paths:
            gt_dir = os.path.join(args.gt_dir, rel)
            pred_dir = os.path.join(args.pred_dir, rel)
            if not os.path.exists(pred_dir):
                print(f"[Warning - TableTEDS] pred missing for {rel}, skipping.")
                continue
            md_pred = read_md(pred_dir)
            md_gt = read_md(gt_dir)
            parsed_pred = parse_md(md_pred)
            parsed_gt = parse_md(md_gt)
            has_table_gt = len(parsed_gt["tables"]) > 0
            if not has_table_gt:
                # GT에 테이블이 없으면 해당 샘플은 table_TEDS에서 제외
                continue
            table_type = parsed_gt["table_type"] or parsed_pred["table_type"]
            table_scores = compute_table_scores(parsed_pred["tables"], parsed_gt["tables"], table_type)
            records.append(
                {
                    "filename": rel,
                    "table_teds": table_scores["table_teds"],
                    "table_teds_s": table_scores["table_teds_s"],
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame([{"filename": "average"}])
        else:
            df = df.sort_values(by="filename")
            avg_row = {"filename": "average"}
            avg_row["table_teds"] = df["table_teds"].mean()
            avg_row["table_teds_s"] = df["table_teds_s"].mean()
            df = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)

        date_dir = getattr(args, "run_dir", os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
        os.makedirs(date_dir, exist_ok=True)
        output_dir = os.path.join(date_dir, "table_TEDS.csv")
        round_numeric(df).to_csv(output_dir, index=False)
        print(f"[table_TEDS] Saved to {output_dir}")


def run_table_teds_eval(args):
    evaluator = TableTEDSEvaluator()
    evaluator.run(args)


def run_table_teds_s_eval(args):
    """
    table_TEDS_S는 table_TEDS와 동일한 산출물을 사용하며,
    출력 csv에 구조/의미 점수가 모두 포함됩니다.
    """
    evaluator = TableTEDSEvaluator()
    evaluator.run(args)
