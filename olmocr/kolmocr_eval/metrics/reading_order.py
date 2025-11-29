import os
from datetime import datetime
from typing import List

import pandas as pd

from olmocr.kolmocr_eval.metrics.base import Metric
from olmocr.kolmocr_eval.metrics.common import normalized_edit_distance, similarity_from_distance, round_numeric
from olmocr.kolmocr_eval.utils.data_io import list_md_files, read_md


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
            # 문서 전체 라인 시퀀스 기반 읽기 순서 비교 (빈 줄 제외, 양끝 공백 제거)
            reading_pred = "\n".join([ln.strip() for ln in md_pred.splitlines() if ln.strip()])
            reading_gt = "\n".join([ln.strip() for ln in md_gt.splitlines() if ln.strip()])
            reading_ned = normalized_edit_distance(reading_pred, reading_gt)
            reading_score = similarity_from_distance(reading_ned)

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
