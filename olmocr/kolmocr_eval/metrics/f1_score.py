import os
from datetime import datetime
from typing import List

import pandas as pd

from olmocr.kolmocr_eval.metrics.base import Metric
from olmocr.kolmocr_eval.metrics.text_edit import Evaluator as TextEditEvaluator
from olmocr.kolmocr_eval.utils.data_io import list_md_files


class F1ScoreEvaluator(Metric):
    """
    Heading precision/recall/F1만 별도로 저장하는 evaluator.
    """

    name = "f1_score"

    def run(self, args):
        gt_rel_paths = list_md_files(args.gt_dir)
        list_pred_dir: List[str] = []
        list_gt_dir: List[str] = []
        sample_ids: List[str] = []
        for rel in gt_rel_paths:
            gt_path = os.path.join(args.gt_dir, rel)
            pred_path = os.path.join(args.pred_dir, rel)
            if not os.path.exists(pred_path):
                print(f"[Warning] pred missing for {rel}, skipping.")
                continue
            list_gt_dir.append(gt_path)
            list_pred_dir.append(pred_path)
            sample_ids.append(rel)

        evaluator = TextEditEvaluator(args.threshold_headings, args.version, include_f1=True)
        df = evaluator.eval_batch(list_pred_dir, list_gt_dir, sample_ids)
        # 필요한 컬럼만 남기기
        keep_cols = [
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
        ]
        df = df[[c for c in keep_cols if c in df.columns]]
                             
        date_dir = getattr(args, "run_dir", os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
        os.makedirs(date_dir, exist_ok=True)
        output_dir = os.path.join(date_dir, "f1_score.csv")
        df.to_csv(output_dir, index=False)
        print(f"[f1_score] Saved to {output_dir}")


def run_f1_score_eval(args):
    evaluator = F1ScoreEvaluator()
    evaluator.run(args)
