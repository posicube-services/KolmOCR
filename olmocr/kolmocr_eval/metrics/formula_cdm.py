import os
from datetime import datetime
from typing import List

import pandas as pd

from olmocr.kolmocr_eval.metrics.base import Metric
from olmocr.kolmocr_eval.metrics.common import compute_formula_cdm
from olmocr.kolmocr_eval.utils.data_io import list_md_files, read_md


class FormulaCDMEvaluator(Metric):
    name = "formular_CDM"

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
            cdm = compute_formula_cdm(md_pred, md_gt)
            records.append({"filename": rel, "formula_cdm": cdm})

        df = pd.DataFrame(records).sort_values(by="filename")
        avg_row = {
            "filename": "average",
            "formula_cdm": df["formula_cdm"].mean(),
        }
        df = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)

        date_dir = getattr(args, "run_dir", os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
        os.makedirs(date_dir, exist_ok=True)
        output_dir = os.path.join(date_dir, "formular_CDM.csv")
        df.to_csv(output_dir, index=False)
        print(f"[formular_CDM] Saved to {output_dir}")


def run_formula_cdm_eval(args):
    evaluator = FormulaCDMEvaluator()
    evaluator.run(args)
