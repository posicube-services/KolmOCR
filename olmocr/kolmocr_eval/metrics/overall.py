import os
from datetime import datetime
from typing import List

import pandas as pd

from olmocr.kolmocr_eval.metrics.base import Metric
from olmocr.kolmocr_eval.metrics.common import (
    compute_formula_cdm,
    compute_table_scores,
    compute_text_reading_scores,
    average,
    round_numeric,
)
from olmocr.kolmocr_eval.utils.data_io import list_md_files, read_md
from olmocr.kolmocr_eval.utils.parser import parse_md


class OverallEvaluator(Metric):
    """
    OmniDocBench 스타일 주요 지표(TextEdit/ReadingOrder/TableTEDS/TableTEDS-S/FormulaCDM)를 종합.
    """

    name = "overall"

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
            parsed_pred = parse_md(md_pred)
            parsed_gt = parse_md(md_gt)
            has_table_gt = len(parsed_gt["tables"]) > 0

            text_scores = compute_text_reading_scores(parsed_pred["text"], parsed_gt["text"])
            table_type = parsed_gt["table_type"] or parsed_pred["table_type"]
            table_scores = compute_table_scores(parsed_pred["tables"], parsed_gt["tables"], table_type)
            formula_cdm = compute_formula_cdm(md_pred, md_gt)

            overall = average(
                [
                    text_scores["text_edit"],
                    text_scores["reading_order"],
                    table_scores["table_teds"],
                    table_scores["table_teds_s"],
                    formula_cdm,
                ]
            )

            record = {
                "filename": rel,
                "overall": overall,
                "text_edit": text_scores["text_edit"],
                "text_edit_ned": text_scores["text_edit_ned"],
                "reading_order": text_scores["reading_order"],
                "reading_order_ned": text_scores["reading_order_ned"],
                "table_teds": table_scores["table_teds"],
                "table_teds_s": table_scores["table_teds_s"],
                "formula_cdm": formula_cdm,
                "has_table_gt": has_table_gt,
            }
            records.append(record)

        if not records:
            df = pd.DataFrame(
                [
                    {
                        "filename": "average",
                        "overall": 0.0,
                        "text_edit": 0.0,
                        "text_edit_ned": 0.0,
                        "reading_order": 0.0,
                        "reading_order_ned": 0.0,
                        "table_teds": 0.0,
                        "table_teds_s": 0.0,
                        "formula_cdm": 0.0,
                    }
                ]
            )
        else:
            df = pd.DataFrame(records).sort_values(by="filename")
            avg_row = {"filename": "average"}
            for col in df.columns:
                if col == "filename":
                    continue
                if col in ["table_teds", "table_teds_s"]:
                    mask = df["has_table_gt"] == True
                    subset = df[mask] if mask.any() else df
                    avg_row[col] = subset[col].mean()
                elif col == "has_table_gt":
                    continue
                else:
                    avg_row[col] = df[col].mean()
            df = pd.concat([pd.DataFrame([avg_row]), df.drop(columns=["has_table_gt"])], ignore_index=True)
            df = df[
                [
                    "filename",
                    "overall",
                    "text_edit",
                    "text_edit_ned",
                    "reading_order",
                    "reading_order_ned",
                    "table_teds",
                    "table_teds_s",
                    "formula_cdm",
                ]
            ]

        date_dir = getattr(args, "run_dir", os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
        os.makedirs(date_dir, exist_ok=True)
        output_dir = os.path.join(date_dir, "overall.csv")
        round_numeric(df).to_csv(output_dir, index=False)
        print(f"[overall] Saved to {output_dir}")


def run_overall_eval(args):
    evaluator = OverallEvaluator()
    evaluator.run(args)
