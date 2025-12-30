import os
from itertools import zip_longest
from datetime import datetime
from typing import List, Optional

import pandas as pd

from olmocr.kolmocr_eval.metrics.base import Metric
from olmocr.kolmocr_eval.metrics.common import round_numeric, align_sequences
from olmocr.kolmocr_eval.utils.parser import parse_md
from olmocr.kolmocr_eval.utils.data_io import list_md_files, read_md


def _compute_iou(b1: Optional[List[float]], b2: Optional[List[float]]) -> float:
    if not b1 or not b2:
        return 0.0
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _match_bboxes(pred_boxes: List[List[float]], gt_boxes: List[List[float]]):
    """
    Adjacency-style 매칭으로 GT 순서를 유지하며 IoU 합을 극대화.
    - GT가 없으면 스킵
    - pred가 더 적으면 남은 GT는 IoU 0으로 처리
    """
    gt_list = [g for g in gt_boxes if g is not None]
    pred_list = [p for p in pred_boxes if p is not None]
    num_gt = len(gt_list)
    num_pred = len(pred_list)
    if num_gt == 0:
        return {"avg_iou": 1.0, "pairs": 0}

    pairs = align_sequences(pred_list, gt_list, lambda p, g: _compute_iou(p, g))
    matched_ious = [0.0] * num_gt
    for gi, pi in pairs:
        matched_ious[gi] = _compute_iou(pred_list[pi], gt_list[gi])

    avg_iou = sum(matched_ious) / num_gt if num_gt > 0 else 1.0

    return {
        "avg_iou": avg_iou,
        "pairs": num_gt,
    }


class ImageIOUEvaluator(Metric):
    name = "image_iou"

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

            # GT에 이미지가 없으면 평가에서 제외
            if len(parsed_gt["image_bboxes"]) == 0:
                continue
            scores = _match_bboxes(parsed_pred["image_bboxes"], parsed_gt["image_bboxes"])

            records.append(
                {
                    "filename": rel,
                    "image_avg_iou": scores["avg_iou"],
                    "image_pairs": scores["pairs"],
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
        output_dir = os.path.join(date_dir, "image_iou.csv")
        round_numeric(df).to_csv(output_dir, index=False)
        print(f"[image_iou] Saved to {output_dir}")


def run_image_iou_eval(args):
    evaluator = ImageIOUEvaluator()
    evaluator.run(args)
