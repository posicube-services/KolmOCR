from typing import Callable, Dict, List

from olmocr.kolmocr_eval.metrics.f1_score import run_f1_score_eval
from olmocr.kolmocr_eval.metrics.formula_cdm import run_formula_cdm_eval
from olmocr.kolmocr_eval.metrics.code_ted import run_code_ted_eval
from olmocr.kolmocr_eval.metrics.image_iou import run_image_iou_eval
from olmocr.kolmocr_eval.metrics.overall import run_overall_eval
from olmocr.kolmocr_eval.metrics.reading_order import run_reading_order_eval
from olmocr.kolmocr_eval.metrics.table_f1 import run_table_f1_eval
from olmocr.kolmocr_eval.metrics.table_teds import run_table_teds_eval, run_table_teds_s_eval
from olmocr.kolmocr_eval.metrics.text_edit import run_text_edit_eval


def _placeholder(metric_name: str, *_args, **_kwargs):
    print(f"[{metric_name}] Not implemented yet. Skipping.")


REGISTRY: Dict[str, Callable] = {
    "text_edit": run_text_edit_eval,
    "reading_order": run_reading_order_eval,
    "table_TEDS": run_table_teds_eval,
    "table_TEDS_S": run_table_teds_s_eval,
    "formular_CDM": run_formula_cdm_eval,
    "overall": run_overall_eval,
    "f1_score": run_f1_score_eval,
    "table_f1": run_table_f1_eval,
    "image_iou": run_image_iou_eval,
    "code_TED": run_code_ted_eval,
    "TQA": lambda args: _placeholder("TQA (run scripts/tqa.py directly)"),
}


def supported_metrics() -> List[str]:
    return list(REGISTRY.keys())


def run_metric(metric_name: str, args):
    if metric_name not in REGISTRY:
        raise ValueError(f"Unsupported metric: {metric_name}")
    return REGISTRY[metric_name](args)
