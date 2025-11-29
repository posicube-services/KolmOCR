import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
import shutil

import yaml

from olmocr.kolmocr_eval.metrics.registry import run_metric, supported_metrics
from olmocr.kolmocr_eval.metrics.common import round_numeric
from olmocr.kolmocr_eval.utils.data_io import render_csv_table_image

PROJECT_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)


def main():
    default_config = PROJECT_ROOT / "configs" / "kolmocr_eval.yaml"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="YAML config path (defaults to configs/kolmocr_eval.yaml). CLI flags override config values.",
    )
    parser.add_argument("--pred_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="결과 csv를 저장할 경로 (기존 output_dir도 호환)",
    )
    parser.add_argument(
        "--threshold_headings",
        type=int,
        default=1,
        help="matching이 heading을 찾기 위한 edit distance 문턱치, 이 값 이하인 것 중 최초 만나는 하나가 매치된다. ",
    )
    parser.add_argument(
        "--threshold_table",
        type=float,
        default=0.6,
        help="table_f1 계산 시 테이블 매칭으로 인정할 최소 유사도(0~1). 구조/내용 모두 동일 문턱값을 사용.",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["1.9", "1.10"],
        default=None,
        help="모델 버전 (선택 입력, text_edit에서 1.9일 때만 추가 전처리 수행)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=supported_metrics(),
        default=["text_edit"],
        help="실행할 평가 metric 선택. 여러 개 입력 가능. 미입력 시 text_edit만 수행.",
    )
    parser.add_argument(
        "--text_include_f1",
        action="store_true",
        default=True,
        help="text_edit 수행 시 heading F1 관련 컬럼을 포함할지 여부 (기본 포함).",
    )
    parser.add_argument(
        "--no_text_f1",
        dest="text_include_f1",
        action="store_false",
        help="text_edit 수행 시 heading F1 컬럼을 제외합니다.",
    )
    # 1) Pre-parse to read config path
    pre_args, _ = parser.parse_known_args()

    # 2) Load config (if exists) as parser defaults so CLI flags override
    if pre_args.config:
        cfg_path = Path(pre_args.config)
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            valid_dests = {a.dest for a in parser._actions if hasattr(a, "dest")}
            filtered_cfg = {k: v for k, v in cfg.items() if k in valid_dests}
            parser.set_defaults(**filtered_cfg)
            logger.info("[config] Loaded defaults from %s", cfg_path)
        else:
            logger.warning("[config] Not found: %s. Using CLI defaults.", cfg_path)

    # 3) Parse final args with config defaults + CLI overrides
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not args.output_dir:
        raise ValueError("output_dir is required.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_dir = os.path.join(str(output_root), run_tag)
    os.makedirs(args.run_dir, exist_ok=True)

    log_path = Path(args.run_dir) / "evaluate.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to %s", log_path)

    def order_columns(cols):
        preferred = [
            "element",
            "metric",
            # user-preferred front order
            "heading_structure_f1",
            "list_f1",
            "table_struct_f1",
            "code_ted",
            "image_avg_iou",
            "text_edit",
            "table_teds",
            "table_teds_s",
            # table
            "table_struct_p",
            "table_struct_r",
            "table_struct_avg_sim",
            "table_semantic_p",
            "table_semantic_r",
            "table_semantic_f1",
            "table_semantic_avg_sim",
            # image
            "image_pairs",
            # code
            "code_ted_distance",
            "code_nodes_pred",
            "code_nodes_gt",
            # text / heading / list
            "heading_structure_p",
            "heading_structure_r",
            "level_p",
            "level_r",
            "level_f1",
            "list_p",
            "list_r",
            # text similarity / edit
            "text_edit_ned",
            "contents_sim",
            # reading order
            "reading_order",
            "reading_order_ned",
            # overall & others
            "overall",
            "formula_cdm",
        ]
        ordered = [c for c in preferred if c in cols]
        ordered.extend([c for c in cols if c not in ordered])
        return ordered

    output_files = {}
    avg_rows_dict = {}
    for metric in args.metrics:
        run_metric(metric, args)
        # 예상 파일명을 기록
        metric_to_filename = {
            "text_edit": "text_edit.csv",
            "reading_order_edit": "reading_order_edit.csv",
            "table_TEDS": "table_TEDS.csv",
            "table_TEDS_S": "table_TEDS.csv",
            "formular_CDM": "formular_CDM.csv",
            "overall": "overall.csv",
            "f1_score": "f1_score.csv",
            "table_f1": "table_f1.csv",
            "image_iou": "image_iou.csv",
            "code_TED": "code_TED.csv",
        }
        fname = metric_to_filename.get(metric)
        if fname:
            output_files[metric] = os.path.join(args.run_dir, fname)

    # 메트릭별 average 행을 모아 average.csv 작성
    try:
        import pandas as pd

        rows = []
        for metric, path in output_files.items():
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if "filename" not in df.columns:
                continue
            avg_rows = df[df["filename"] == "average"]
            if avg_rows.empty:
                continue
            avg_row = avg_rows.iloc[0].to_dict()
            avg_row["metric"] = metric
            # filename은 metric과 중복이므로 제거
            avg_row.pop("filename", None)
            rows.append(avg_row)
            avg_rows_dict[metric] = avg_row

        if rows:
            avg_df = pd.DataFrame(rows)
            summary = {"metric": "average"}
            for col in avg_df.columns:
                if col in ["element", "metric"]:
                    continue
                series = avg_df[col]
                if pd.api.types.is_numeric_dtype(series):
                    summary[col] = series.mean()
                else:
                    first_nonnull = next((v for v in series if pd.notna(v)), None)
                    summary[col] = first_nonnull
            summary_df = pd.DataFrame([summary])
            summary_df = summary_df[order_columns(list(summary_df.columns))]
            avg_path = os.path.join(args.run_dir, "average.csv")
            round_numeric(summary_df).to_csv(avg_path, index=False)
            logger.info("[average] Saved to %s", avg_path)

        # 요소별 평균 요약
        if avg_rows_dict:
            # 모든 컬럼의 합집합 (metric/element 제외)
            all_cols = set()
            for v in avg_rows_dict.values():
                all_cols.update([k for k in v.keys() if k != "metric"])
            ordered_cols = order_columns(["element"] + list(all_cols))

            element_map = {
                "heading": (
                    [
                        "heading_structure_p",
                        "heading_structure_r",
                        "heading_structure_f1",
                        "level_p",
                        "level_r",
                        "level_f1",
                        "list_p",
                        "list_r",
                        "list_f1",
                    ],
                    ["text_edit", "f1_score"],
                ),
                "table": (
                    [
                        "table_struct_p",
                        "table_struct_r",
                        "table_struct_f1",
                        "table_semantic_p",
                        "table_semantic_r",
                        "table_semantic_f1",
                        "table_teds",
                        "table_teds_s",
                    ],
                    ["table_f1", "table_TEDS"],
                ),
                "list": ([], []),
                "code-block": (
                    [
                        "code_ted",
                        "code_ted_distance",
                        "code_nodes_pred",
                        "code_nodes_gt",
                    ],
                    ["code_TED"],
                ),
                "image": (
                    [
                        "image_avg_iou",
                        "image_pairs",
                    ],
                    ["image_iou"],
                ),
            }

            element_rows = []
            for elem, (cols, metric_priority) in element_map.items():
                row = {c: "N/A" for c in ordered_cols}
                row["element"] = elem
                # 컬럼별로 metric 우선순위에 따라 채움 (없으면 N/A 유지)
                for c in cols:
                    for m in metric_priority:
                        if m not in avg_rows_dict:
                            continue
                        val = avg_rows_dict[m].get(c)
                        if pd.isna(val):
                            continue
                        row[c] = val
                        break
                element_rows.append(row)

            elem_df = pd.DataFrame(element_rows)[ordered_cols]
            elem_path = os.path.join(args.run_dir, "element_average.csv")
            round_numeric(elem_df).to_csv(elem_path, index=False)
            logger.info("[element_average] Saved to %s", elem_path)

            # NIPA 제출용 테이블: 핵심 지표만 요약
            def _nipa_value(metric_name: str, col: str):
                if metric_name not in avg_rows_dict:
                    return "N/A"
                val = avg_rows_dict[metric_name].get(col, "N/A")
                if pd.isna(val):
                    return "N/A"
                return val

            nipa_rows = [
                {"Element": "Text", "F1-score": _nipa_value("text_edit", "text_edit_sim")},
                {"Element": "Heading", "F1-score": _nipa_value("text_edit", "heading_structure_f1")},
                {"Element": "List", "F1-score": _nipa_value("text_edit", "list_f1")},
                {"Element": "Table", "F1-score": _nipa_value("table_f1", "table_struct_f1")},
                {"Element": "Image IoU", "F1-score": _nipa_value("image_iou", "image_avg_iou")},
                {"Element": "Code-Block", "F1-score": _nipa_value("code_TED", "code_ted")},
            ]
            nipa_df = pd.DataFrame(nipa_rows, columns=["Element", "F1-score"])
            nipa_df_rounded = round_numeric(nipa_df)
            nipa_path = os.path.join(args.run_dir, "nipa_table.csv")
            nipa_df_rounded.to_csv(nipa_path, index=False)
            logger.info("[nipa_table] Saved to %s", nipa_path)

            # Markdown 버전도 함께 저장 (NIPA 테이블을 리포트에 바로 삽입 가능하게)
            def _df_to_markdown_table(df: pd.DataFrame) -> str:
                headers = list(df.columns)
                lines = [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join("---" for _ in headers) + " |",
                ]
                for _, row in df.iterrows():
                    cells = []
                    for val in row:
                        if pd.isna(val):
                            cells.append("")
                        else:
                            cells.append(str(val))
                    lines.append("| " + " | ".join(cells) + " |")
                return "\n".join(lines) + "\n"

            nipa_md_path = os.path.join(args.run_dir, "nipa_table.md")
            with open(nipa_md_path, "w", encoding="utf-8") as f:
                f.write(_df_to_markdown_table(nipa_df_rounded))
            logger.info("[nipa_table] Markdown saved to %s", nipa_md_path)
            # Also place a copy at output root for quick access.
            try:
                nipa_md_copy = output_root / "nipa_table.md"
                shutil.copyfile(nipa_md_path, nipa_md_copy)
                logger.info("[nipa_table] Markdown copy saved to %s", nipa_md_copy)
            except Exception as e:
                logger.warning("[nipa_table] Failed to copy markdown to output root: %s", e)
            try:
                img_path = render_csv_table_image(nipa_path)
                logger.info("[nipa_table] Image saved to %s", img_path)
            except Exception as e:
                logger.warning("[nipa_table] Failed to render image: %s", e)
    except Exception as e:
        logger.warning("[average] Skipped creating average.csv due to error: %s", e)


if __name__ == "__main__":
    main()
