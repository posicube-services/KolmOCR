import argparse
from pathlib import Path
from typing import List

import pytest

from lab.tr_md.main import ColumnAnalysisResult
from lab.mdg_dataset import process_md_direct as pipeline

STAGE_DATASET = Path("tests/sample_dataset/stage_test")


def _stage_args() -> argparse.Namespace:
    return argparse.Namespace(
        server_url="http://localhost:8000",
        model_name="qwen-vl",
        max_tokens=512,
        temperature=0.0,
        max_concurrent=1,
        target_image_dim=512,
        detection_image_dim=256,
    )


def test_split_markdown_counts_ranges() -> None:
    state = pipeline._extract_doc_ele_markdown(
        STAGE_DATASET / "document.md",
        "stage_test/document",
    )
    assert state.figure_count == 0
    assert state.table_count == 0
    assert state.formula_count == 2
    assert len(state.text_ranges) == 4


def test_normalize_bbox_and_text_blocks() -> None:
    normalized = pipeline._normalize_bbox([0, 0, 500, 500], 1000, 1000)
    assert normalized is not None
    assert normalized.x == 0
    assert normalized.y == 0
    assert normalized.width == 500
    assert normalized.height == 500

    boxes: List[pipeline.NormalizedElement] = [
        pipeline.NormalizedElement(0, 0, 100, 20),
        pipeline.NormalizedElement(0, 200, 100, 20),
    ]
    blocks = pipeline._build_text_blocks(boxes, column_count=1)
    assert len(blocks) == 2
    assert blocks[0].column_index == 0
    assert blocks[1].center_y >= blocks[0].center_y


def test_pipeline_stages_write_annotations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = pipeline._extract_doc_ele_markdown(
        STAGE_DATASET / "document.md",
        "stage_test/document",
    )
    state.pdf_path = STAGE_DATASET / "document.pdf"

    args = _stage_args()
    monkeypatch.setattr(pipeline, "analyze_columns", _fake_analyze_columns)
    monkeypatch.setattr(pipeline, "_render_pdf_page", lambda pdf_path, target, dim: (1000, 1200))

    async def fake_run_layout_detection(detector, image_path):
        return ([
            {
                "boxes": [
                    {"label": "text", "coordinate": [0, 0, 1000, 200]},
                    {"label": "text", "coordinate": [0, 400, 1000, 600]},
                    {"label": "text", "coordinate": [0, 900, 1000, 1100]},
                    {"label": "text", "coordinate": [0, 1300, 1000, 1500]},
                ]
            }
        ], None)

    monkeypatch.setattr(pipeline, "run_layout_detection", fake_run_layout_detection)
    monkeypatch.setattr(pipeline, "VLLMClient", _DummyVLLMClient)

    pipeline._column_stage([state], STAGE_DATASET, args, tmp_path)
    pipeline._dla_stage([state], args, object(), tmp_path)
    state.figure_count = 1
    state.figure_ranges = [pipeline.MarkdownBlock("figure", 0, 0)]
    state.dla_figures = [pipeline.NormalizedElement(0, 0, 1000, 200, label="figure")]
    pipeline._match_elements_stage([state], tmp_path)
    pipeline._rewrite_md_stage([state], STAGE_DATASET, tmp_path)

    annotated = tmp_path / "annotated" / "document" / "document.md"
    assert annotated.exists()
    lines = annotated.read_text().splitlines()
    comment_lines = [line for line in lines if line.startswith("<!-- bbox:")]
    assert len(comment_lines) == 1
    end_markers = [line for line in lines if line.strip() == "<!-- bbox_blk_end -->"]
    assert len(end_markers) == 1
    pdf_copy = tmp_path / "annotated" / "document" / "document.pdf"
    assert pdf_copy.exists()


def test_mul_column_dataset_split() -> None:
    doc_path = Path(
        "tests/sample_dataset/mul_column_data/documents/0222/6b570831ee1cdf68b601450e4023369892b6-3.md"
    )
    state = pipeline._extract_doc_ele_markdown(
        doc_path,
        "mul_column_data/0222/6b570831ee1cdf68b601450e4023369892b6-3",
    )
    assert state.pdf_path == doc_path.with_suffix(".pdf")
    expected_figures = [
        pipeline.MarkdownBlock("figure", 13, 13),
        pipeline.MarkdownBlock("figure", 16, 16),
        pipeline.MarkdownBlock("figure", 38, 38),
        pipeline.MarkdownBlock("figure", 43, 43),
    ]
    assert state.figure_count == len(expected_figures)
    assert state.figure_ranges == expected_figures
    assert state.table_count == 0
    assert state.formula_count == 0
    assert state.text_ranges[0].type == "text"


async def _fake_analyze_columns(
    pdf_path, input_root, render_cache, client, doc_registry
):
    for doc in doc_registry.values():
        doc.column_analysis = ColumnAnalysisResult(
            page=1,
            column_count=1,
            confidence=1.0,
            raw_response="",
        )


class _DummyVLLMClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_DummyVLLMClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    async def chat(self, messages):
        return None
