"""Unit tests covering shared document model helpers."""

from pathlib import Path

from lab.utils.document.model import (
    ColumnAnalysisResult,
    DocumentResult,
    FigureResult,
    reorder_figures,
    replace_fig_placeholders,
)


def _write_dummy_image(root: Path, rel_path: str) -> Path:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"content")
    return target


def test_reorder_figures_respects_layout_order(tmp_path: Path) -> None:
    doc = DocumentResult(document_id="doc1")
    doc.column_analysis = ColumnAnalysisResult(
        page=1,
        column_count=2,
        confidence=0.8,
        raw_response="{}",
    )

    left_top = FigureResult(
        alt_text="top-left",
        coordinates={"x": 10.0, "y": 60.0, "width": 20.0, "height": 20.0},
        output_path="figs/layout_page1_image00.png",
        status="detected",
    )
    right = FigureResult(
        alt_text="right",
        coordinates={"x": 120.0, "y": 20.0, "width": 10.0, "height": 10.0},
        output_path="figs/layout_page1_image01.png",
        status="detected",
    )
    left_bottom = FigureResult(
        alt_text="bottom-left",
        coordinates={"x": 12.0, "y": 10.0, "width": 15.0, "height": 15.0},
        output_path="figs/layout_page1_image02.png",
        status="detected",
    )

    doc.figures = [left_top, right, left_bottom]
    for figure in doc.figures:
        _write_dummy_image(tmp_path, figure.output_path)

    reorder_figures(doc, str(tmp_path))

    assert [figure.alt_text for figure in doc.figures] == [
        "top-left",
        "bottom-left",
        "right",
    ]
    assert [figure.output_path for figure in doc.figures] == [
        "figs/layout_page1_image01.png",
        "figs/layout_page1_image02.png",
        "figs/layout_page1_image03.png",
    ]
    for figure in doc.figures:
        assert (tmp_path / figure.output_path).exists()


def test_replace_fig_placeholders_emits_relative_paths(tmp_path: Path) -> None:
    doc = DocumentResult(document_id="doc2")
    output_path = "figs/new-figure.png"
    doc.figures = [
        FigureResult(
            alt_text="figure",
            coordinates={"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
            output_path=output_path,
            status="detected",
        )
    ]
    _write_dummy_image(tmp_path, output_path)

    target_md = tmp_path / "translated" / "doc.md"
    target_md.parent.mkdir(parents=True, exist_ok=True)
    text = "intro\n![caption](page_0_0_0_0.png)\nend"

    replaced = replace_fig_placeholders(
        doc,
        str(target_md),
        str(tmp_path),
        text,
    )

    assert replaced == "intro\n![caption](../figs/new-figure.png)\nend"
