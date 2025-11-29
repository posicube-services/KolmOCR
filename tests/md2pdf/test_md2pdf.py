import pytest
import json
from pathlib import Path

from lab.md2pdf import md2pdf
from lab.md2pdf.data_structures.output_paths import build_document_paths


def test_collect_markdown_paths_covers_sample_dataset():
    pattern = "tests/sample_dataset/**/*.md"

    paths = md2pdf._collect_markdown_paths(pattern)

    assert len(paths) == 29
    assert paths[0].relative_to(Path("tests/sample_dataset")).as_posix() == "empty_document/blanktext.md"
    assert paths[-1].name == "edgar.md"
    assert all(path.exists() for path in paths)
    assert any("has_table_data" in path.parts for path in paths)


def test_load_markdown_paths_from_json_with_sample_base(tmp_path: Path):
    json_payload = {
        "documents": [
            {
                "document_id": "blank",
                "source_files": {"markdown": "empty_document/blanktext.md"},
                "column_count": 2,
            },
            {
                "document_id": "edgar",
                "source_files": {"markdown": "simple_document/edgar.md"},
            },
        ]
    }
    json_path = tmp_path / "metadata.json"
    json_path.write_text(json.dumps(json_payload), encoding="utf-8")

    results = md2pdf._load_markdown_paths_from_json(json_path)

    assert results == [
        ("blank", Path("tests/sample_dataset/empty_document/blanktext.md"), 2),
        ("edgar", Path("tests/sample_dataset/simple_document/edgar.md"), 1),
    ]


def test_main_stage1_uses_sample_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    input_json = {
        "documents": [
            {
                "document_id": "blank",
                "source_files": {"markdown": "empty_document/blanktext.md"},
                "column_count": 2,
            },
            {
                "document_id": "edgar",
                "source_files": {"markdown": "simple_document/edgar.md"},
            },
        ]
    }
    json_path = tmp_path / "output_metadata.json"
    json_path.write_text(json.dumps(input_json), encoding="utf-8")

    captured = {}

    def fake_stage(
        markdown_paths,
        output_dir,
        print_json=False,
        print_html=False,
        document_ids=None,
        column_counts=None,
    ):
        captured["markdown_paths"] = list(markdown_paths)
        captured["output_dir"] = Path(output_dir)
        captured["document_ids"] = list(document_ids) if document_ids is not None else None
        captured["column_counts"] = list(column_counts) if column_counts is not None else None

    monkeypatch.setattr(md2pdf, "extract_html_and_bbox", fake_stage)

    md2pdf.main(["1", "--input-json", str(json_path), "--output-dir", str(tmp_path)])

    md_relatives = [path.relative_to(Path("tests/sample_dataset")) for path in captured["markdown_paths"]]
    assert md_relatives == [Path("empty_document/blanktext.md"), Path("simple_document/edgar.md")]
    assert captured["document_ids"] == ["blank", "edgar"]
    assert captured["column_counts"] == [2, 1]
    assert captured["output_dir"] == tmp_path


def test_build_document_paths_preserves_hierarchy(tmp_path: Path):
    source = tmp_path / "data" / "docs" / "input.md"
    doc_id = "topic/subtopic/input"
    output_dir = tmp_path / "generated_md"
    paths = build_document_paths(
        source,
        output_dir,
        document_id=doc_id,
        preserve_document_path=True,
    )

    base_dir = output_dir / "topic" / "subtopic"
    assert paths.document_dir == base_dir
    assert paths.md == base_dir / "input.md"
    assert paths.html == base_dir / "input.html"
    assert paths.pdf == base_dir / "input.pdf"


def test_build_document_paths_avoids_leaf_folder(tmp_path: Path):
    source = tmp_path / "data" / "docs" / "input.md"
    doc_id = "topic/subtopic/input"
    output_dir = tmp_path / "generated_md"
    paths = build_document_paths(
        source,
        output_dir,
        document_id=doc_id,
        preserve_document_path=False,  # should still avoid creating leaf dir
    )

    base_dir = output_dir / "topic" / "subtopic"
    assert paths.document_dir == base_dir
    assert paths.md == base_dir / "input.md"
    assert paths.pdf == base_dir / "input.pdf"
