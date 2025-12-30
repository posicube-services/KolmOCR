from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import pytest

from lab.utils.text.markdown import parse_markdown_elements
from lab.md2pdf.md2pdf import main

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "sampled_kolmocr_bench"
REFERENCE_DIR = PROJECT_ROOT / "tests/sampled_kolmocr_bench_md2pdf_testset"
GENERATED_DIR = PROJECT_ROOT / "tests/sampled_kolmocr_bench_md2pdf_testset_tmp"
IMAGE_MD_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)")
MAX_N_ENV_VAR = "MD2PDF_KOLMOCR_MAX_N"
ENV_MAX_N = os.environ.get(MAX_N_ENV_VAR)


def _run_md2pdf(argv: list[str]) -> None:
    try:
        main(argv)
    except SystemExit as exc:
        if exc.code != 0:
            pytest.fail(f"md2pdf exited with {exc.code}")
    except Exception as exc:  # pragma: no cover - surfaced as pytest failure
        pytest.fail(f"md2pdf failed with exception: {exc}")


def _count_image_refs(text: str) -> int:
    return len(IMAGE_MD_PATTERN.findall(text or ""))


def _count_images_in_lists(elements: list[dict]) -> int:
    images = 0
    for element in elements:
        if element.get("type") != "list":
            continue
        for item in element.get("items", []):
            if isinstance(item, str):
                images += _count_image_refs(item)
    return images


def _count_images_in_tables(elements: list[dict]) -> int:
    images = 0
    for element in elements:
        if element.get("type") != "table":
            continue
        # table markdown keeps inline image syntax intact; html may not
        table_markdown = element.get("markdown") or ""
        images += _count_image_refs(table_markdown)
    return images


def _collect_markdown_files(root: Path) -> dict[str, Path]:
    """Return doc_id -> markdown path mapping for all *.md under root."""
    mapping: dict[str, Path] = {}
    for md_path in sorted(root.rglob("*.md")):
        if md_path.is_file():
            doc_id = md_path.relative_to(root).with_suffix("").as_posix()
            mapping[doc_id] = md_path
    return mapping


def _summarize_dir(root: Path, doc_ids: list[str] | None = None) -> dict[str, dict[str, int]]:
    paths = _collect_markdown_files(root)
    if doc_ids is not None:
        paths = {doc_id: paths[doc_id] for doc_id in doc_ids if doc_id in paths}

    summaries: dict[str, dict[str, int]] = {}
    for doc_id, md_path in paths.items():
        raw_text = md_path.read_text(encoding="utf-8")
        elements = parse_markdown_elements(raw_text)
        summaries[doc_id] = {
            "images": _count_image_refs(raw_text),
            "headings": sum(1 for el in elements if el.get("type") == "heading"),
            "lists": sum(1 for el in elements if el.get("type") == "list"),
            "images_in_lists_or_tables": _count_images_in_lists(elements) + _count_images_in_tables(elements),
        }
    return summaries


_ref_doc_ids = sorted(_collect_markdown_files(REFERENCE_DIR).keys())

# Keep param set in sync with --max-n to avoid expecting docs we did not generate.
if ENV_MAX_N:
    try:
        _max_n_val = int(ENV_MAX_N)
    except ValueError:
        _max_n_val = None
    else:
        _ref_doc_ids = _ref_doc_ids[:_max_n_val]


@pytest.fixture(scope="session")
def md2pdf_summaries():
    if not INPUT_DIR.exists():
        pytest.skip("kolmocr_bench directory not found")
    if not REFERENCE_DIR.exists():
        pytest.skip("reference output directory not found")
    if GENERATED_DIR.exists():
        shutil.rmtree(GENERATED_DIR)

    argv = ["all", "--input-dir", str(INPUT_DIR), "--output-dir", str(GENERATED_DIR)]
    if ENV_MAX_N:
        try:
            max_n_val = int(ENV_MAX_N)
        except ValueError:
            pytest.fail(f"{MAX_N_ENV_VAR} must be an integer")
        argv.extend(["--max-n", str(max_n_val)])

    _run_md2pdf(argv)

    ref_summary = _summarize_dir(REFERENCE_DIR, _ref_doc_ids)
    gen_summary = _summarize_dir(GENERATED_DIR)

    shutil.rmtree(GENERATED_DIR, ignore_errors=True)
    return ref_summary, gen_summary


@pytest.mark.parametrize("doc_id", _ref_doc_ids, ids=_ref_doc_ids)
def test_md2pdf_element_counts_match_reference_per_doc(md2pdf_summaries, doc_id):
    """
    문서별로 이미지/heading/list/리스트·테이블 내 이미지 개수가 레퍼런스와 동일한지 검증한다.
    """
    ref_summary, gen_summary = md2pdf_summaries
    assert doc_id in gen_summary, f"Generated output missing doc: {doc_id}"
    assert gen_summary[doc_id] == ref_summary[doc_id], f"Count mismatch for {doc_id}: {gen_summary[doc_id]} vs {ref_summary[doc_id]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
