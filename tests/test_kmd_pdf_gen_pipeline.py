import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from lab.utils.pipeline.bucketing import load_or_create_pdf_list, pdf_list_state_path
from lab.utils.document.model import ColumnAnalysisResult, DocumentResult, TranslationResult
from lab.utils.pipeline.runner import PipelineRunner

try:
    from lab.mdg_dataset.main import (
        AnalyzeColumnsStep,
        CropFiguresStep,
        GenerateDatasetStep,
        KmdPdfGenRunner,
        TranslateMarkdownStep,
    )
except ModuleNotFoundError as exc:
    pytest.skip(f"Missing dependency {exc.name}; cannot import KMD pipeline", allow_module_level=True)

REF_OUTPUT_ROOT = Path("tests/sample_dataset_tr_ref")


async def _fake_crop_execute(
    self,
    pdfs: list[str],
    doc_registry: Dict[str, Any],
    bucket_name: str,
    step_logger,
    **kwargs,
) -> None:
    input_root = Path(kwargs["input_root"])
    for pdf_path in pdfs:
        doc_id = PipelineRunner.make_document_id(str(pdf_path), str(input_root))
        doc = DocumentResult(document_id=doc_id)
        doc.matched_fig_cnt = True
        doc_registry[doc_id] = doc


async def _fake_columns_execute(
    self,
    pdfs: list[str],
    doc_registry: Dict[str, Any],
    bucket_name: str,
    step_logger,
    **kwargs,
) -> None:
    for doc in doc_registry.values():
        if isinstance(doc, DocumentResult):
            doc.column_analysis = ColumnAnalysisResult(
                page=1,
                column_count=2,
                confidence=0.9,
                raw_response="mock",
            )


async def _fake_translate_execute(
    self,
    pdfs: list[str],
    doc_registry: Dict[str, Any],
    bucket_name: str,
    step_logger,
    **kwargs,
) -> None:
    for doc_id, doc in doc_registry.items():
        if isinstance(doc, DocumentResult):
            doc.translation = TranslationResult(
                output_path=f"translated/{doc_id}.md",
                raw_text="translated",
                elements=[],
            )


async def _fake_generator_execute(
    self,
    pdfs: list[str],
    doc_registry: Dict[str, Any],
    bucket_name: str,
    step_logger,
    **kwargs,
) -> None:
    assert any(isinstance(doc, DocumentResult) and doc.translation for doc in doc_registry.values())


@pytest.mark.asyncio
async def test_kmd_pdf_gen_runner_sequence(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir(parents=True)
    output_root.mkdir(parents=True)

    pdf_path = input_root / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.5\n%%EOF")
    (input_root / "doc.md").write_text("# Title\n", encoding="utf-8")

    state_path = pdf_list_state_path(output_root)
    load_or_create_pdf_list([pdf_path], bucket_size=1, state_path=state_path)

    runner = KmdPdfGenRunner(
        input_root=input_root,
        output_root=output_root,
        data_bucket_size=1,
    )

    monkeypatch.setattr(CropFiguresStep, "execute", _fake_crop_execute)
    monkeypatch.setattr(AnalyzeColumnsStep, "execute", _fake_columns_execute)
    monkeypatch.setattr(TranslateMarkdownStep, "execute", _fake_translate_execute)
    monkeypatch.setattr(GenerateDatasetStep, "execute", _fake_generator_execute)

    await runner.run(
        data_bucket_range="1",
        input_exclude_globs=(),
        dataset_max_n=None,
        resume=False,
        target_image_dim=512,
        detection_dpi=100,
        crop_dpi=50,
        dataset_print_json=False,
        dataset_print_html=False,
        md2pdf_max_concurrency=1,
        md2pdf_html_concurrency=1,
    )

    registry_file = output_root / "bucket_states" / "bucket_1_registry.json"
    assert registry_file.exists(), "Doc registry should be persisted"
    persisted = json.loads(registry_file.read_text())
    assert len(persisted) == 1
    doc_id, doc_payload = next(iter(persisted.items()))
    assert doc_payload["matched_fig_cnt"] is True
    assert doc_payload.get("column_analysis", {}).get("column_count") == 2
    assert doc_payload.get("translation", {}).get("output_path")


@pytest.mark.asyncio
async def test_kmd_pdf_gen_runner_sample_dataset(tmp_path, monkeypatch):
    sample_root = Path("tests/sample_dataset/simple_document")
    assert sample_root.is_dir()

    input_root = tmp_path / "sample_input"
    shutil.copytree(sample_root, input_root)
    pdf_paths = sorted(input_root.rglob("*.pdf"))
    assert pdf_paths, "Sample dataset must contain a PDF"

    output_root = tmp_path / "sample_output"
    output_root.mkdir(parents=True)

    runner = KmdPdfGenRunner(
        input_root=input_root,
        output_root=output_root,
        data_bucket_size=1,
    )

    monkeypatch.setattr(CropFiguresStep, "execute", _fake_crop_execute)
    monkeypatch.setattr(AnalyzeColumnsStep, "execute", _fake_columns_execute)
    monkeypatch.setattr(TranslateMarkdownStep, "execute", _fake_translate_execute)
    monkeypatch.setattr(GenerateDatasetStep, "execute", _fake_generator_execute)

    state_path = pdf_list_state_path(output_root)
    load_or_create_pdf_list(pdf_paths, bucket_size=1, state_path=state_path)

    await runner.run(
        data_bucket_range="all",
        input_exclude_globs=(),
        dataset_max_n=None,
        resume=False,
        target_image_dim=512,
        detection_dpi=100,
        crop_dpi=50,
        dataset_print_json=False,
        dataset_print_html=False,
        md2pdf_max_concurrency=1,
        md2pdf_html_concurrency=1,
    )

    registry_file = output_root / "bucket_states" / "bucket_1_registry.json"
    assert registry_file.exists()


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _collect_hashes(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue
        relative = str(file_path.relative_to(directory))
        hashes[relative] = _hash_file(file_path)
    return hashes


@pytest.mark.asyncio
async def test_kmd_pdf_gen_runner_all_sample_datasets(tmp_path, monkeypatch):

    monkeypatch.setattr(CropFiguresStep, "execute", _fake_crop_execute)
    monkeypatch.setattr(AnalyzeColumnsStep, "execute", _fake_columns_execute)
    monkeypatch.setattr(TranslateMarkdownStep, "execute", _fake_translate_execute)
    monkeypatch.setattr(GenerateDatasetStep, "execute", _fake_generator_execute)

    out_root = Path.cwd() / "outputs" / "test_kmd_pdf_gen_runner_all_sample_datasets"
    out_root.mkdir(exist_ok=True)
    input_root = Path("tests/sample_dataset")

    pdf_paths = sorted(input_root.rglob("*.pdf"))

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    state_path = pdf_list_state_path(out_root)
    load_or_create_pdf_list(pdf_paths, bucket_size=1, state_path=state_path)

    runner = KmdPdfGenRunner(
        input_root=input_root,
        output_root=out_root,
        data_bucket_size=1,
    )

    await runner.run(
        data_bucket_range="all",
        input_exclude_globs=(),
        dataset_max_n=None,
        resume=False,
        target_image_dim=512,
        detection_dpi=100,
        crop_dpi=50,
        dataset_print_json=False,
        dataset_print_html=False,
        md2pdf_max_concurrency=1,
        md2pdf_html_concurrency=1,
    )
    dataset_name = "sample_dataset"
    bucket_states_dir = out_root / "bucket_states"
    assert bucket_states_dir.exists(), f"No bucket states for {dataset_name}"
    registry_files = sorted(bucket_states_dir.glob("bucket_*_registry.json"))
    assert registry_files, f"No registries for {dataset_name}"

    for registry_file in registry_files:
        persisted = json.loads(registry_file.read_text())
        assert persisted
        for doc_payload in persisted.values():
            assert doc_payload["matched_fig_cnt"] is True
            assert doc_payload.get("column_analysis", {}).get("column_count") == 2
            translation_path = doc_payload.get("translation", {}).get("output_path")
            assert translation_path and translation_path.startswith("translated/")


@pytest.mark.skipif(
    os.environ.get("REAL_OUTPUT_TEST") is None,
    reason="Requires running the full KMD pipeline against outputs/ref (set REAL_OUTPUT_TEST=1)",
)
def test_kmd_pdf_gen_runner_reference_outputs(tmp_path):
    assert REF_OUTPUT_ROOT.exists(), "Reference outputs must exist"
    runner_output = Path("outputs") / "test_kmd_pdf_gen_runner_reference_outputs"
    if runner_output.exists():
        shutil.rmtree(runner_output)
    runner_output.mkdir(parents=True)

    command = [
        sys.executable,
        "-m",
        "lab.mdg_dataset.kmd_pdf_gen_refactored",
        "--input-root",
        str(Path("tests/sample_dataset")),
        "--input-exclude-glob",
        "hugging_face/**",
        "--output-root",
        str(runner_output),
        "--tr-server-url",
        "http://localhost:8000",
        "--data-bucket-size",
        "64",
        "--tr-max-concurrent",
        "16",
        "--md2pdf-max-concurrency",
        "16",
        "--md2pdf-html-concurrency",
        "1",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    subprocess.run(command, check=True, env=env)
    command_with_bucket = command + [
        "--data-bucket-range",
        "1-1",
        "--resume",
    ]
    subprocess.run(command_with_bucket, check=True, env=env)

    for deterministic_dir in ("dataset_output", "dla"):
        expected_dir = REF_OUTPUT_ROOT / deterministic_dir
        actual_dir = runner_output / deterministic_dir
        assert expected_dir.exists() and expected_dir.is_dir(), f"Missing reference tree {deterministic_dir}"
        assert actual_dir.exists() and actual_dir.is_dir(), f"Pipeline did not produce {deterministic_dir}"
        for expected_file in expected_dir.rglob("*"):
            if not expected_file.is_file():
                continue
            relative = expected_file.relative_to(expected_dir)
            actual_file = actual_dir / relative
            assert actual_file.exists(), f"{deterministic_dir}/{relative} missing from pipeline output"
            expected_size = expected_file.stat().st_size
            actual_size = actual_file.stat().st_size
            ext = expected_file.suffix.lower()
            if ext in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
                tolerance = 30000
            elif ext in {".json", ".md", ".html", ".txt", ".csv"}:
                tolerance = 2000
            elif ext in {".pdf", ".pkl"}:
                tolerance = 30000
            else:
                tolerance = 500
            assert abs(actual_size - expected_size) <= tolerance, f"{deterministic_dir}/{relative}: size {actual_size} deviates from {expected_size}"
