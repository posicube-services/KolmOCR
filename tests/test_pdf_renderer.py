from pathlib import Path
from subprocess import TimeoutExpired
from unittest.mock import patch

from olmocr.train.dataloader import PDFRenderer

PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAwAB/gb5kQAAAABJRU5ErkJggg=="


def test_pdf_renderer_success_creates_image():
    renderer = PDFRenderer(target_longest_image_dim=1288)
    sample = {"pdf_path": Path("dummy.pdf")}

    with patch("olmocr.train.dataloader.render_pdf_to_base64png", return_value=PNG_BASE64):
        result = renderer(sample)

    assert result is not None
    assert "image" in result
    assert result["image"].size == (1, 1)


def test_pdf_renderer_timeout_is_skipped():
    renderer = PDFRenderer(target_longest_image_dim=1288)
    sample = {"pdf_path": Path("timeout.pdf")}

    with patch(
        "olmocr.train.dataloader.render_pdf_to_base64png",
        side_effect=TimeoutExpired(cmd="pdfinfo", timeout=60),
    ):
        result = renderer(sample)

    assert result is None


def test_pdf_renderer_generic_failure_is_skipped():
    renderer = PDFRenderer(target_longest_image_dim=1288)
    sample = {"pdf_path": Path("broken.pdf")}

    with patch(
        "olmocr.train.dataloader.render_pdf_to_base64png",
        side_effect=RuntimeError("render failed"),
    ):
        result = renderer(sample)

    assert result is None
