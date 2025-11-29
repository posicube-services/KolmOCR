"""
Tests for convert_html.py CLI tool.
"""
from __future__ import annotations

import pytest

import json
import tempfile
from pathlib import Path



from lab.md2pdf.html2png import main
from lab.md2pdf.md2pdf import (
    _load_markdown_paths_from_json,
)
from lab.md2pdf.cli_parser import (
    extract_html_and_bbox,
)


@pytest.fixture
def test_input_json(tmp_path: Path) -> Path:
    """Create a minimal test JSON file with one document."""
    test_data = {
        "generated_at": "2025-11-18T08:09:25.025558Z",
        "documents": [
            {
                "document_id": "has_eq_data/documents/a120/49cb0a97a037ddcf1b59c75388ac6a961d0a-20",
                "source_files": {
                    "markdown": "has_eq_data/documents/a120/49cb0a97a037ddcf1b59c75388ac6a961d0a-20.md",
                    "pdf": "has_eq_data/documents/a120/49cb0a97a037ddcf1b59c75388ac6a961d0a-20.pdf"
                },
                "figures": [],
                "translation": {
                    "output_path": "has_eq_data/documents/a120/49cb0a97a037ddcf1b59c75388ac6a961d0a-20.md",
                    "raw_text": "정리 A.3. 가설 1과 2 하에서, 모든 \\( \\kappa \\geq 2 \\) 와 \\( \\delta > 0 \\) 에 대해 다음이 성립한다:\n\\[\n\\gamma > d - 2 + \\alpha,\n\\]\n에 대해\n\\[\n\\mathbb{E} \\sup_{t \\in [0,T]} |z^\\delta_s(t)|_{\\tilde{H}^\\kappa} \\leq c_{\\kappa,\\gamma}(T) \\delta^{-\\frac{\\gamma}{2}}, \\quad \\delta \\in (0,1).\n\\tag{A.8}\n\\]",
                    "elements": []
                }
            }
        ]
    }
    
    json_file = tmp_path / "test_input.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    return json_file


@pytest.fixture
def sample_dataset_base() -> Path:
    """Get the sample dataset base path."""
    return Path(__file__).parent / "sample_dataset"


class TestConvertHtmlCLI:
    """Tests for convert_html.py CLI."""
    
    def test_load_markdown_paths_from_json(self, test_input_json: Path, sample_dataset_base: Path) -> None:
        """Test loading markdown paths from JSON with proper base path."""
        # Load paths with base_path
        doc_id_paths_cols = _load_markdown_paths_from_json(
            test_input_json,
            base_path=sample_dataset_base
        )
        
        assert len(doc_id_paths_cols) == 1
        doc_id, md_path, column_count = doc_id_paths_cols[0]
        
        # Check document ID
        assert doc_id == "has_eq_data/documents/a120/49cb0a97a037ddcf1b59c75388ac6a961d0a-20"
        
        # Check path has been properly resolved with base path
        expected_path = sample_dataset_base / "has_eq_data/documents/a120/49cb0a97a037ddcf1b59c75388ac6a961d0a-20.md"
        assert md_path == expected_path
        
        # Check default column count
        assert column_count == 1
    
    def test_load_markdown_paths_from_json_default_base_path(self, test_input_json: Path) -> None:
        """Test loading markdown paths with default base path (tests/sample_dataset)."""
        doc_id_paths_cols = _load_markdown_paths_from_json(test_input_json)
        
        assert len(doc_id_paths_cols) == 1
        doc_id, md_path, column_count = doc_id_paths_cols[0]
        
        # Check that default base path is applied
        assert str(md_path).startswith("tests/sample_dataset")
        assert "has_eq_data" in str(md_path)
        
        # Check default column count
        assert column_count == 1
    
    def test_html_to_pdf_png_conversion(self, sample_dataset_base: Path) -> None:
        """Test HTML to PDF/PNG conversion with real markdown file."""
        # Check if sample markdown file exists
        md_file = sample_dataset_base / "has_eq_data/documents/a120/49cb0a97a037ddcf1b59c75388ac6a961d0a-20.md"
        
        if not md_file.exists():
            pytest.skip(f"Sample markdown file not found: {md_file}")
        
        # Stage 1: Convert markdown to HTML and extract bounding boxes
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            # Run stage 1 to generate HTML and JSON
            extract_html_and_bbox(
                [md_file],
                output_dir,
                document_ids=["has_eq_data/documents/a120/49cb0a97a037ddcf1b59c75388ac6a961d0a-20"],
            )
            
            # Check that output files were created with new structure
            # New structure: output_dir/{last_part_of_doc_id}/{last_part_of_doc_id}.{ext}
            doc_dir = output_dir / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20"
            expected_html = doc_dir / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20.html"
            expected_json = doc_dir / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20.blocks.json"
            
            assert expected_html.exists(), f"HTML file not created: {expected_html}"
            assert expected_json.exists(), f"JSON file not created: {expected_json}"
            
            # Test convert_html CLI with the generated HTML
            result = main([
                str(expected_html),
                "--output-dir", str(output_dir),
                "--dpi", "200"
            ])
            
            assert result == 0, "convert_html.py should return 0 on success"
            
            # Check that PDF and PNG were created with new structure
            expected_pdf = output_dir / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20" / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20.pdf"
            expected_png = output_dir / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20" / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20.png"
            
            assert expected_pdf.exists(), f"PDF file not created: {expected_pdf}"
            assert expected_png.exists(), f"PNG file not created: {expected_png}"
    
    def test_convert_html_nonexistent_file(self) -> None:
        """Test convert_html with non-existent file."""
        result = main(["/nonexistent/path/to/file.html"])
        assert result == 1, "Should return 1 for non-existent file"
    
    def test_convert_html_with_custom_output_dir(self, sample_dataset_base: Path) -> None:
        """Test convert_html with custom output directory."""
        md_file = sample_dataset_base / "has_eq_data/documents/a120/49cb0a97a037ddcf1b59c75388ac6a961d0a-20.md"
        
        if not md_file.exists():
            pytest.skip(f"Sample markdown file not found: {md_file}")
        
        with tempfile.TemporaryDirectory() as tmp_dir1:
            with tempfile.TemporaryDirectory() as tmp_dir2:
                output_dir1 = Path(tmp_dir1)
                output_dir2 = Path(tmp_dir2)
                
                # Generate HTML from markdown
                extract_html_and_bbox(
                    [md_file],
                    output_dir1,
                    document_ids=["has_eq_data/documents/a120/49cb0a97a037ddcf1b59c75388ac6a961d0a-20"],
                )
                
                doc_dir = output_dir1 / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20"
                html_file = doc_dir / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20.html"
                
                if not html_file.exists():
                    pytest.skip(f"HTML file not created: {html_file}")
                
                # Convert HTML to PDF/PNG with custom output directory
                result = main([
                    str(html_file),
                    "--output-dir", str(output_dir2),
                    "--dpi", "150"
                ])
                
                assert result == 0
                
                # Verify output is in custom directory with new structure
                # Output structure: {output_dir}/{last_part_of_doc_id}/{last_part_of_doc_id}.{ext}
                expected_pdf = output_dir2 / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20" / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20.pdf"
                expected_png = output_dir2 / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20" / "49cb0a97a037ddcf1b59c75388ac6a961d0a-20.png"
                
                assert expected_pdf.exists(), f"PDF not in custom dir: {expected_pdf}"
                assert expected_png.exists(), f"PNG not in custom dir: {expected_png}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
