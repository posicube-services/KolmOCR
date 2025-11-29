"""
Validation script to test the new pipeline stages.
새로운 파이프라인 단계들이 올바르게 작동하는지 테스트합니다.
"""

from pathlib import Path
import sys


def validate_imports():
    """모든 새로운 모듈을 import할 수 있는지 확인."""
    print("✓ Validating imports...")
    try:
        from lab.md2pdf import (
            extract_bbox_from_html,
            render_html_to_pdf_and_png,
            json2markdown,
        )
        print("  ✓ extract_bbox_from_html imported")
        print("  ✓ render_html_to_pdf_and_png imported")
        print("  ✓ json2markdown imported")
        
        from lab.md2pdf.cli_parser import (
            extract_html_and_bbox,
            render_pdf_and_png,
            stage_3_generate_markdown_from_bbox,
        )
        print("  ✓ extract_html_and_bbox imported")
        print("  ✓ stage_2_render_pdf_and_png imported")
        print("  ✓ stage_3_generate_markdown_from_bbox imported")
        
        from lab.md2pdf.md2pdf import main
        print("  ✓ generate_korean_md_dataset.main imported")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def validate_modules():
    """각 모듈의 함수 시그니처를 확인."""
    print("\n✓ Validating module signatures...")
    try:
        from lab.md2pdf.converters.html2json import (
            extract_bbox_from_html,
            is_playwright_available,
        )
        print("  ✓ html2json module OK")
        
        from lab.md2pdf.converters.html2pdf2png import (
            render_html_to_pdf_and_png,
        )
        print("  ✓ html2pdf2png module OK")
        
        from lab.md2pdf.converters.json2md import (
            json2markdown,
        )
        print("  ✓ json2markdown module OK")
        
        return True
    except Exception as e:
        print(f"  ✗ Module validation failed: {e}")
        return False


def validate_cli_functions():
    """CLI 함수들이 올바른 파라미터를 받는지 확인."""
    print("\n✓ Validating CLI function signatures...")
    try:
        from lab.md2pdf.cli_parser import (
            extract_html_and_bbox,
            render_pdf_and_png,
            stage_3_generate_markdown_from_bbox,
        )
        import inspect
        
        # Stage 1 시그니처 확인
        sig1 = inspect.signature(extract_html_and_bbox)
        params1 = list(sig1.parameters.keys())
        assert "markdown_paths" in params1, "stage_1: markdown_paths missing"
        assert "output_dir" in params1, "stage_1: output_dir missing"
        print("  ✓ extract_html_and_bbox signature OK")
        
        # Stage 2 시그니처 확인
        sig2 = inspect.signature(render_pdf_and_png)
        params2 = list(sig2.parameters.keys())
        assert "markdown_paths" in params2, "stage_2: markdown_paths missing"
        assert "output_dir" in params2, "stage_2: output_dir missing"
        print("  ✓ stage_2_render_pdf_and_png signature OK")
        
        # Stage 3 시그니처 확인
        sig3 = inspect.signature(stage_3_generate_markdown_from_bbox)
        params3 = list(sig3.parameters.keys())
        assert "markdown_paths" in params3, "stage_3: markdown_paths missing"
        assert "output_dir" in params3, "stage_3: output_dir missing"
        print("  ✓ stage_3_generate_markdown_from_bbox signature OK")
        
        return True
    except Exception as e:
        print(f"  ✗ CLI function validation failed: {e}")
        return False


def validate_json_format():
    """bbox.json 형식 검증."""
    print("\n✓ Validating bbox.json format...")
    try:
        import json
        from lab.md2pdf.converters.html2json import _BBOX_EXTRACTION_SCRIPT
        
        # 스크립트가 JavaScript 형식인지 확인
        assert "elements" in _BBOX_EXTRACTION_SCRIPT, "bbox script doesn't mention 'elements'"
        assert "bbox" in _BBOX_EXTRACTION_SCRIPT, "bbox script doesn't mention 'bbox'"
        print("  ✓ bbox extraction script format OK")
        
        return True
    except Exception as e:
        print(f"  ✗ JSON format validation failed: {e}")
        return False


def main():
    """모든 검증 실행."""
    print("="*60)
    print("Pipeline Stages Validation")
    print("="*60)
    
    results = []
    
    results.append(("Imports", validate_imports()))
    results.append(("Module signatures", validate_modules()))
    results.append(("CLI functions", validate_cli_functions()))
    results.append(("JSON format", validate_json_format()))
    
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("✓ All validations passed!")
        return 0
    else:
        print("✗ Some validations failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
