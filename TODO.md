# TODO

- Math/LaTeX 유틸 단일화: `lab/common/latex_utils.py`로 수식 정규화 기능을 모으고, `lab/md2pdf/latex_utils.py`는 shim으로 유지. `new_latex_utils.py`, `json2md.py`의 변환 로직과 `blocks2html.py`의 fallback 흐름을 공용 함수로 교체.
- Playwright + bbox 공용화: `html2json.py`와 `html2pdf2png.py`의 Playwright 설치/대기/bbox 스크립트/시각화 중복을 `lab/common/html_rendering.py`(또는 유사 경로)로 묶고 두 모듈이 재사용하도록 변경.
- Markdown 전처리/구조 공용화: front-matter 파싱, `FrontMatter`/`Block` dataclass를 `lab/common/markdown/` 하위로 올리고, md2pdf 파서와 다른 파이프라인이 같은 구현을 사용하도록 정리.
- 이미지 자산 처리 유틸 분리: `cli_parser._resolve_image_paths`의 이미지 복사/경로 재작성 로직을 공용 유틸(예: `lab/common/assets.py`)로 분리해 다른 파이프라인에서도 동일 규칙을 사용하도록 함.
- 산출물 모델 정렬: Stage 3(`write_bbox2md`)가 만드는 document/figure/translation 구조를 `lab/common/document_model`의 dataclass(`DocumentResult`, `FigureResult`, `TranslationResult`) 기반으로 교체하고 `to_dict()` 사용. `output_paths` shim은 유지하되 내부 import를 `lab/common/document_paths.py`로 정렬.
- PDF/PNG bbox 변환 일원화: `html2pdf2png.py`의 `BboxTransformInfo`와 DPI/픽셀 제한 로직을 `lab/common/pdf_rendering.py`(또는 별도 공용 모듈)로 옮겨 HTML→PDF→PNG 변환이 동일한 스케일/클램프 규칙을 따르도록 함.
- 파이프라인 실행기 정비: Stage 1/2/3를 `PipelineStep`으로 감싸 `lab/common/pipeline_runner.py` 기반 러너를 도입(`lab/md2pdf/pipeline/steps.py`, `runner.py` 등)하고 CLI는 인자 파싱 + 러너 호출로 단순화.
