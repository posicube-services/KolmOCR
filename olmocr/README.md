# KolmOCR 트랜스포머 이미지 추론

`olmocr/inference_kolmocr_transformer.py`는 디렉터리 안의 이미지(PNG/JPG/JPEG)를 순회하며 KolmOCR(Qwen2.5-VL 계열) 체크포인트로 마크다운을 생성합니다. 입력 폴더 구조를 그대로 따라 `output_dir` 아래에 `.md` 파일을 저장합니다.

## 준비 사항

- KolmOCR(Qwen2.5-VL) 체크포인트 경로(`--checkpoint`)와 대응되는 프로세서가 로컬에 있어야 합니다.
- 입력 이미지가 있는 디렉터리(`--input-dir`)와 결과를 쓸 디렉터리(`--output-dir`)를 지정합니다.
- 기본 설정은 `configs/kolmocr_infer.yaml`를 사용하며, CLI 인자가 우선합니다.
- GPU 사용을 권장하며, CUDA가 없으면 자동으로 CPU/float32로 동작합니다.

## 빠른 실행 예시

```bash
# config에 checkpoint/input/output 등을 정의해 두고 실행
python olmocr/inference_kolmocr_transformer.py \
  --config configs/kolmocr_infer.yaml

# 또는 CLI로 직접 지정
python olmocr/inference_kolmocr_transformer.py \
  --checkpoint /path/to/kolmocr-checkpoint \
  --input-dir data/images \
  --output-dir output/markdown \
  --prompt "이 문서를 구조화된 markdown으로 만들어줘." \
  --max-new-tokens 4000 \
  --temperature 0.7
```

## 주요 옵션

- `--config` (기본: `configs/kolmocr_infer.yaml`): YAML로 기본값을 설정. CLI 인자가 있으면 그 값이 우선.
- `--checkpoint`: 모델/프로세서가 들어 있는 체크포인트 경로. 필수.
- `--input-dir`: 이미지를 스캔할 루트 디렉터리. 필수, 재귀적으로 읽음.
- `--output-dir`: 생성된 마크다운을 저장할 디렉터리. 필수, 입력 구조를 그대로 복제.
- `--prompt`: 모델에게 줄 지시문. 기본값은 "Read this document."
- `--max-new-tokens`: 생성 토큰 최대 길이(기본 1024; config에서 조정 가능).
- `--temperature`: 0이면 greedy, 그 외에는 `top_p`(기본 0.9)와 함께 샘플링.
- `--device-map`: HuggingFace 디바이스 맵(예: `auto`, `balanced`). CUDA가 있으면 기본 `auto`.
- `--use-slow-processor`: 필요한 경우 fast tokenizer 대신 slow tokenizer 사용.

## 참고

- 실행 로그는 로드된 체크포인트, 처리된 파일, 소요 시간을 출력합니다.
- 입력/체크포인트 경로가 잘못되면 즉시 에러 로그를 내고 종료합니다.
- 생성된 마크다운에는 모델이 응답한 본문만 저장되며, 템플릿 프롬프트는 자동으로 제거됩니다.
