# MDGenEval

Markdown Generation Evaluation Tool and Data

## Usage

### Basic Command (Recommended)

The easiest invocation is from the repository root with the shared config:

```bash
python -m olmocr.kolmocr_eval.scripts.evaluate \
  --config configs/kolmocr_eval.yaml
```

### Direct Usage (from repository root)

```bash
python olmocr/kolmocr_eval/scripts/evaluate.py \
  --pred_dir <예측 md 디렉토리> \
  --gt_dir <GT md 디렉토리> \
  --output_dir <결과 csv 저장 경로> \
  --metrics text_edit table_TEDS code_TED image_iou
```

### Examples

```bash
# 설정 파일을 사용한 평가 (권장)
python olmocr/kolmocr_eval/scripts/evaluate.py \
  --config configs/kolmocr_eval.yaml

# 기본 평가 (text_edit만)
python olmocr/kolmocr_eval/scripts/evaluate.py \
  --pred_dir olmocr/kolmocr_eval/data/test_data/test_pred \
  --gt_dir olmocr/kolmocr_eval/data/test_data/sample_dataset \
  --output_dir output

# 모든 메트릭 실행
python olmocr/kolmocr_eval/scripts/evaluate.py \
  --pred_dir olmocr/kolmocr_eval/data/test_data/test_pred \
  --gt_dir olmocr/kolmocr_eval/data/test_data/sample_dataset \
  --output_dir output \
  --metrics text_edit reading_order table_TEDS table_TEDS_S table_f1 \
            code_TED formula_cdm image_iou overall f1_score

# 커스텀 옵션으로 실행
python olmocr/kolmocr_eval/scripts/evaluate.py \
  --pred_dir olmocr/kolmocr_eval/data/test_data/test_pred \
  --gt_dir olmocr/kolmocr_eval/data/test_data/sample_dataset \
  --output_dir output \
  --metrics text_edit table_f1 \
  --threshold_headings 2 \
  --threshold_table 0.7 \
```

### Arguments

- `--config`: YAML 설정 파일 경로 (기본값: `configs/kolmocr_eval.yaml`)
- `--pred_dir`: 예측 마크다운 디렉토리 경로
- `--gt_dir`: Ground Truth 마크다운 디렉토리 경로
- `--output_dir`: 결과 CSV 저장 경로
- `--metrics`: 실행할 메트릭 (여러 개 입력 가능)
- `--threshold_headings`: 헤딩 매칭 edit distance 문턱치 (기본값: 1)
- `--threshold_table`: 테이블 매칭 유사도 문턱치 0~1 (기본값: 0.6)
- `--version`: 모델 버전 (선택, 1.9 또는 1.10)
- `--text_include_f1`: text_edit에서 F1 점수 포함 여부 (기본값: true)
- `--no_text_f1`: F1 점수 제외

## Implemented Metrics

모든 값은 소수점 4째 자리에서 반올림됩니다.

| 메트릭          | 설명                                                                                             | 출력 파일         |
| --------------- | ------------------------------------------------------------------------------------------------ | ----------------- |
| `text_edit`     | 본문 기준 Normalized Edit Distance 및 유사도, 헤딩/리스트 F1 점수                                | text_edit.csv     |
| `reading_order` | 테이블/이미지를 제외한 라인 순서 기반 Edit Distance                                              | reading_order.csv |
| `table_TEDS`    | 테이블 구조 및 구조+내용 유사도 (Tree Edit Distance)                                             | table_TEDS.csv    |
| `table_TEDS_S`  | `table_TEDS`와 동일 (출력 파일 공유)                                                             | table_TEDS.csv    |
| `table_f1`      | 테이블 블록 매칭 기반 precision/recall/F1 (구조/내용 모두 제공)                                  | table_f1.csv      |
| `image_iou`     | 이미지 bbox 순서 매칭 기반 평균 IoU                                                              | image_iou.csv     |
| `code_TED`      | 코드 블록 추출 후 언어별 트리 변환 및 Tree Edit Distance 유사도<br/>(지원: python, c, cpp, java) | code_TED.csv      |
| `formula_cdm`   | LaTeX 수식 문자 매칭 비율 (CDM 근사)                                                             | formula_cdm.csv   |
| `overall`       | 주요 지표 평균: text_edit, reading_order, table_TEDS, table_TEDS_S, formula_cdm                  | overall.csv       |
| `f1_score`      | 헤딩/리스트 구조 F1 점수만 별도 저장                                                             | f1_score.csv      |

## Output Files

평가 실행 후 생성되는 파일 구조:

```
output_dir/
  ├── YYYYMMDD_HHMMSS/        # 타임스탬프 디렉토리
  │   ├── average.csv         # 전체 평가 지표 평균
  │   ├── element_average.csv # 요소별(텍스트/테이블/이미지/코드) 평균 점수
  │   ├── nipa_table.csv      # NIPA 제출용 요약 테이블
  │   ├── text_edit.csv       # 세부 메트릭 파일들...
  │   ├── table_TEDS.csv
  │   └── ...
  └── evaluate.log            # 평가 실행 로그
```

## Directory Structure

GT와 Pred 디렉토리는 동일한 구조를 유지해야 하며, 같은 경로에 있는 `.md` 파일끼리 비교됩니다.

```
gt_dir/
  ├── category1/
  │   ├── doc1.md
  │   └── doc2.md
  └── category2/
      └── doc3.md

pred_dir/
  ├── category1/
  │   ├── doc1.md
  │   └── doc2.md
  └── category2/
      └── doc3.md
```
