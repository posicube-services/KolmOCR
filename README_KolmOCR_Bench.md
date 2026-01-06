# KolmOCR Benchmark

표/이미지/코드/그래픽 등 다양한 한국어 문서를 포함한 Markdown 생성 과업 평가용 데이터 및 평가 스크립트

## 데이터셋

- 데이터셋 위치: `kolmocr_bench` 폴더
- 평가 스크립트: `olmocr/kolmocr_eval/scripts/evaluate.py`

| Split                          | Docs | 특징                                              |
| ------------------------------ | ---: | ------------------------------------------------- |
| fail document in qwen2.5 7b    |  100 | Qwen2.5-7B-Instruct의 MD생성 성능이 미흡한 문서셋 |
| success document in qwen2.5 7b |  100 | Qwen2.5-7B-Instruct의 MD생성 성능이 좋은 문서셋   |
| table                          |   10 | 셀 병합/멀티헤더 포함                             |
| graphic                        |   10 | 이미지 캡션·도표                                  |
| code_blocks                    |   10 | 코드/리스트 혼재                                  |
| multicolumn                    |   10 | 다단문서                                          |

## LeaderBoard using KolmOCR Benchmark

- KolmOCR Benchmark 에서의 모델에 대한 순위 결정에는 모든 split에 대한 `text_edit`(Text), `table_f1`(Table) `image_iou`(Image IoU), `f1_score` (Heading, List) score의 평균 값이 사용됨.

## Metrics

| 메트릭      | 설명                                                                                             | 출력 파일     |
| ----------- | ------------------------------------------------------------------------------------------------ | ------------- |
| `text_edit` | 본문 기준 Normalized Edit Distance 및 유사도, 헤딩/리스트 F1 점수                                | text_edit.csv |
| `table_f1`  | 테이블 블록 매칭 기반 precision/recall/F1 (구조/내용 모두 제공)                                  | table_f1.csv  |
| `image_iou` | 이미지 bbox 순서 매칭 기반 평균 IoU                                                              | image_iou.csv |
| `code_TED`  | 코드 블록 추출 후 언어별 트리 변환 및 Tree Edit Distance 유사도<br/>(지원: python, c, cpp, java) | code_TED.csv  |
| `overall`   | 주요 지표 평균: text_edit, reading_order, table_TEDS, table_TEDS_S, formula_cdm                  | overall.csv   |
| `f1_score`  | 헤딩/리스트 구조 F1 점수만 별도 저장                                                             | f1_score.csv  |

## Inference 및 평가 프로세스 (Inference & Evaluation Process)

KolmOCR의 전체 프로세스는 크게 **1) Inference (추론)** 단계와 **2) Evaluation (평가)** 단계로 나뉩니다.

```
입력 이미지/PDF → [Inference] → Markdown 출력 → [Evaluation] → 정량적 평가 결과
```

---

## 1. Inference 프로세스 (추론 단계)

### 1.1 모델 및 기본 설정

**KolmOCR v251228 모델:**

- **Base Model**: Qwen2.5-VL-7B (Vision-Language Model)
- **최종 Model**: 한국어/영어 문서에 최적화된 KolmOCR 체크포인트
- **Backend**: vLLM (고속 추론 엔진)
- **Attention**: Flash Attention 2 지원

**KolmOCR 추론 엔진:**

- vLLM 서버를 통한 OpenAI 호환 API 사용
- Tensor Parallelism 지원 (다중 GPU 병렬 처리)
- 자동 재시도 및 회전 보정 기능


### 1.2 입력 이미지 전처리

1. **이미지 로딩**
   - PNG, JPG, JPEG 형식 지원
   - PIL로 로드 후 RGB 변환

2. **이미지 리사이징**
   - 긴 변을 기준으로 타겟 크기로 조정 (예: 1288px)
   - LANCZOS 리샘플링 사용
   - 업스케일링은 하지 않음 (다운스케일만)

3. **회전 처리**
   - 0°, 90°, 180°, 270° 회전 지원
   - 재시도 시 자동 회전 보정

4. **Base64 인코딩**
   - PNG 형식으로 base64 인코딩
   - Data URL 형식: `data:image/png;base64,{encoded_data}`

### 1.3 Inference 프롬프트

**평가 대상 모델 공통으로 사용되는 프롬프트 함수:** 
`build_no_anchoring_v4_yaml_prompt_with_bbox_wo_frontmatter_for_qwen()`
**파일 위치:** `olmocr/prompts/prompts.py`

**프롬프트 내용:**

```
Attached is one page of a document that you must process.
Return a faithful textual representation of the document in reading order.
Convert equations to LaTeX and tables to HTML.
For each block of text, image, table or formula, output exactly one line in the format:
 <|box_start|> x0,y0,x1,y1 <|box_end|>
CONTENT
 [BBOX_BLK_END]
If there are any figures or charts, include the Markdown image tag inside the block content.
![Alt text describing the contents of the figure](NAME_##.png)
Return your output as markdown.
```

**프롬프트 특징:**

- Bounding box 좌표 출력 요구: `<|box_start|> x0,y0,x1,y1 <|box_end|>`
- 각 블록마다 `[BBOX_BLK_END]` 마커로 구분
- 수식은 LaTeX 형식으로 변환: `\[ ... \]`, `\( ... \)`
- 테이블은 HTML 형식으로 출력: `<table>...</table>`
- 이미지는 Markdown 형식: `![alt](image.png)`
- Reading order (읽기 순서) 보존 강조

### 1.4 출력 형식

**Markdown 구조 예시:**

```markdown
<!-- [x0,y0,x1,y1] -->
## 제목

<!-- [x0,y0,x1,y1] -->
본문 텍스트 **굵게** 및 *이탤릭*.

<!-- [x0,y0,x1,y1] -->
- 리스트 항목 1
- 리스트 항목 2

<!-- [x0,y0,x1,y1] -->
<table>
  <tr><th>헤더</th></tr>
  <tr><td>데이터</td></tr>
</table>

<!-- [x0,y0,x1,y1] -->
\[ E = mc^2 \]

<!-- [x0,y0,x1,y1] -->
![그림 캡션](matched_01.png)
```

---

## 2. Evaluation 프로세스 (평가 단계)

### 2.1 평가 개요

KolmOCR Benchmark의 평가는 **규칙 기반(rule-based)** 방식으로 진행됩니다. Inference 단계에서 생성된 예측 마크다운 파일과 Ground Truth(GT) 마크다운 파일을 직접 비교하여 정량적 지표를 산출합니다.

### 2.2 평가 실행 방법

```bash
python olmocr/kolmocr_eval/scripts/evaluate.py \
  --config configs/eval/eval_default.yaml \
  --pred_dir output/preds \
  --gt_dir kolmocr_bench/table \
  --output_dir output/eval
```

**평가 프로세스 단계:**

1. **전처리 (Preprocessing)**
   - 예측 파일에서 `<!-- bbox_blk_end -->` 주석 제거
   - 바운딩 박스 형식 통일: `<|box_start|> x0,y0,x1,y1 <|box_end|>` → `<!-- [x0,y0,x1,y1] -->`

2. **메트릭별 평가 실행**
   - 각 메트릭이 독립적으로 실행되어 CSV 파일로 저장
   - 지원 메트릭: `text_edit`, `table_f1`, `image_iou`, `code_TED`

3. **결과 집계**
   - `average.csv`: 모든 메트릭의 평균
   - `element_average.csv`: 요소별 평균 (Text, Heading, List, Table, Image, Code)
   - `nipa_table.csv` / `nipa_table.md`: NIPA 제출 형식의 핵심 지표 요약

**평가 설정 (Config):**

주요 설정 파라미터 (`configs/eval/eval_default.yaml`):

```yaml
gt_dir: kolmocr_bench           # Ground Truth 경로
pred_dir: output/predictions    # 예측 결과 경로
output_dir: output/eval_results # 평가 결과 저장 경로

metrics:                        # 실행할 메트릭 목록
  - text_edit
  - table_f1
  - image_iou
  - code_TED

threshold_headings: 1           # 헤딩 매칭용 edit distance 임계값
threshold_table: 0.6            # 테이블 매칭용 유사도 임계값 (0~1)
version: "1.10"                 # 모델 버전 (전처리 방식 결정)
text_include_f1: true           # text_edit에 헤딩 F1 포함 여부
```

### 2.3 메트릭 상세 설명

리더보드에 사용되는 4개 핵심 메트릭에 대한 설명입니다. 이 4개 메트릭이 6개 리더보드 항목을 산출합니다:

#### 1. `text_edit` - 텍스트 정확도 평가

**목적:** 순수 텍스트 내용의 정확도, 헤딩 구조, 리스트 항목을 평가

**리더보드 산출 항목:**

- **Text**: `text_edit_sim` (텍스트 유사도)
- **Heading**: `heading_structure_f1` (헤딩 구조 F1 점수)
- **List**: `list_f1` (리스트 항목 F1 점수)

**계산 방법:**

**1.1 텍스트 전처리**

다음 요소들을 제거하여 순수 텍스트만 추출:

- 시스템 헤더 (# Document, ## Page X)
- YAML front matter
- 이미지, 테이블, 코드 블록, 수식
- HTML 주석 (bbox 포함)
- 마크다운 문법 (헤딩, 리스트, 링크)
- LaTeX 문법을 유니코드로 변환 (α, β, γ 등)
- 공백 정규화 및 하이픈 처리
- 전각 문자 → 반각 문자 변환

**1.2 Adjacency Normalized Edit Distance (NED)**

- 텍스트를 문단 단위로 분할
- 최대 3개 문단까지 병합하면서 greedy 정렬
- Levenshtein 편집 거리 계산: `NED = edit_distance / max(len(pred), len(gt))`
- 유사도 변환: `similarity = 1 - NED`

**1.3 헤딩(Heading) 평가**

- 정규식으로 헤딩 추출: `^(#{1,6})\s+(.+)`
- 편집 거리 기반 매칭 (임계값: 1)
- Precision/Recall/F1 계산:
  - `heading_structure`: 헤딩 텍스트 매칭
  - `level`: 헤딩 레벨 정확도 (##, ###, 등)

**1.4 리스트(List) 평가**

- 정규식으로 리스트 항목 추출: `^[ \t]*(?:[-*+]|\d+\.)\s+(.*)`
- 헤딩과 동일한 매칭 로직 적용

#### 2. `table_f1` - 테이블 구조 및 내용 평가

**목적:** 테이블 추출의 정확도를 Precision/Recall/F1로 평가

**리더보드 산출 항목:**

- **Table**: `table_struct_f1` (테이블 구조 F1 점수)

**계산 방법:**

**2.1 테이블 파싱**

- 2가지 형식 지원:
  - **HTML**: `<table>...</table>`
  - **Markdown**: 파이프 테이블 (`| col1 | col2 |`)
- HTML 정규화: th→td 변환, thead 제거, 수식 alttext 추출, style 속성 제거
- 행/셀 구조로 파싱 (rowspan/colspan 보존)

**2.2 테이블 직렬화**

- **Structure-only (구조만)**: `r{rowspan}c{colspan}|r{rowspan}c{colspan}|...`
- **Semantic (내용 포함)**: `r{rowspan}c{colspan}:텍스트|...`

**2.3 Adjacency Alignment**

- Needleman-Wunsch 전역 정렬 알고리즘 사용
- 순서를 보존하면서 총 유사도 최대화
- Gap penalty = 0 (매칭 안 된 항목은 유사도 0)

**2.4 매칭 및 점수 계산**

- 각 매칭 쌍의 편집 거리 유사도 계산
- 임계값 (기본 0.6): 이 값 이상이면 True Positive로 카운트
- `Precision = 매칭된 개수 / 예측 개수`
- `Recall = 매칭된 개수 / GT 개수`
- `F1 = 2 * P * R / (P + R)`
- 구조만 비교 (structure)와 내용까지 포함 (semantic) 두 가지 점수 제공

#### 3. `image_iou` - 이미지 바운딩 박스 평가

**목적:** 이미지 위치 검출 정확도를 IoU(Intersection over Union)로 평가

**리더보드 산출 항목:**

- **Image IoU**: `image_avg_iou` (평균 IoU)

**계산 방법:**

**3.1 바운딩 박스 추출**

- HTML 주석 형식: `<!-- [x0,y0,x1,y1] -->`
- 이미지 마크다운 바로 앞에 위치: `![alt](url)`
- 좌표 형식: `[x_min, y_min, x_max, y_max]`

**3.2 IoU 계산**

```python
intersection = 겹치는 영역의 넓이
union = 박스1 넓이 + 박스2 넓이 - intersection
IoU = intersection / union
```

**3.3 Adjacency Matching**

- 전역 정렬로 pred/gt 바운딩 박스 매칭
- IoU 점수의 합을 최대화
- 매칭 안 된 GT 박스는 IoU = 0

**3.4 점수 계산**

- 모든 GT 이미지에 대한 평균 IoU
- GT 이미지가 없는 경우 1.0 (완벽한 점수)

#### 4. `code_TED` - 코드 블록 구조 평가

**목적:** 코드 블록 추출 정확도를 Tree Edit Distance로 평가

**리더보드 산출 항목:**

- **Code-Block**: `code_ted` (코드 블록 유사도)

**계산 방법:**

**4.1 코드 블록 추출**

- **Fenced 코드 블록**: ` ```lang\n...\n``` `
- **Indented 코드 블록**: 4칸 이상 들여쓰기
- **지원 언어**: python, py, c, c++, cpp, java

**4.2 코드 정규화**

- 탭 → 4칸 공백 변환
- trailing 공백 제거
- 앞뒤 빈 줄 제거
- 들여쓰기 단위 추정 (최소 양수 들여쓰기)
- 깊이 정규화: depth = leading_spaces // unit

**4.3 코드 트리 구축**

- **Python**: 들여쓰기 기반 계층 구조 (스택 기반 트리)
- **C/C++/Java**: 중괄호 기반 계층 구조 (`{`로 자식 생성, `}`로 스택 pop)
- **기타 언어**: 평면 라인 노드

**4.4 Tree Edit Distance (TED)**

- 순서 있는 트리 편집 거리 (DP 알고리즘)
- 삽입/삭제 비용 = 서브트리 크기
- 치환 비용 = 라벨 일치 시 0, 불일치 시 1
- 유사도 = `max(0, 1 - TED / max(|pred|, |gt|))`

**4.5 Adjacency Matching**

- pred/gt 코드 블록 순서대로 정렬
- 총 유사도 최대화

---

## 3. 전체 프로세스 요약

### Inference에서 프롬프트 사용

**Inference 단계**에서는 프롬프트를 사용합니다:

- 모델에게 문서 처리 방법을 지시 (`build_no_anchoring_v4_yaml_prompt_with_bbox_wo_frontmatter`)
- Bbox 좌표, LaTeX 수식, HTML 테이블 형식 출력 요구
- Reading order 보존 강조

**Evaluation 단계**에서는 프롬프트를 사용하지 않습니다:

- 규칙 기반 알고리즘으로 GT와 예측 마크다운을 직접 비교
- LLM 없이 **결정론적(deterministic)** 메트릭만 사용
- 재현 가능한 정량 평가 보장

### 전체 파이프라인

```
[입력] 이미지/PDF
    ↓
[Inference] 모델 + 프롬프트
    → vLLM 서버를 통한 추론
    → 온도 기반 재시도 (0.1 → 1.0)
    ↓
[출력] Markdown + Bbox 주석
    ↓
[Evaluation] 규칙 기반 메트릭
    → text_edit: 텍스트 유사도, 헤딩/리스트 F1
    → table_f1: 테이블 구조/의미 P/R/F1
    → image_iou: 이미지 bbox IoU
    → code_TED: 코드 블록 Tree Edit Distance
    ↓
[결과] CSV 파일 + 요약 테이블
```
