
<div align="center">
  <img width="300" alt="KolmOCR logo" src="./docs/logo.jpg" />
</div>
<hr/>

# KolmOCR 

KolmOCR은 기존의 [olmOCR](https://github.com/allenai/olmocr)를 한국어/영어 문서에 최적화된 OCR 파이프라인으로, 이미지·PDF를 구조화된 Markdown으로 변환합니다. 한국어 문서에 대한 Markdown 생성 과업에 대한 벤치마크도 함께 제공합니다. 

## Features

- 한국어/영어 혼합 문서, 표, 코드블록, 수식까지 Markdown으로 구조화
- Markdown 내 이미지·리스트·표·코드블록 보존
- 단일/멀티프로세스 추론 지원 (`--num-workers`)
- 평가 스크립트와 파이프라인 실행 스크립트 제공

## KolmOCR Benchmark

- 표/이미지/코드/그래픽 등 다양한 한국어 문서를 포함한 Markdown 생성 과업 평가용 데이터 및 평가 스크립트
- 데이터셋 위치: `kolmocr_bench` 폴더
- 평가 스크립트: `olmocr/kolmocr_eval/scripts/evaluate.py`

| Split       | Docs | 특징                  |
| ----------- | ---: | --------------------- |
| fail document in qwen2.5 7b | 100 | Qwen2.5-7B-Instruct의 MD생성 성능이 미흡한 문서셋 |
| success document in qwen2.5 7b | 100 | Qwen2.5-7B-Instruct의 MD생성 성능이 좋은 문서셋  |
| table       | 10 | 셀 병합/멀티헤더 포함 |
| graphic     | 10 | 이미지 캡션·도표      |
| code_blocks |  10 | 코드/리스트 혼재      |
| multicolumn |  10 | 다단문서      |

- 상기 모든 split에 대한 `text_edit`(Text), `table_f1`(Table) `image_iou`(Image IoU), `f1_score` (Heading, List) score가 사용됨. Image IoU 현재 평가 코드상 오류로 N/A로 표시됨.

## LeaderBoard using KolmOCR Benchmark
| Element        | KolmOCR 7B v251129 (Ours) | Qwen2.5-VL-7B-Instruct | Qwen2.5-VL-32B-Instruct |
| -------------- | ----------------- | ---------------------- | ----------------------- |
| **Text**       | 0.5695            | 0.5993                 | 0.5938                  |
| **Heading**    | 0.3099            | 0.3775                 | 0.3197                  |
| **List**       | 0.1931            | 0.3256                 | 0.2448                  |
| **Table**      | 0.5857            | 0.1333                 | 0.364                   |
| **Image IoU**  | N/A               | N/A                    | N/A                     |
| **Code-Block** | 0.0143            | 0.0321                 | 0.037                   |

## Metrics
| 메트릭          | 설명                                                                                             | 출력 파일         |
| --------------- | ------------------------------------------------------------------------------------------------ | ----------------- |
| `text_edit`     | 본문 기준 Normalized Edit Distance 및 유사도, 헤딩/리스트 F1 점수                                | text_edit.csv     |
| `table_f1`      | 테이블 블록 매칭 기반 precision/recall/F1 (구조/내용 모두 제공)                                  | table_f1.csv      |
| `image_iou`     | 이미지 bbox 순서 매칭 기반 평균 IoU                                                              | image_iou.csv     |
| `code_TED`      | 코드 블록 추출 후 언어별 트리 변환 및 Tree Edit Distance 유사도<br/>(지원: python, c, cpp, java) | code_TED.csv      |
| `overall`       | 주요 지표 평균: text_edit, reading_order, table_TEDS, table_TEDS_S, formula_cdm                  | overall.csv       |
| `f1_score`      | 헤딩/리스트 구조 F1 점수만 별도 저장                                                             | f1_score.csv      |

## Installation (uv 권장)

```bash
# uv 설치 및 venv 생성
python -m pip install --upgrade pip
python -m pip install uv
uv venv --python 3.11
source .venv/bin/activate

# 프로젝트 의존성 설치
uv sync --active --extra dev --extra train
```

- GPU 환경: CUDA 12.8 기준 PyTorch/FlashAttention 호환을 사용하세요.
- CPU 전용이라면 PyTorch CPU 빌드를 설치하거나 `--extra train` 없이 최소 구성으로 설치 가능합니다.

## Usage

### Train

```bash
python olmocr/train/train.py --config configs/train.yaml
```

### Inference

```bash
python olmocr/inference_kolmocr_transformer.py --config configs/infer_config.yaml \
  --checkpoint <모델 경로> --tokenizer <토크나이저 경로> \
  --input-dir <이미지 혹은 md 경로> --output-dir <결과 경로> \
  --num-workers 2
```

### Evaluate

```bash
python olmocr/kolmocr_eval/scripts/evaluate.py --config configs/kolmocr_eval.yaml \
  --pred_dir <inference 결과 경로> --gt_dir <GT md/html 경로>
```

## Script Quickstart

- `olmocr/inference_kolmocr_transformer.py`

  - **용도:** 로컬 HF 모델로 이미지/MD를 Markdown으로 추론. 멀티프로세싱 지원.
  - **예시:** `python olmocr/inference_kolmocr_transformer.py --checkpoint /path/to/ckpt --tokenizer /path/to/tokenizer --input-dir kolmocr_bench --output-dir output/preds --num-workers 4`

- `olmocr/inference_kolmocr_vllm.py`

  - **용도:** OpenAI 호환 vLLM 서버에 배치 추론. `--launch-vllm`로 로컬 서버 기동 가능.
  - **예시(기동 포함):** `python olmocr/inference_kolmocr_vllm.py --launch-vllm --model Qwen/Qwen2.5-VL-7B-Instruct --input-dir imgs --output-dir output/preds_vllm --api-base http://localhost:8080/v1`

- `olmocr/kolmocr_eval/scripts/evaluate.py`

  - **용도:** GT와 예측 MD를 비교해 text/table/reading_order 등 메트릭 계산.
  - **예시:** `python olmocr/kolmocr_eval/scripts/evaluate.py --pred_dir output/preds --gt_dir kolmocr_bench/table --output_dir output/eval_run --config configs/kolmocr_eval.yaml`

- `scripts/run_infrapartner_benchmark.sh`
  - **용도:** 인프라 파트너 벤치마크 자동 실행 스크립트(환경에 맞게 경로/모델 수정).
  - **예시:** `bash scripts/run_infrapartner_benchmark.sh` (스크립트 내 변수로 모델/입출력 경로 설정)

### End-to-end Pipeline

```bash
python olmocr/bench/runners/run_olmocr_pipeline.py --config configs/pipeline_kolmocr_bench.yaml
```

## Output Format

- Markdown에 표/리스트/이미지/코드블록을 보존
- 예시:

````markdown
## 섹션 제목

- 항목 1
- 항목 2

```python
print("hello kolmocr")
```
````

## License
- 'kolmocr_bench' 폴더의 데이터는 CC-BY-NC 4.0 기준으로 배포합니다. 
- 나머지 소스코드의 경우 Apache 2.0 으로 배포합니다.
  
## 기존 olmocr의 README.md
---

<div align="center">
  <img width="350" alt="olmocr-2-full@2x" src="https://github.com/user-attachments/assets/24f1b596-4059-46f1-8130-5d72dcc0b02e" />
<hr/>
</div>
<p align="center">
  <a href="https://github.com/allenai/OLMo/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://github.com/allenai/olmocr/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/olmocr.svg">
  </a>
  <a href="https://arxiv.org/abs/2502.18443">
    <img alt="Tech Report v1" src="https://img.shields.io/badge/Paper_v1-olmOCR-blue">
  </a>
  <a href="https://arxiv.org/abs/2510.19817">
    <img alt="Tech Report v2" src="https://img.shields.io/badge/Paper_v2-olmOCR-blue">
  </a>
  <a href="https://olmocr.allenai.org">
    <img alt="Demo" src="https://img.shields.io/badge/Ai2-Demo-F0529C">
  </a>
  <a href="https://discord.gg/sZq3jTNVNG">
    <img alt="Discord" src="https://img.shields.io/badge/Discord%20-%20blue?style=flat&logo=discord&label=Ai2&color=%235B65E9">
  </a>
</p>

A toolkit for converting PDFs and other image-based document formats into clean, readable, plain text format.

Try the online demo: [https://olmocr.allenai.org/](https://olmocr.allenai.org/)

Features:

- Convert PDF, PNG, and JPEG based documents into clean Markdown
- Support for equations, tables, handwriting, and complex formatting
- Automatically removes headers and footers
- Convert into text with a natural reading order, even in the presence of
  figures, multi-column layouts, and insets
- Efficient, less than $200 USD per million pages converted
- (Based on a 7B parameter VLM, so it requires a GPU)

### News

- October 21, 2025 - v0.4.0 - [New model release](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8), boosts olmOCR-bench score by ~4 points using synthetic data and introduces RL training.
- August 13, 2025 - v0.3.0 - [New model release](https://huggingface.co/allenai/olmOCR-7B-0825-FP8), fixes auto-rotation detection, and hallucinations on blank documents.
- July 24, 2025 - v0.2.1 - [New model release](https://huggingface.co/allenai/olmOCR-7B-0725-FP8), scores 3 points higher on [olmOCR-Bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench), also runs significantly faster because it's default FP8, and needs much fewer retries per document.
- July 23, 2025 - v0.2.0 - New cleaned up [trainer code](https://github.com/allenai/olmocr/tree/main/olmocr/train), makes it much simpler to train olmOCR models yourself.
- June 17, 2025 - v0.1.75 - Switch from sglang to vllm based inference pipeline, updated docker image to CUDA 12.8.
- May 23, 2025 - v0.1.70 - Official docker support and images are now available! [See Docker usage](#using-docker)
- May 19, 2025 - v0.1.68 - [olmOCR-Bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench) launch, scoring 77.4. Launch includes 2 point performance boost in olmOCR pipeline due to bug fixes with prompts.
- Mar 17, 2025 - v0.1.60 - Performance improvements due to better temperature selection in sampling.
- Feb 25, 2025 - v0.1.58 - Initial public launch and demo.

### Benchmark

[**olmOCR-Bench**](https://github.com/allenai/olmocr/tree/main/olmocr/bench):
We also ship a comprehensive benchmark suite covering over 7,000 test cases across 1,400 documents to help measure performance of OCR systems.

<table>
    <thead>
        <tr>
            <th></th>
            <th>ArXiv</th>
            <th>Old<br>scans<br>math</th>
            <th>Tables</th>
            <th>Old<br>scans</th>
            <th>Headers<br>&<br>footers</th>
            <th>Multi<br>column</th>
            <th>Long<br>tiny<br>text</th>
            <th>Base</th>
            <th>Overall</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Mistral OCR API</td>
            <td>77.2</td>
            <td>67.5</td>
            <td>60.6</td>
            <td>29.3</td>
            <td>93.6</td>
            <td>71.3</td>
            <td>77.1</td>
            <td>99.4</td>
            <td>72.0±1.1</td>
        </tr>
        <tr>
            <td>Marker 1.10.1</td>
            <td>83.8</td>
            <td>66.8</td>
            <td>72.9</td>
            <td>33.5</td>
            <td>86.6</td>
            <td>80.0</td>
            <td>85.7</td>
            <td>99.3</td>
            <td>76.1±1.1</td>
        </tr>
        <tr>
            <td>MinerU 2.5.4*</td>
            <td>76.6</td>
            <td>54.6</td>
            <td>84.9</td>
            <td>33.7</td>
            <td>96.6</td>
            <td>78.2</td>
            <td>83.5</td>
            <td>93.7</td>
            <td>75.2±1.1</td>
        </tr>
        <tr>
            <td>DeepSeek-OCR</td>
            <td>77.2</td>
            <td>73.6</td>
            <td>80.2</td>
            <td>33.3</td>
            <td>96.1</td>
            <td>66.4</td>
            <td>79.4</td>
            <td>99.8</td>
            <td>75.7±1.0</td>
        </tr>
        <tr>
            <td>Nanonets-OCR2-3B</td>
            <td>75.4</td>
            <td>46.1</td>
            <td>86.8</td>
            <td>40.9</td>
            <td>32.1</td>
            <td>81.9</td>
            <td>93.0</td>
            <td>99.6</td>
            <td>69.5±1.1</td>
        </tr>
        <tr>
            <td>PaddleOCR-VL*</td>
            <td>85.7</td>
            <td>71.0</td>
            <td>84.1</td>
            <td>37.8</td>
            <td>97.0</td>
            <td>79.9</td>
            <td>85.7</td>
            <td>98.5</td>
            <td>80.0±1.0</td>
        </tr>
        <tr>
            <td>Infinity-Parser 7B*</td>
            <td>84.4</td>
            <td>83.8</td>
            <td>85.0</td>
            <td>47.9</td>
            <td>88.7</td>
            <td>84.2</td>
            <td>86.4</td>
            <td>99.8</td>
            <td>82.5±?</td>
        </tr>
        <tr>
            <td>Chandra OCR 0.1.0*</td>
            <td>82.2</td>
            <td>80.3</td>
            <td>88.0</td>
            <td>50.4</td>
            <td>90.8</td>
            <td>81.2</td>
            <td>92.3</td>
            <td>99.9</td>
            <td>83.1±0.9</td>
        </tr>
        <tr>
            <td colspan="10"><hr></td>
        </tr>
        <tr>
            <td><strong>olmOCR v0.4.0</strong></td>
            <td>83.0</td>
            <td>82.3</td>
            <td>84.9</td>
            <td>47.7</td>
            <td>96.1</td>
            <td>83.7</td>
            <td>81.9</td>
            <td>99.7</td>
            <td>82.4±1.1</td>
        </tr>
    </tbody>
</table>

### Installation

Requirements:

- Recent NVIDIA GPU (tested on RTX 4090, L40S, A100, H100) with at least 15 GB of GPU RAM
- 30GB of free disk space

You will need to install poppler-utils and additional fonts for rendering PDF images.

Install dependencies (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
```

We recommend relying on the [`uv`](https://uvpkg.dev/) package manager instead of `conda`.
`uv` reads `pyproject.toml` plus the committed `uv.lock` file, so installing dependencies is a single reproducible step and the same workflow works for development, CI, and deployment.

```bash
# install the uv CLI (can be done inside any existing python3 install)
python -m pip install --upgrade pip
python -m pip install uv
# create venv and activate it
uv venv --python 3.11
source ./.venv/bin/activate

# install olmocr dependencies into uv's managed virtual environment
uv sync --active --extra dev --extra train            # creates/refreshes the per-project virtual environment defined in `.venv/`
# add `--extra train` when you need to train and develop your codes.

# to train with flash attention
uv pip install flash-attn --no-build-isolation


# inside the shell you can run commands exactly as before (or prefix with `uv run` outside the shell)
python -m olmocr.pipeline ./localworkspace --markdown --pdfs tests/gnarly_pdfs/*.pdf
```

When you add or update dependencies, run `uv add <package>` (or edit `pyproject.toml` and `uv lock --check`) and commit the changed `uv.lock` so everyone else installs the same versions.

If you still prefer a `conda` environment, the old instructions remain below, but we suggest trying `uv sync` first for a reproducible setup.

Set up a conda environment and install olmocr. The requirements for running olmOCR
are difficult to install in an existing python environment, so please do make a clean python environment to install into.

```bash
conda create -n olmocr python=3.11
conda activate olmocr

# For CPU-only operations, ex running the benchmark
pip install olmocr[bench]

# For actually converting the files with your own GPU
pip install olmocr[gpu]  --extra-index-url https://download.pytorch.org/whl/cu128

# Recommended: Install flash infer for faster inference on GPU
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
```

### Local Usage Example

For quick testing, try the [web demo](https://olmocr.allen.ai/). To run locally, a GPU is required, as inference is powered by [sglang](https://github.com/sgl-project/sglang) under the hood.

Convert a Single PDF:

```bash
# Download a sample PDF
curl -o olmocr-sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf

# Convert it to markdown
python -m olmocr.pipeline ./localworkspace --markdown --pdfs olmocr-sample.pdf
```

Convert an Image file:

```bash
python -m olmocr.pipeline ./localworkspace --markdown --pdfs random_page.png
```

Convert Multiple PDFs:

```bash
python -m olmocr.pipeline ./localworkspace --markdown --pdfs tests/gnarly_pdfs/*.pdf
```

With the addition of the `--markdown` flag, results will be stored as markdown files inside of `./localworkspace/markdown/`.

#### Viewing Results

The `./localworkspace/` workspace folder will then have both [Dolma](https://github.com/allenai/dolma) and markdown files (if using `--markdown`).

```bash
cat localworkspace/markdown/olmocr-sample.md
```

```
olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models
...
```

### Using an Inference Provider or External Server

If you have a vLLM server already running elsewhere (or any inference platform implementing the OpenAI API), you can point olmOCR to use it instead of spawning a local instance:

```bash
# Use external vLLM server instead of local one
python -m olmocr.pipeline ./localworkspace --server http://remote-server:8000/v1 --model allenai/olmOCR-2-7B-1025-FP8 --markdown --pdfs tests/gnarly_pdfs/*.pdf
```

The served model name in VLLM needs to match the value provided in `--model`.

An example vLLM launch command would be:

```bash
vllm serve allenai/olmOCR-2-7B-1025-FP8 --max-model-len 16384
```

#### Verified External Providers

We have tested `olmOCR-2-7B-1025-FP8` on these external model providers and confirmed that they work

|                                                                             | $/1M Input tokens | $/1M Output tokens | Example Command                                                                                                                                                                |
| --------------------------------------------------------------------------- | ----------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Cirrascale](https://ai2endpoints.cirrascale.ai/models/overview)            | $0.07             | $0.15              | `python -m olmocr.pipeline ./localworkspace1 --server https://ai2endpoints.cirrascale.ai/api --api_key sk-XXXXXXX --model olmOCR-2-7B-1025 --pdfs tests/gnarly_pdfs/*.pdf`     |
| [DeepInfra](https://deepinfra.com/)                                         | $0.09             | $0.19              | `python -m olmocr.pipeline ./localworkspace1 --server https://api.deepinfra.com/v1/openai --api_key DfXXXXXXX --model allenai/olmOCR-2-7B-1025 --pdfs tests/gnarly_pdfs/*.pdf` |
| [Parasail](https://www.saas.parasail.io/serverless?name=olmocr-7b-1025-fp8) | $0.10             | $0.20              | `python -m olmocr.pipeline ./localworkspace1 --server https://api.parasail.io/v1 --api_key psk-XXXXX --model allenai/olmOCR-2-7B-1025 --pdfs tests/gnarly_pdfs/*.pdf`          |

Notes on arguments

- `--server`: Defines the OpenAI-compatible endpoint: ex `https://api.deepinfra.com/v1/openai`
- `--api_key`: Your API key, bassed in via Authorization Bearer HTTP header
- `--pages_per_group`: You may want a smaller number of pages per group as many external provides have lower concurrent request limits
- `--model`: The model identifier, ex. `allenai/olmOCR-2-7B-1025`, different providers have different names, and if you run locally, you can use `olmocr`
- Other arguments work the same as with local inference

### Multi-node / Cluster Usage

If you want to convert millions of PDFs, using multiple nodes running in parallel, then olmOCR supports
reading your PDFs from AWS S3, and coordinating work using an AWS S3 output bucket.

For example, you can start this command on your first worker node, and it will set up
a simple work queue in your AWS bucket and start converting PDFs.

```bash
python -m olmocr.pipeline s3://my_s3_bucket/pdfworkspaces/exampleworkspace --pdfs s3://my_s3_bucket/jakep/gnarly_pdfs/*.pdf
```

Now on any subsequent nodes, just run this and they will start grabbing items from the same workspace queue.

```bash
python -m olmocr.pipeline s3://my_s3_bucket/pdfworkspaces/exampleworkspace
```

If you are at Ai2 and want to linearize millions of PDFs efficiently using [beaker](https://www.beaker.org), just add the `--beaker`
flag. This will prepare the workspace on your local machine, and then launch N GPU workers in the cluster to start
converting PDFs.

For example:

```bash
python -m olmocr.pipeline s3://my_s3_bucket/pdfworkspaces/exampleworkspace --pdfs s3://my_s3_bucket/jakep/gnarly_pdfs/*.pdf --beaker --beaker_gpus 4
```

### Using Docker

Pull the Docker image (large, includes the model, ~30GB):

```bash
docker pull alleninstituteforai/olmocr:latest-with-model
```

For advanced users who want to manage their own model downloads, we also provide a base image without the model:

```bash
docker pull alleninstituteforai/olmocr:latest
```

#### Quick Start - Process PDFs

Process a single PDF in your current directory:

```bash
docker run --gpus all \
  -v $(pwd):/workspace \
  alleninstituteforai/olmocr:latest-with-model \
  -c "python -m olmocr.pipeline /workspace/output --markdown --pdfs /workspace/sample.pdf"
```

Process multiple PDFs:

```bash
docker run --gpus all \
  -v /path/to/pdfs:/input \
  -v /path/to/output:/output \
  alleninstituteforai/olmocr:latest-with-model \
  -c "python -m olmocr.pipeline /output --markdown --pdfs /input/*.pdf"
```

#### Interactive Mode

Run the container interactively for exploration and debugging:

```bash
docker run -it --gpus all alleninstituteforai/olmocr:latest-with-model
```

> Visit our Docker repository on [Docker Hub](https://hub.docker.com/r/alleninstituteforai/olmocr) for more information.

### Full documentation for the pipeline

```bash
python -m olmocr.pipeline --help
usage: pipeline.py [-h] [--pdfs [PDFS ...]] [--model MODEL] [--workspace_profile WORKSPACE_PROFILE] [--pdf_profile PDF_PROFILE] [--pages_per_group PAGES_PER_GROUP] [--max_page_retries MAX_PAGE_RETRIES] [--max_page_error_rate MAX_PAGE_ERROR_RATE] [--workers WORKERS]
                   [--apply_filter] [--stats] [--markdown] [--target_longest_image_dim TARGET_LONGEST_IMAGE_DIM] [--target_anchor_text_len TARGET_ANCHOR_TEXT_LEN] [--guided_decoding] [--gpu-memory-utilization GPU_MEMORY_UTILIZATION] [--max_model_len MAX_MODEL_LEN]
                   [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--data-parallel-size DATA_PARALLEL_SIZE] [--port PORT] [--server SERVER] [--beaker] [--beaker_workspace BEAKER_WORKSPACE] [--beaker_cluster BEAKER_CLUSTER] [--beaker_gpus BEAKER_GPUS] [--beaker_priority BEAKER_PRIORITY]
                   workspace

Manager for running millions of PDFs through a batch inference pipeline

positional arguments:
  workspace             The filesystem path where work will be stored, can be a local folder, or an s3 path if coordinating work with many workers, s3://bucket/prefix/

options:
  -h, --help            show this help message and exit
  --pdfs [PDFS ...]     Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths
  --model MODEL         Path where the model is located, allenai/olmOCR-7B-0725-FP8 is the default, can be local, s3, or hugging face.
  --workspace_profile WORKSPACE_PROFILE
                        S3 configuration profile for accessing the workspace
  --pdf_profile PDF_PROFILE
                        S3 configuration profile for accessing the raw pdf documents
  --pages_per_group PAGES_PER_GROUP
                        Aiming for this many pdf pages per work item group
  --max_page_retries MAX_PAGE_RETRIES
                        Max number of times we will retry rendering a page
  --max_page_error_rate MAX_PAGE_ERROR_RATE
                        Rate of allowable failed pages in a document, 1/250 by default
  --workers WORKERS     Number of workers to run at a time
  --apply_filter        Apply basic filtering to English pdfs which are not forms, and not likely seo spam
  --stats               Instead of running any job, reports some statistics about the current workspace
  --markdown            Also write natural text to markdown files preserving the folder structure of the input pdfs
  --target_longest_image_dim TARGET_LONGEST_IMAGE_DIM
                        Dimension on longest side to use for rendering the pdf pages
  --target_anchor_text_len TARGET_ANCHOR_TEXT_LEN
                        Maximum amount of anchor text to use (characters), not used for new models
  --guided_decoding     Enable guided decoding for model YAML type outputs

VLLM arguments:
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        Fraction of VRAM vLLM may pre-allocate for KV-cache (passed through to vllm serve).
  --max_model_len MAX_MODEL_LEN
                        Upper bound (tokens) vLLM will allocate KV-cache for, lower if VLLM won't start
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
                        Tensor parallel size for vLLM
  --data-parallel-size DATA_PARALLEL_SIZE, -dp DATA_PARALLEL_SIZE
                        Data parallel size for vLLM
  --port PORT           Port to use for the VLLM server
  --server SERVER       URL of external vLLM (or other compatible provider)
                        server (e.g., http://hostname:port). If provided,
                        skips spawning local vLLM instance

beaker/cluster execution:
  --beaker              Submit this job to beaker instead of running locally
  --beaker_workspace BEAKER_WORKSPACE
                        Beaker workspace to submit to
  --beaker_cluster BEAKER_CLUSTER
                        Beaker clusters you want to run on
  --beaker_gpus BEAKER_GPUS
                        Number of gpu replicas to run
  --beaker_priority BEAKER_PRIORITY
                        Beaker priority level for the job
```

## Code overview

There are some nice reusable pieces of the code that may be useful for your own projects:

- A prompting strategy to get really good natural text parsing using ChatGPT 4o - [buildsilver.py](https://github.com/allenai/olmocr/blob/main/olmocr/data/buildsilver.py)
- Basic filtering by language and SEO spam removal - [filter.py](https://github.com/allenai/olmocr/blob/main/olmocr/filter/filter.py)
- SFT Finetuning code for Qwen2.5-VL - [train.py](https://github.com/allenai/olmocr/blob/main/olmocr/train/train.py)
- GRPO RL Trainer - [grpo_train.py](https://github.com/allenai/olmocr/blob/main/olmocr/train/grpo_train.py)
- Synthetic data generation - [mine_html_templates.py](https://github.com/allenai/olmocr/blob/main/olmocr/bench/synth/mine_html_templates.py)
- Processing millions of PDFs through a finetuned model using VLLM - [pipeline.py](https://github.com/allenai/olmocr/blob/main/olmocr/pipeline.py)
- Viewing [Dolma docs](https://github.com/allenai/dolma) created from PDFs - [dolmaviewer.py](https://github.com/allenai/olmocr/blob/main/olmocr/viewer/dolmaviewer.py)

## Team

<!-- start team -->

**olmOCR** is developed and maintained by the AllenNLP team, backed by [the Allen Institute for Artificial Intelligence (AI2)](https://allenai.org/).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
To learn more about who specifically contributed to this codebase, see [our contributors](https://github.com/allenai/olmocr/graphs/contributors) page.

<!-- end team -->

## License

<!-- start license -->

**olmOCR** is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
A full copy of the license can be found [on GitHub](https://github.com/allenai/olmocr/blob/main/LICENSE).

<!-- end license -->

## Citing

For olmOCR v1 and OlmOCR-bench:

```bibtex
@misc{olmocrbench,
      title={{olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models}},
      author={Jake Poznanski and Jon Borchardt and Jason Dunkelberger and Regan Huff and Daniel Lin and Aman Rangapur and Christopher Wilhelm and Kyle Lo and Luca Soldaini},
      year={2025},
      eprint={2502.18443},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18443},
}
```

For olmOCR v2 Unit Testing Rewards with RL:

```bibtex
@misc{olmocr2,
      title={olmOCR 2: Unit Test Rewards for Document OCR},
      author={Jake Poznanski and Luca Soldaini and Kyle Lo},
      year={2025},
      eprint={2510.19817},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.19817},
}
```
