# 공통: GPU/BLAS 스레드 제한
export CUDA_VISIBLE_DEVICES=0,1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1;

# 1) ours (subset20)
python olmocr/inference_kolmocr_transformer.py --checkpoint /my_home/olmocr-qwen2.5-vl-7b-1epoch-mix1128/qwen2.5-vl-olmocrv4_1epoch_ko_mix1128/checkpoint-1500 --tokenizer /my_home/olmocr-qwen2.5-vl-7b-1epoch-mix1128/qwen2.5-vl-olmocrv4_1epoch_ko_mix1128/checkpoint-1500 --input-dir subset20 --output-dir output/exp1_subset_ours;
python olmocr/kolmocr_eval/scripts/evaluate.py --pred_dir output/exp1_subset_ours --gt_dir subset20 --output_dir output/eval_exp1_subset_ours --config configs/kolmocr_eval.yaml;

# 2) vLLM qwen2.5-vl-7b (subset20) - 자동 서버 기동
python olmocr/inference_kolmocr_vllm.py --launch-vllm --tensor-parallel-size 1 --input-dir subset20 --output-dir output/exp2_subset_vllm7b --model Qwen/Qwen2.5-VL-7B-Instruct --api-base http://localhost:8080/v1 --api-key EMPTY;
python olmocr/kolmocr_eval/scripts/evaluate.py --pred_dir output/exp2_subset_vllm7b --gt_dir subset20 --output_dir output/eval_exp2_subset_vllm7b --config configs/kolmocr_eval.yaml;

# 3) vLLM qwen2.5-vl-32b (subset20) - 이전 vLLM 종료 후 실행
python olmocr/inference_kolmocr_vllm.py --launch-vllm --tensor-parallel-size 1 --input-dir subset20 --output-dir output/exp3_subset_vllm32b --model Qwen/Qwen2.5-VL-32B-Instruct --api-base http://localhost:8080/v1 --api-key EMPTY;
python olmocr/kolmocr_eval/scripts/evaluate.py --pred_dir output/exp3_subset_vllm32b --gt_dir subset20 --output_dir output/eval_exp3_subset_vllm32b --config configs/kolmocr_eval.yaml;

# 4) ours (전체)
python olmocr/inference_kolmocr_transformer.py --checkpoint /my_home/olmocr-qwen2.5-vl-7b-1epoch-mix1128/qwen2.5-vl-olmocrv4_1epoch_ko_mix1128/checkpoint-1500 --tokenizer /my_home/olmocr-qwen2.5-vl-7b-1epoch-mix1128/qwen2.5-vl-olmocrv4_1epoch_ko_mix1128/checkpoint-1500 --input-dir kolmocr_bench --output-dir output/exp4_full_ours;
python olmocr/kolmocr_eval/scripts/evaluate.py --pred_dir output/exp4_full_ours --gt_dir kolmocr_bench --output_dir output/eval_exp4_full_ours --config configs/kolmocr_eval.yaml;

# 5) vLLM qwen2.5-vl-7b (전체)
python olmocr/inference_kolmocr_vllm.py --launch-vllm --tensor-parallel-size 1 --input-dir kolmocr_bench --output-dir output/exp5_full_vllm7b --model Qwen/Qwen2.5-VL-7B-Instruct --api-base http://localhost:8080/v1 --api-key EMPTY;
python olmocr/kolmocr_eval/scripts/evaluate.py --pred_dir output/exp5_full_vllm7b --gt_dir kolmocr_bench --output_dir output/eval_exp5_full_vllm7b --config configs/kolmocr_eval.yaml;

# 6) vLLM qwen2.5-vl-32b (전체)
python olmocr/inference_kolmocr_vllm.py --launch-vllm --tensor-parallel-size 1 --input-dir kolmocr_bench --output-dir output/exp6_full_vllm32b --model Qwen/Qwen2.5-VL-32B-Instruct --api-base http://localhost:8080/v1 --api-key EMPTY;
python olmocr/kolmocr_eval/scripts/evaluate.py --pred_dir output/exp6_full_vllm32b --gt_dir kolmocr_bench --output_dir output/eval_exp6_full_vllm32b --config configs/kolmocr_eval.yaml;
