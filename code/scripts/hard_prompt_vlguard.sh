set -euo pipefail

cd /home/srj/VLMsafeguard/code

MODEL="/s/models/llava-series/llava-1.5-7b-hf"
MM_JSONL="/home/srj/VLMsafeguard/code/data/data_vlguard/processed/test_mm.jsonl"
OUT="./outputs_vlguard_testmm"

python generate.py \
  --pretrained_model_path "${MODEL}" \
  --enable_vision \
  --mm_jsonl "${MM_JSONL}" \
  --use_default_prompt \
  --n_samples 1 \
  --output_path "${OUT}"
