#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/srj/VLMsafeguard"
CODE_DIR="${REPO_ROOT}/code"

MODEL="${MODEL:-/s/models/llava-series/llava-1.5-7b-hf}"
MM_JSONL="${MM_JSONL:-${CODE_DIR}/data/data_vlguard/processed/train_mm.jsonl}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-all}"
PROMPT_LENGTH="${PROMPT_LENGTH:-default}"
OUT_DIR="${OUT_DIR:-./trained_prompts_mm}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LR="${LR:-1e-4}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"

if [[ ! -f "${MM_JSONL}" ]]; then
  echo "Missing mm_jsonl: ${MM_JSONL}" >&2
  exit 1
fi

cd "${CODE_DIR}"

cmd=(
  python3 train.py
  --pretrained_model_path "${MODEL}"
  --config sampling
  --system_prompt_type "${SYSTEM_PROMPT_TYPE}"
  --prompt_length "${PROMPT_LENGTH}"
  --enable_vision
  --mm_jsonl "${MM_JSONL}"
  --batch_size "${BATCH_SIZE}"
  --effective_batch_size "${EFFECTIVE_BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --lr "${LR}"
  --output_path "${OUT_DIR}"
)

if [[ -n "${RUN_NAME_SUFFIX}" ]]; then
  cmd+=(--run_name_suffix "${RUN_NAME_SUFFIX}")
fi

"${cmd[@]}"
