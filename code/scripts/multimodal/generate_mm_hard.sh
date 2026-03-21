#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/srj/VLMsafeguard"
CODE_DIR="${REPO_ROOT}/code"

MODEL="${MODEL:-/s/models/llava-series/llava-1.5-7b-hf}"
MM_JSONL="${MM_JSONL:-}"
OUT_DIR="${OUT_DIR:-./outputs/mm/hard_default}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
N_SAMPLES="${N_SAMPLES:-1}"

if [[ -z "${MM_JSONL}" ]]; then
  echo "MM_JSONL is required. Example: MM_JSONL=${CODE_DIR}/data/mme_build/llava_mme_mm.jsonl bash $0" >&2
  exit 1
fi

if [[ ! -f "${MM_JSONL}" ]]; then
  echo "Missing mm_jsonl: ${MM_JSONL}" >&2
  exit 1
fi

cd "${CODE_DIR}"

python3 generate.py \
  --pretrained_model_path "${MODEL}" \
  --enable_vision \
  --mm_jsonl "${MM_JSONL}" \
  --use_default_prompt \
  --n_samples "${N_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --output_path "${OUT_DIR}"
