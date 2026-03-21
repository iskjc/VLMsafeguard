#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/srj/VLMsafeguard"
CODE_DIR="${REPO_ROOT}/code"

MODEL="${MODEL:-/s/models/llava-series/llava-1.5-7b-hf}"
MM_JSONL="${MM_JSONL:-${CODE_DIR}/data/coco2014_build/coco2014_val_mm.jsonl}"
BASE_OUT_DIR="${BASE_OUT_DIR:-./outputs/coco2014/soft_sweep}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
N_SAMPLES="${N_SAMPLES:-1}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-all}"
PROMPT_LENGTH="${PROMPT_LENGTH:-default}"

MODEL="${MODEL}" \
MM_JSONL="${MM_JSONL}" \
BASE_OUT_DIR="${BASE_OUT_DIR}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
N_SAMPLES="${N_SAMPLES}" \
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE}" \
PROMPT_LENGTH="${PROMPT_LENGTH}" \
  bash "${SCRIPT_DIR}/generate_mm_soft_sweep.sh"
