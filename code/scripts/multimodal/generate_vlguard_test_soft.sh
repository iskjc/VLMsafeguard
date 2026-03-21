#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/srj/VLMsafeguard"
CODE_DIR="${REPO_ROOT}/code"

MODEL="${MODEL:-/s/models/llava-series/llava-1.5-7b-hf}"
MM_JSONL="${MM_JSONL:-${CODE_DIR}/data/data_vlguard/processed/test_mm.jsonl}"
OUT_DIR="${OUT_DIR:-./outputs/vlguard_test/soft_all_default}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-all}"
PROMPT_LENGTH="${PROMPT_LENGTH:-default}"

MODEL="${MODEL}" MM_JSONL="${MM_JSONL}" OUT_DIR="${OUT_DIR}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE}" PROMPT_LENGTH="${PROMPT_LENGTH}" \
  bash "${SCRIPT_DIR}/generate_mm_soft.sh"
