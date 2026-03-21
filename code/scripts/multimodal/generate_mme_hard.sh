#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/srj/VLMsafeguard"
CODE_DIR="${REPO_ROOT}/code"

MODEL="${MODEL:-/s/models/llava-series/llava-1.5-7b-hf}"
MM_JSONL="${MM_JSONL:-${CODE_DIR}/data/mme_build/llava_mme_mm.jsonl}"
OUT_DIR="${OUT_DIR:-./outputs/mme/hard_default}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"

MODEL="${MODEL}" MM_JSONL="${MM_JSONL}" OUT_DIR="${OUT_DIR}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  bash "${SCRIPT_DIR}/generate_mm_hard.sh"
