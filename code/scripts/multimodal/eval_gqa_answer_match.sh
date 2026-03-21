#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/srj/VLMsafeguard"
CODE_DIR="${REPO_ROOT}/code"
SCRIPT_DIR="${CODE_DIR}/scripts/multimodal"

PRED_CSV="${PRED_CSV:-${CODE_DIR}/outputs/gqa/hard_default/llava-1.5-7b-hf_with_default_llava_gqa_testdev_balanced/output_greedy.csv}"
MM_JSONL="${MM_JSONL:-${CODE_DIR}/data/gqa_build/llava_gqa_testdev_balanced_mm.jsonl}"
OUT_JSON="${OUT_JSON:-${CODE_DIR}/eval_results/gqa/hard_default_answer_match.json}"

cd "${CODE_DIR}"

python3 "${SCRIPT_DIR}/eval_mm_answer_match.py" \
  --pred-csv "${PRED_CSV}" \
  --mm-jsonl "${MM_JSONL}" \
  --out-json "${OUT_JSON}"
