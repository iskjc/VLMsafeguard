#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/srj/VLMsafeguard"
CODE_DIR="${REPO_ROOT}/code"

MM_JSONL="${MM_JSONL:-${CODE_DIR}/data/data_vlguard/processed/test_mm.jsonl}"
BASELINE_CSV="${BASELINE_CSV:-${CODE_DIR}/outputs/vlguard_test/hard_default/llava-1.5-7b-hf_with_default_vlguard_test/output_greedy.csv}"
SOFT_CSV="${SOFT_CSV:-${CODE_DIR}/outputs/vlguard_test/soft_all_default/llava-1.5-7b-hf_with_soft_all_default_vlguard_test/output_greedy.csv}"
OUT_JSON="${OUT_JSON:-${CODE_DIR}/eval_results/vlguard/test/default_vs_soft.json}"

cd "${CODE_DIR}"

python3 compare_vlguard_outputs.py \
  --mm_jsonl "${MM_JSONL}" \
  --baseline_csv "${BASELINE_CSV}" \
  --soft_csv "${SOFT_CSV}" \
  --out_json "${OUT_JSON}"
