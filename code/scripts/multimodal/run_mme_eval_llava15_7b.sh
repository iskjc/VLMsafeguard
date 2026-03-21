#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/srj/VLMsafeguard"
CODE_DIR="${REPO_ROOT}/code"
SCRIPT_DIR="${CODE_DIR}/scripts/multimodal"

MODEL="${MODEL:-/s/models/llava-series/llava-1.5-7b-hf}"
MODEL_NAME="$(basename "${MODEL}")"
MM_JSONL="${MM_JSONL:-${CODE_DIR}/data/mme_build/llava_mme_mm.jsonl}"
MME_ROOT="${MME_ROOT:-/s/datasets/MME/MME_Benchmark_release_version/MME_Benchmark}"
EVAL_TOOL_DIR="${EVAL_TOOL_DIR:-/s/datasets/MME/eval_tool}"

GEN_OUT_DIR="${GEN_OUT_DIR:-./outputs/mme/hard_default_lbase}"
EXPORT_DIR="${EXPORT_DIR:-${CODE_DIR}/eval_results/mme/hard_for_eval_tool_lbase}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"

if [[ ! -f "${MM_JSONL}" ]]; then
  echo "Missing mm_jsonl: ${MM_JSONL}" >&2
  exit 1
fi

if [[ ! -d "${MME_ROOT}" ]]; then
  echo "Missing MME root: ${MME_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${EVAL_TOOL_DIR}" ]]; then
  echo "Missing eval_tool dir: ${EVAL_TOOL_DIR}" >&2
  exit 1
fi

cd "${CODE_DIR}"

python3 generate.py \
  --pretrained_model_path "${MODEL}" \
  --enable_vision \
  --mm_jsonl "${MM_JSONL}" \
  --use_default_prompt \
  --n_samples 1 \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --output_path "${GEN_OUT_DIR}"

PRED_CSV="${CODE_DIR}/${GEN_OUT_DIR#./}/${MODEL_NAME}_with_default_llava_mme/output_greedy.csv"

if [[ ! -f "${PRED_CSV}" ]]; then
  echo "Expected prediction csv not found: ${PRED_CSV}" >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/export_mme_for_eval_tool.py" \
  --pred-csv "${PRED_CSV}" \
  --mm-jsonl "${MM_JSONL}" \
  --mme-root "${MME_ROOT}" \
  --output-dir "${EXPORT_DIR}"

bash "${SCRIPT_DIR}/eval_mme_results.sh" "${EXPORT_DIR}" "${EVAL_TOOL_DIR}"
