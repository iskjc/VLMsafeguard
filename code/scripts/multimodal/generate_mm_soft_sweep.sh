#!/usr/bin/env bash
set -euo pipefail

# Generic multimodal sweep entry.
# Dataset-specific wrappers like generate_mme_soft_sweep.sh call into this script
# after filling MM_JSONL / BASE_OUT_DIR.

REPO_ROOT="/home/srj/VLMsafeguard"
CODE_DIR="${REPO_ROOT}/code"
SCRIPT_DIR="${CODE_DIR}/scripts/multimodal"

MODEL="${MODEL:-/s/models/llava-series/llava-1.5-7b-hf}"
MODEL_NAME="$(basename "${MODEL}")"
MM_JSONL="${MM_JSONL:-${CODE_DIR}/data/data_vlguard/processed/test_mm.jsonl}"
BASE_OUT_DIR="${BASE_OUT_DIR:-./outputs/vlguard_test/soft_sweep}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
N_SAMPLES="${N_SAMPLES:-1}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-all}"
PROMPT_LENGTH="${PROMPT_LENGTH:-default}"
SOFT_PROMPT_DIR="${SOFT_PROMPT_DIR:-${CODE_DIR}/trained_prompts_mm/${MODEL_NAME}}"
SOFT_PROMPT_GLOB="${SOFT_PROMPT_GLOB:-type.${SYSTEM_PROMPT_TYPE}_length.${PROMPT_LENGTH}*.safetensors}"

if [[ ! -f "${MM_JSONL}" ]]; then
  echo "Missing mm_jsonl: ${MM_JSONL}" >&2
  exit 1
fi

if [[ ! -d "${SOFT_PROMPT_DIR}" ]]; then
  echo "Missing soft prompt dir: ${SOFT_PROMPT_DIR}" >&2
  exit 1
fi

mapfile -t SOFT_PROMPT_FILES < <(find "${SOFT_PROMPT_DIR}" -maxdepth 1 -type f -name "${SOFT_PROMPT_GLOB}" | sort)

if [[ "${#SOFT_PROMPT_FILES[@]}" -eq 0 ]]; then
  echo "No soft prompt files matched ${SOFT_PROMPT_DIR}/${SOFT_PROMPT_GLOB}" >&2
  exit 1
fi

for SOFT_PROMPT_PATH in "${SOFT_PROMPT_FILES[@]}"; do
  RUN_NAME="$(basename "${SOFT_PROMPT_PATH}" .safetensors)"
  OUT_DIR="${BASE_OUT_DIR}/${RUN_NAME}"

  echo "============================================================"
  echo "Starting generate run: ${RUN_NAME}"
  echo "  out_dir=${OUT_DIR}"
  echo "  soft_prompt_path=${SOFT_PROMPT_PATH}"
  echo "============================================================"

  MODEL="${MODEL}" \
  MM_JSONL="${MM_JSONL}" \
  OUT_DIR="${OUT_DIR}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  N_SAMPLES="${N_SAMPLES}" \
  SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE}" \
  PROMPT_LENGTH="${PROMPT_LENGTH}" \
  SOFT_PROMPT_PATH="${SOFT_PROMPT_PATH}" \
    bash "${SCRIPT_DIR}/generate_mm_soft.sh"
done
