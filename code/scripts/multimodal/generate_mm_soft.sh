#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/srj/VLMsafeguard"
CODE_DIR="${REPO_ROOT}/code"

MODEL="${MODEL:-/s/models/llava-series/llava-1.5-7b-hf}"
MM_JSONL="${MM_JSONL:-}"
OUT_DIR="${OUT_DIR:-./outputs/mm/soft_all_default}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
N_SAMPLES="${N_SAMPLES:-1}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-all}"
PROMPT_LENGTH="${PROMPT_LENGTH:-default}"
MODEL_NAME="$(basename "${MODEL}")"
SOFT_PROMPT_PATH="${SOFT_PROMPT_PATH:-}"
SOFT_PROMPT_VERSION="${SOFT_PROMPT_VERSION:-${RUN_NAME_SUFFIX:-}}"
SOFT_PROMPT_DIR="${CODE_DIR}/trained_prompts_mm/${MODEL_NAME}"
SOFT_PROMPT_STEM="type.${SYSTEM_PROMPT_TYPE}_length.${PROMPT_LENGTH}"
DEFAULT_SOFT_PROMPT_PATH="${SOFT_PROMPT_DIR}/${SOFT_PROMPT_STEM}.safetensors"

if [[ -z "${MM_JSONL}" ]]; then
  echo "MM_JSONL is required. Example: MM_JSONL=${CODE_DIR}/data/mme_build/llava_mme_mm.jsonl bash $0" >&2
  exit 1
fi

if [[ ! -f "${MM_JSONL}" ]]; then
  echo "Missing mm_jsonl: ${MM_JSONL}" >&2
  exit 1
fi

if [[ -z "${SOFT_PROMPT_PATH}" ]]; then
  if [[ -n "${SOFT_PROMPT_VERSION}" ]]; then
    SOFT_PROMPT_PATH="${SOFT_PROMPT_DIR}/${SOFT_PROMPT_STEM}__${SOFT_PROMPT_VERSION}.safetensors"
  elif [[ -f "${DEFAULT_SOFT_PROMPT_PATH}" ]]; then
    SOFT_PROMPT_PATH="${DEFAULT_SOFT_PROMPT_PATH}"
  elif [[ -d "${SOFT_PROMPT_DIR}" ]]; then
    SOFT_PROMPT_PATH="$(
      find "${SOFT_PROMPT_DIR}" -maxdepth 1 -type f -name "${SOFT_PROMPT_STEM}__*.safetensors" -printf '%T@ %p\n' \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"
  fi
fi

if [[ ! -f "${SOFT_PROMPT_PATH}" ]]; then
  echo "Missing soft prompt: ${SOFT_PROMPT_PATH}" >&2
  if [[ -d "${SOFT_PROMPT_DIR}" ]]; then
    echo "Available candidates under ${SOFT_PROMPT_DIR}:" >&2
    find "${SOFT_PROMPT_DIR}" -maxdepth 1 -type f -name "${SOFT_PROMPT_STEM}*.safetensors" | sort >&2 || true
  fi
  exit 1
fi

cd "${CODE_DIR}"

python3 generate.py \
  --pretrained_model_path "${MODEL}" \
  --enable_vision \
  --mm_jsonl "${MM_JSONL}" \
  --use_soft_prompt \
  --system_prompt_type "${SYSTEM_PROMPT_TYPE}" \
  --prompt_length "${PROMPT_LENGTH}" \
  --soft_prompt_path "${SOFT_PROMPT_PATH}" \
  --n_samples "${N_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --output_path "${OUT_DIR}"
