#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/srj/VLMsafeguard"
CODE_DIR="${REPO_ROOT}/code"
SCRIPT_DIR="${CODE_DIR}/scripts/multimodal"

MODEL="${MODEL:-/s/models/llava-series/llava-1.5-7b-hf}"
MM_JSONL="${MM_JSONL:-${CODE_DIR}/data/data_vlguard/processed/train_mm.jsonl}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-all}"
PROMPT_LENGTH="${PROMPT_LENGTH:-default}"
OUT_DIR="${OUT_DIR:-./trained_prompts_mm}"
DEFAULT_EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-40}"

# Format per row:
# run_name learning_rate num_epochs batch_size effective_batch_size
RUNS=(
  "lr1e-5_ep10_ebs80_rl1 1e-5 10 2 80"
  "lr1e-4_ep10_ebs80_rl1 1e-4 10 2 80"
  "lr1e-5_ep10_ebs40_rl1 1e-5 10 2 40"
  "lr1e-5_ep15_ebs80_rl1 1e-5 15 2 80"
  "lr1e-4_ep10_ebs40_rl1 1e-4 10 2 40"
  "lr1e-5_ep5_ebs80_rl1 1e-5 5 2 80"
)


if [[ ! -f "${MM_JSONL}" ]]; then
  echo "Missing mm_jsonl: ${MM_JSONL}" >&2
  exit 1
fi

for run_cfg in "${RUNS[@]}"; do
  read -r RUN_NAME LR NUM_EPOCHS BATCH_SIZE EFFECTIVE_BATCH_SIZE <<< "${run_cfg}"
  EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-${DEFAULT_EFFECTIVE_BATCH_SIZE}}"
  echo "============================================================"
  echo "Starting run: ${RUN_NAME}"
  echo "  lr=${LR}"
  echo "  num_epochs=${NUM_EPOCHS}"
  echo "  batch_size=${BATCH_SIZE}"
  echo "  effective_batch_size=${EFFECTIVE_BATCH_SIZE}"
  echo "============================================================"

  MODEL="${MODEL}" \
  MM_JSONL="${MM_JSONL}" \
  SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE}" \
  PROMPT_LENGTH="${PROMPT_LENGTH}" \
  OUT_DIR="${OUT_DIR}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE}" \
  NUM_EPOCHS="${NUM_EPOCHS}" \
  LR="${LR}" \
  RUN_NAME_SUFFIX="${RUN_NAME}" \
    bash "${SCRIPT_DIR}/train_soft_prompt_mm.sh"
done
