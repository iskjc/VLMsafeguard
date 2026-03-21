#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${1:-}"
EVAL_TOOL_DIR="${2:-/s/datasets/MME/eval_tool}"

if [[ -z "${RESULT_DIR}" ]]; then
  echo "Usage: bash code/scripts/eval_mme_results.sh <result_dir> [eval_tool_dir]" >&2
  exit 1
fi

RESULT_DIR="$(realpath "${RESULT_DIR}")"
EVAL_TOOL_DIR="$(realpath "${EVAL_TOOL_DIR}")"

if [[ ! -d "${RESULT_DIR}" ]]; then
  echo "Result dir does not exist: ${RESULT_DIR}" >&2
  exit 1
fi

cd "${EVAL_TOOL_DIR}"
echo "Running MME eval with results_dir: ${RESULT_DIR}"
python3 calculation.py --results_dir "${RESULT_DIR}"
