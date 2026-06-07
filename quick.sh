#!/bin/bash

# Quick benchmark launcher.
# Most defaults live in quick_benchmark_config.json. For a normal run, edit that
# file's methods/profile list, then run:
#   bash quick.sh
#
# Common one-off overrides:
#   MAX_TOKENS=256 DATASET_SPLIT=pg19_4K bash quick.sh
#   METHODS="direct specextend_gpu" PROMPT_COUNT=3 bash quick.sh
#   DRAFT_MODE=branching DRAFT_TREE_NODES=32 DRAFT_TREE_MAX_DEPTH=8 bash quick.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="${CONFIG_FILE:-quick_benchmark_config.json}"
PYTHON="${PYTHON:-python3}"

# ---------- Draft model / server defaults ----------
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/root/autodl-tmp/Qwen3-1.7B}"
export DRAFT_GPU_ID="${DRAFT_GPU_ID:-1}"
export DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:${DRAFT_GPU_ID}}"
export DRAFT_DTYPE="${DRAFT_DTYPE:-fp16}"
export DRAFT_MAX_LEN="${DRAFT_MAX_LEN:-32768}"
export DRAFT_ATTN_IMPLEMENTATION="${DRAFT_ATTN_IMPLEMENTATION:-sdpa}"
export VERIFY_SERVER_URL="${VERIFY_SERVER_URL:-http://localhost:6007}"
export QUICK_TEST_TIMEOUT_SEC="${QUICK_TEST_TIMEOUT_SEC:-600}"
export SPECEXTEND_ASYNC_PIPELINE="${SPECEXTEND_ASYNC_PIPELINE:-1}"
export SPECEXTEND_PIPELINE_OFFSETS="${SPECEXTEND_PIPELINE_OFFSETS:-full,half}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

CMD=(
  "$PYTHON" scripts/experiments/quick_test.py
  --config "$CONFIG_FILE"
)

[[ -n "${METHODS:-}" ]] && CMD+=(--methods ${METHODS})
[[ -n "${MAX_TOKENS:-}" ]] && CMD+=(--max-tokens "$MAX_TOKENS")
[[ -n "${PROMPT_COUNT:-}" ]] && CMD+=(--prompt-count "$PROMPT_COUNT")
[[ -n "${PROMPT_TYPES:-}" ]] && CMD+=(--prompt-types ${PROMPT_TYPES})
[[ -n "${DATASET_SPLIT:-}" ]] && CMD+=(--dataset-split "$DATASET_SPLIT")
[[ -n "${PROMPT_INPUT_TOKENS:-}" ]] && CMD+=(--prompt-input-tokens "$PROMPT_INPUT_TOKENS")
[[ -n "${DRAFT_MODE:-}" ]] && CMD+=(--draft-mode "$DRAFT_MODE")
[[ -n "${DRAFT_LENGTH:-}" ]] && CMD+=(--draft-length "$DRAFT_LENGTH")
[[ -n "${DRAFT_TREE_NODES:-}" ]] && CMD+=(--draft-tree-nodes "$DRAFT_TREE_NODES")
[[ -n "${DRAFT_TREE_MAX_DEPTH:-}" ]] && CMD+=(--draft-tree-max-depth "$DRAFT_TREE_MAX_DEPTH")
[[ -n "${RETRIEVAL_CHUNK_SIZE:-}" ]] && CMD+=(--retrieval-chunk-size "$RETRIEVAL_CHUNK_SIZE")
[[ -n "${RETRIEVE_TOP_K:-}" ]] && CMD+=(--retrieve-top-k "$RETRIEVE_TOP_K")
[[ -n "${RETRIEVE_EVERY_N_STEPS:-}" ]] && CMD+=(--retrieve-every-n-steps "$RETRIEVE_EVERY_N_STEPS")

echo "=========================================="
echo "Running Quick Benchmark"
echo "Config:            ${CONFIG_FILE}"
echo "Methods override:  ${METHODS:-<from config>}"
echo "Verify Server URL: ${VERIFY_SERVER_URL}"
echo "Draft Model:       ${DRAFT_MODEL_PATH}"
echo "Draft Device:      ${DRAFT_DEVICE}"
echo "=========================================="
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n==========================================\n\n'

"${CMD[@]}"

"$PYTHON" scripts/experiments/analyze_quick_run.py
