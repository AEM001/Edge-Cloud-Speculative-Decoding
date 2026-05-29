#!/usr/bin/env bash
# Run the quick test experiment only.
# Assumes the verify server is already running.
#
# Usage:
#   bash run_quick.sh
#
# Key env overrides (all optional):
#   VERIFY_SERVER_URL, DRAFT_MODEL_PATH, DRAFT_GPU_ID, MAX_TOKENS,
#   PROMPT_TYPES, PROMPT_COUNT, NODES, MAX_DEPTH, RETRIEVE_EVERY_N_STEPS

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

export VERIFY_SERVER_URL="${VERIFY_SERVER_URL:-http://localhost:6008}"
export DRAFT_GPU_ID="${DRAFT_GPU_ID:-1}"
export DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:${DRAFT_GPU_ID}}"
export DRAFT_DTYPE="${DRAFT_DTYPE:-fp16}"
export DRAFT_ATTN_IMPLEMENTATION="${DRAFT_ATTN_IMPLEMENTATION:-sdpa}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$REPO_DIR/models/Qwen3-1.7B}"
export DRAFT_RECENT_TOKENS="${DRAFT_RECENT_TOKENS:-256}"
export DRAFT_MAX_LEN="${DRAFT_MAX_LEN:-32768}"
export RETRIEVAL_CHUNK_SIZE="${RETRIEVAL_CHUNK_SIZE:-64}"
export RETRIEVE_TOP_K="${RETRIEVE_TOP_K:-16}"
export RETRIEVE_EVERY_N_STEPS="${RETRIEVE_EVERY_N_STEPS:-16}"
export SPECEXTEND_ASYNC_PIPELINE="${SPECEXTEND_ASYNC_PIPELINE:-1}"
export SPECEXTEND_PIPELINE_OFFSETS="${SPECEXTEND_PIPELINE_OFFSETS:-full,half}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MAX_TOKENS="${MAX_TOKENS:-256}"
PROMPT_COUNT="${PROMPT_COUNT:-1}"
PROMPT_TYPES="${PROMPT_TYPES:-pg19}"
PROMPT_INPUT_TOKENS="${PROMPT_INPUT_TOKENS:-2048}"
NODES="${NODES:-32}"
MAX_DEPTH="${MAX_DEPTH:-8}"
THRESHOLD="${THRESHOLD:-0.7}"

echo "=========================================="
echo "Running Quick Test"
echo "Verify Server: ${VERIFY_SERVER_URL}"
echo "Draft Model:   ${DRAFT_MODEL_PATH} @ ${DRAFT_DEVICE}"
echo "Max Tokens:    ${MAX_TOKENS}"
echo "Prompt Types:  ${PROMPT_TYPES}"
echo "Nodes:         ${NODES}  Depth: ${MAX_DEPTH}"
echo "Retrieve every N steps: ${RETRIEVE_EVERY_N_STEPS}"
echo "=========================================="

cd "$REPO_DIR"
PYTHONPATH="$REPO_DIR/scripts" python3 scripts/experiments/quick_test.py \
    --max-tokens "${MAX_TOKENS}" \
    --prompt-count "${PROMPT_COUNT}" \
    --prompt-types ${PROMPT_TYPES} \
    --prompt-input-tokens "${PROMPT_INPUT_TOKENS}" \
    --nodes "${NODES}" \
    --max-depth "${MAX_DEPTH}" \
    --threshold "${THRESHOLD}" \
    --retrieval-chunk-size "${RETRIEVAL_CHUNK_SIZE}" \
    --retrieve-top-k "${RETRIEVE_TOP_K}" \
    --retrieve-every-n-steps "${RETRIEVE_EVERY_N_STEPS}"
