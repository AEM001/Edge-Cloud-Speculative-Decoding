#!/usr/bin/env bash
# Convenience launcher for a two-3090 local run.
#
# GPU 0: target verify server
# GPU 1: draft model / quick experiment

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY_PORT="${VERIFY_PORT:-6008}"
VERIFY_SERVER_URL="${VERIFY_SERVER_URL:-http://localhost:${VERIFY_PORT}}"

export VERIFY_GPU_ID="${VERIFY_GPU_ID:-0}"
export VERIFY_DEVICE="${VERIFY_DEVICE:-cuda:${VERIFY_GPU_ID}}"
export VERIFY_DTYPE="${VERIFY_DTYPE:-auto}"
export VERIFY_ATTN_IMPLEMENTATION="${VERIFY_ATTN_IMPLEMENTATION:-eager}"
export VERIFY_MODEL_PATH="${VERIFY_MODEL_PATH:-/root/autodl-tmp/Qwen3-14B-AWQ}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export DRAFT_GPU_ID="${DRAFT_GPU_ID:-1}"
export DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:${DRAFT_GPU_ID}}"
export DRAFT_DTYPE="${DRAFT_DTYPE:-fp16}"
export DRAFT_ATTN_IMPLEMENTATION="${DRAFT_ATTN_IMPLEMENTATION:-sdpa}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/root/autodl-tmp/Qwen3-1.7B}"

export RETRIEVAL_CHUNK_SIZE="${RETRIEVAL_CHUNK_SIZE:-64}"
export RETRIEVE_TOP_K="${RETRIEVE_TOP_K:-16}"
export RETRIEVE_EVERY_N_STEPS="${RETRIEVE_EVERY_N_STEPS:-16}"
export SPECEXTEND_ASYNC_PIPELINE="${SPECEXTEND_ASYNC_PIPELINE:-1}"
export SPECEXTEND_PIPELINE_OFFSETS="${SPECEXTEND_PIPELINE_OFFSETS:-full,half}"
export VERIFY_SERVER_URL

SERVER_LOG="$REPO_DIR/verify_server_2x3090.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting verify server on GPU ${VERIFY_GPU_ID}, port ${VERIFY_PORT} ..."
PORT="$VERIFY_PORT" bash "$REPO_DIR/start_verify.sh" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for verify server health check ..."
for _ in $(seq 1 120); do
  if curl -fsS "${VERIFY_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "Verify server is ready."
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Verify server exited early. Last log lines:"
    tail -80 "$SERVER_LOG"
    exit 1
  fi
  sleep 2
done

if ! curl -fsS "${VERIFY_SERVER_URL}/health" >/dev/null 2>&1; then
  echo "Verify server did not become healthy. Last log lines:"
  tail -80 "$SERVER_LOG"
  exit 1
fi

echo "Running quick test on draft GPU ${DRAFT_GPU_ID} ..."
bash "$REPO_DIR/quick.sh"
