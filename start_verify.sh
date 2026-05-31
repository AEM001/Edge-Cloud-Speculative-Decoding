#!/usr/bin/env bash
# Start the cloud verify server on this machine.
# Serves Qwen3-14B-AWQ on GPU 0, binds 0.0.0.0 for local network access.
#
# Usage:
#   bash start_verify.sh [--port 6007]
#
# Stop with: Ctrl+C

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=${PORT:-6007}

# ============================================
# VERIFY SERVER CONFIGURATION
# ============================================

# Verify Model Settings
VERIFY_MODEL_PATH="${VERIFY_MODEL_PATH:-$REPO_DIR/models/Qwen3-14B-AWQ}"
VERIFY_GPU_ID="${VERIFY_GPU_ID:-0}"
VERIFY_DEVICE="${VERIFY_DEVICE:-cuda:$VERIFY_GPU_ID}"
VERIFY_MAX_LEN="${VERIFY_MAX_LEN:-6000}"
VERIFY_DTYPE="${VERIFY_DTYPE:-fp8}"
VERIFY_ATTN_IMPLEMENTATION="${VERIFY_ATTN_IMPLEMENTATION:-sdpa}"
# Target model (14B AWQ) — reserve most memory by default
VERIFY_GPU_MEMORY_FRACTION="${VERIFY_GPU_MEMORY_FRACTION:-0.85}"

# Use .venv environment
PYTHON="$REPO_DIR/.venv/bin/python3"

export CUDA_VISIBLE_DEVICES=$VERIFY_GPU_ID
export VERIFY_MODEL_PATH
export VERIFY_DEVICE
export VERIFY_MAX_LEN
export VERIFY_DTYPE
export VERIFY_ATTN_IMPLEMENTATION
export VERIFY_GPU_MEMORY_FRACTION

# Parse optional --port arg
while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== Starting verify server on port $PORT ==="
echo "Repo: $REPO_DIR"
echo "Python: $PYTHON"
echo "Backend: custom_qwen3"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Verify device: $VERIFY_DEVICE"
echo "Verify dtype: $VERIFY_DTYPE"
echo "Verify max model length: $VERIFY_MAX_LEN"
echo ""

cd "$REPO_DIR"
PYTHONPATH="$REPO_DIR/scripts" "$PYTHON" -m server.verify_server --host 0.0.0.0 --port $PORT
