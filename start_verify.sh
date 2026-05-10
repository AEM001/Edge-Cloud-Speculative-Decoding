#!/usr/bin/env bash
# Start the verify server directly.
# By default the 14B verifier runs on GPU 0.
#
# Usage:
#   bash scripts/start_verify.sh [--port 6006]
#
# Stop with: Ctrl+C

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=${PORT:-6006}

# Use .venv environment
PYTHON="$REPO_DIR/.venv/bin/python3"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VERIFY_TENSOR_PARALLEL_SIZE=${VERIFY_TENSOR_PARALLEL_SIZE:-1}
export VERIFY_GPU_MEM=${VERIFY_GPU_MEM:-0.9}
export VERIFY_ENFORCE_EAGER=${VERIFY_ENFORCE_EAGER:-1}
export VERIFY_ENABLE_PREFIX_CACHING=${VERIFY_ENABLE_PREFIX_CACHING:-1}

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
echo "Attention backend: $VLLM_ATTENTION_BACKEND"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Tensor parallel size: $VERIFY_TENSOR_PARALLEL_SIZE"
echo "Verify GPU memory utilization: $VERIFY_GPU_MEM"
echo ""

cd "$REPO_DIR"
PYTHONPATH="$REPO_DIR/scripts" "$PYTHON" -m server.verify_server --host 0.0.0.0 --port $PORT
