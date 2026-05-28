#!/usr/bin/env bash
# Start the verify server directly.
# By default the 14B verifier runs on GPU 0.
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
VERIFY_MODEL_PATH="${VERIFY_MODEL_PATH:-$REPO_DIR/models/Qwen2.5-14B-Instruct-AWQ}"
VERIFY_GPU_ID="${VERIFY_GPU_ID:-0}"
VERIFY_GPU_MEM="${VERIFY_GPU_MEM:-0.60}"
VERIFY_MAX_LEN="${VERIFY_MAX_LEN:-12000}"
VERIFY_QUANTIZATION="${VERIFY_QUANTIZATION:-awq}"
VERIFY_TENSOR_PARALLEL_SIZE="${VERIFY_TENSOR_PARALLEL_SIZE:-1}"

# CUDA Graph Settings
VERIFY_ENFORCE_EAGER="${VERIFY_ENFORCE_EAGER:-1}"  # 0 = enable CUDA graph, 1 = disable
VERIFY_ENABLE_PREFIX_CACHING="${VERIFY_ENABLE_PREFIX_CACHING:-1}"

# Attention Backend
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

# Use .venv environment
PYTHON="$REPO_DIR/.venv/bin/python3"

export CUDA_VISIBLE_DEVICES=$VERIFY_GPU_ID
export VERIFY_MODEL_PATH
export VERIFY_GPU_MEM
export VERIFY_MAX_LEN
export VERIFY_QUANTIZATION
export VERIFY_TENSOR_PARALLEL_SIZE
export VERIFY_ENFORCE_EAGER
export VERIFY_ENABLE_PREFIX_CACHING
export VLLM_ATTENTION_BACKEND

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
echo "Verify max model length: $VERIFY_MAX_LEN"
echo ""

cd "$REPO_DIR"
PYTHONPATH="$REPO_DIR/scripts" "$PYTHON" -m server.verify_server --host 0.0.0.0 --port $PORT
