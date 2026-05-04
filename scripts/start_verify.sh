#!/usr/bin/env bash
# Start the verify server (Qwen2.5-7B-AWQ) directly.
# The server occupies GPU 0 by default.
#
# Usage:
#   bash scripts/start_verify.sh [--port 6006]
#
# Stop with: Ctrl+C

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT=${PORT:-6006}

# Use .venv environment
PYTHON="$REPO_DIR/.venv/bin/python3"

# Use SDPA attention backend (built into PyTorch, no flash-attn required)
export VLLM_ATTENTION_BACKEND=sdpa

# Verify model runs on GPU 0
export CUDA_VISIBLE_DEVICES=0

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
echo ""

cd "$REPO_DIR"
PYTHONPATH="$REPO_DIR" "$PYTHON" -m server.verify_server --host 0.0.0.0 --port $PORT
