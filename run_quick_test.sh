#!/bin/bash
# Quick test: 2 easy + 2 hard prompts comparing Direct vs K=2,4
# This verifies speculative decoding works before running full experiment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UBUNTU_VERIFY_DIR="/root/autodl-tmp/ubuntu-verify"
VERIFY_PORT=6006

# Configuration - reduced GPU memory to avoid conflict
export AINFRA_DRAFT_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
export AINFRA_DRAFT_MODEL_PATH="/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-1.5B-Instruct"
export AINFRA_TEMPERATURE=0.0
export AINFRA_MAX_MODEL_LEN=4096
export AINFRA_GPU_MEMORY_UTILIZATION=0.25  # Adjusted for 14B AWQ verify model

echo "=== Quick Test: Direct vs Speculative (K=2,4) ==="
echo "Verify model: /root/autodl-tmp/ubuntu-verify/models/Qwen2.5-14B-Instruct-AWQ"
echo "Draft model: $AINFRA_DRAFT_MODEL_PATH"
echo "GPU util: $AINFRA_GPU_MEMORY_UTILIZATION (adjusted for 14B AWQ)"
echo ""

# Check if verify server is already running
if curl -s "http://localhost:$VERIFY_PORT/health" > /dev/null 2>&1; then
    echo "✓ Verification server already running on port $VERIFY_PORT"
    VERIFY_PID=""
else
    echo "=== Starting Verification Server ==="
    cd "$UBUNTU_VERIFY_DIR"
    
    if [ -d "$UBUNTU_VERIFY_DIR/.venv" ]; then
        source "$UBUNTU_VERIFY_DIR/.venv/bin/activate"
    else
        echo "Error: Virtual environment not found"
        exit 1
    fi
    
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    
    # Start API server in background
    echo "Starting verification server on port $VERIFY_PORT..."
    python src/api_server.py --host 0.0.0.0 --port $VERIFY_PORT &
    VERIFY_PID=$!
    
    # Wait for server
    echo "Waiting for server to start..."
    for i in {1..60}; do
        if curl -s "http://localhost:$VERIFY_PORT/health" > /dev/null 2>&1; then
            echo "✓ Server ready!"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "✗ Server failed to start"
            exit 1
        fi
        sleep 2
    done
    echo ""
fi

# Run quick test
echo "=== Running Quick Test ==="
cd "$SCRIPT_DIR"

if [ -d "$UBUNTU_VERIFY_DIR/.venv" ]; then
    source "$UBUNTU_VERIFY_DIR/.venv/bin/activate"
fi

mkdir -p outputs_quick
python quick_test.py 2>&1 | tee outputs_quick/test_log.txt

TEST_EXIT=${PIPESTATUS[0]}

# Cleanup if we started the server
if [ -n "$VERIFY_PID" ]; then
    echo ""
    echo "Stopping verification server..."
    kill $VERIFY_PID 2>/dev/null || true
fi

if [ $TEST_EXIT -eq 0 ]; then
    echo ""
    echo "✓ Quick test PASSED - Speculative is faster than Direct!"
    echo "Ready to run full experiment."
    exit 0
else
    echo ""
    echo "✗ Quick test FAILED"
    exit 1
fi
