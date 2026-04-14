#!/bin/bash
# Run local experiment with network simulation: k=2,4,6,8
# This script starts the verification server locally and runs the experiment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UBUNTU_VERIFY_DIR="/root/autodl-tmp/ubuntu-verify"
EXPERIMENT_SCRIPT="experiment_local_k2k8.py"
OUTPUT_DIR="outputs_local_k2k8"

# Configuration
VERIFY_PORT=6006
VERIFY_MODEL_PATH="/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-14B-Instruct-AWQ"

# Draft model configuration (1.5B)
export AINFRA_DRAFT_MODEL_NAME="${AINFRA_DRAFT_MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
export AINFRA_DRAFT_MODEL_PATH="${AINFRA_DRAFT_MODEL_PATH:-/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-1.5B-Instruct}"
export AINFRA_TEMPERATURE=0.0
export AINFRA_MAX_MODEL_LEN=4096
export AINFRA_GPU_MEMORY_UTILIZATION=0.25  # Adjusted for 14B AWQ verify model

echo "=== Local Experiment Runner (k=2,4,6,8 with Network Simulation) ==="
echo "Verify model: $VERIFY_MODEL_PATH"
echo "Draft model: $AINFRA_DRAFT_MODEL_PATH"
echo "Verify port: $VERIFY_PORT"
echo "Output: $OUTPUT_DIR/"
echo ""

# Check if models exist
if [ ! -d "$VERIFY_MODEL_PATH" ]; then
    echo "Error: Verify model not found at $VERIFY_MODEL_PATH"
    exit 1
fi

if [ ! -d "$AINFRA_DRAFT_MODEL_PATH" ]; then
    echo "Warning: Draft model not found at $AINFRA_DRAFT_MODEL_PATH"
    echo "The experiment will try to download from HuggingFace if needed."
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$VERIFY_PID" ]; then
        echo "Stopping verification server (PID: $VERIFY_PID)..."
        kill $VERIFY_PID 2>/dev/null || true
        wait $VERIFY_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Start verification server
echo "=== Starting Verification Server ==="
cd "$UBUNTU_VERIFY_DIR"

# Activate virtual environment
if [ -d "$UBUNTU_VERIFY_DIR/.venv" ]; then
    source "$UBUNTU_VERIFY_DIR/.venv/bin/activate"
else
    echo "Error: Virtual environment not found at $UBUNTU_VERIFY_DIR/.venv"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Start the API server in background
echo "Starting verification server on port $VERIFY_PORT..."
python src/api_server.py --host 0.0.0.0 --port $VERIFY_PORT &
VERIFY_PID=$!

# Wait for server to start
echo "Waiting for verification server to start..."
for i in {1..30}; do
    if curl -s "http://localhost:$VERIFY_PORT/health" > /dev/null 2>&1; then
        echo "✓ Verification server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ Verification server failed to start within 30 seconds"
        exit 1
    fi
    sleep 1
done

# Show server info
echo ""
echo "Server info:"
curl -s "http://localhost:$VERIFY_PORT/health" | python -m json.tool 2>/dev/null || curl -s "http://localhost:$VERIFY_PORT/health"
echo ""

# Run experiment
echo ""
echo "=== Starting Experiment (k=2,4,6,8 with Network Simulation) ==="
cd "$SCRIPT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the experiment with logging
python "experiments/$EXPERIMENT_SCRIPT" 2>&1 | tee "$OUTPUT_DIR/experiment_log.txt"

echo ""
echo "=== Experiment Complete ==="
echo "Results saved to: $SCRIPT_DIR/$OUTPUT_DIR/"
echo "  - comprehensive_results.json"
echo "  - results_{regime}.json (intermediate results)"
echo "  - experiment_log.txt"
