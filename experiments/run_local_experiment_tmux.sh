#!/bin/bash
# Run local experiment with network simulation in tmux (background mode)
# This allows the experiment to continue even if SSH disconnects

set -e

SESSION_NAME="local-experiment-k2k8"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAFT_DIR="$SCRIPT_DIR/.."
UBUNTU_VERIFY_DIR="/root/autodl-tmp/ubuntu-verify"

# Configuration
export AINFRA_DRAFT_MODEL_NAME="${AINFRA_DRAFT_MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
export AINFRA_DRAFT_MODEL_PATH="${AINFRA_DRAFT_MODEL_PATH:-/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-1.5B-Instruct}"
export AINFRA_TEMPERATURE=0.0
export AINFRA_MAX_MODEL_LEN=8192
export AINFRA_GPU_MEMORY_UTILIZATION=0.7

echo "=== Local Experiment Tmux Runner (k=2,4,6,8) ==="
echo "Session: $SESSION_NAME"
echo "Draft model: $AINFRA_DRAFT_MODEL_PATH"
echo "Verify model: /root/autodl-tmp/ubuntu-verify/models/Qwen2.5-32B-Instruct-GPTQ-Int4"
echo ""

# Check if tmux session exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists."
    echo "View: tmux attach-session -t $SESSION_NAME"
    echo "Kill: tmux kill-session -t $SESSION_NAME"
    exit 0
fi

# Create new tmux session
echo "Creating new tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" -c "$DRAFT_DIR"

# Send commands to tmux session
tmux send-keys -t "$SESSION_NAME" "echo '=== Local Experiment: k=2,4,6,8 with Network Simulation ==='" C-m
tmux send-keys -t "$SESSION_NAME" "export AINFRA_DRAFT_MODEL_NAME='$AINFRA_DRAFT_MODEL_NAME'" C-m
tmux send-keys -t "$SESSION_NAME" "export AINFRA_DRAFT_MODEL_PATH='$AINFRA_DRAFT_MODEL_PATH'" C-m
tmux send-keys -t "$SESSION_NAME" "export AINFRA_TEMPERATURE='$AINFRA_TEMPERATURE'" C-m
tmux send-keys -t "$SESSION_NAME" "export AINFRA_MAX_MODEL_LEN='$AINFRA_MAX_MODEL_LEN'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Draft model: '\$AINFRA_DRAFT_MODEL_PATH" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Starting at: '\$(date)" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m

# Check and activate virtual environment
tmux send-keys -t "$SESSION_NAME" "if [ -d '$UBUNTU_VERIFY_DIR/.venv' ]; then source '$UBUNTU_VERIFY_DIR/.venv/bin/activate'; fi" C-m

# Run the experiment
tmux send-keys -t "$SESSION_NAME" "cd '$DRAFT_DIR' && mkdir -p experiments/outputs_local_k2k8 && bash run_local_experiment.sh 2>&1 | tee experiments/outputs_local_k2k8/tmux_log.txt" C-m

echo "Experiment started in tmux session '$SESSION_NAME'"
echo ""
echo "Useful commands:"
echo "  View live:   tmux attach-session -t $SESSION_NAME"
echo "  View log:    tail -f $DRAFT_DIR/experiments/outputs_local_k2k8/tmux_log.txt"
echo "  Detach:      Ctrl+B, then D (when in tmux)"
echo "  Kill:        tmux kill-session -t $SESSION_NAME"
echo ""
echo "Note: The session will continue running even if you disconnect."
