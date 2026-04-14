#!/bin/bash
# Run greedy K=2/K=4 benchmark in tmux - outputs to outputs_run4/

set -e

SESSION_NAME="benchmark-run4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AINFRA_DRAFT_MODEL_NAME="${AINFRA_DRAFT_MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
export AINFRA_DRAFT_MODEL_PATH="${AINFRA_DRAFT_MODEL_PATH:-/home/albert/models/Qwen--Qwen2.5-1.5B-Instruct}"
export AINFRA_TEMPERATURE="${AINFRA_TEMPERATURE:-0.0}"
export AINFRA_MAX_MODEL_LEN="${AINFRA_MAX_MODEL_LEN:-8192}"

echo "=== Run 4 Benchmark Runner ==="
echo "Output: outputs_run4/"
echo ""

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists."
    echo "View: tmux attach-session -t $SESSION_NAME"
    exit 0
fi

echo "Creating new tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR"

tmux send-keys -t "$SESSION_NAME" "echo '=== Run 4: greedy K=2, K=4 ==='" C-m
tmux send-keys -t "$SESSION_NAME" "export AINFRA_DRAFT_MODEL_NAME='$AINFRA_DRAFT_MODEL_NAME'" C-m
tmux send-keys -t "$SESSION_NAME" "export AINFRA_DRAFT_MODEL_PATH='$AINFRA_DRAFT_MODEL_PATH'" C-m
tmux send-keys -t "$SESSION_NAME" "export AINFRA_TEMPERATURE='$AINFRA_TEMPERATURE'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Draft model: \$AINFRA_DRAFT_MODEL_PATH'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Temperature: \$AINFRA_TEMPERATURE'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Starting at: \$(date)'" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME" "source /home/albert/learn/l-vllm/.venv/bin/activate" C-m
tmux send-keys -t "$SESSION_NAME" "mkdir -p outputs_run4 && python comprehensive_benchmark_run4.py 2>&1 | tee outputs_run4/comprehensive_log.txt" C-m

echo "Experiment started in tmux session '$SESSION_NAME'"
echo "View live: tmux attach-session -t $SESSION_NAME"
echo "View log: tail -f $SCRIPT_DIR/outputs_run4/comprehensive_log.txt"
