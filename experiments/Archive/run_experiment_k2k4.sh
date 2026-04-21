#!/bin/bash
# Run K=6, K=8, K=10 benchmark in tmux - outputs to outputs_run3/

set -e

SESSION_NAME="benchmark-k2k4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== K=6, K=8, K=10 Benchmark Runner (Run 3) ==="
echo "Output: outputs_run3/"
echo ""

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists."
    echo "View: tmux attach-session -t $SESSION_NAME"
    exit 0
fi

echo "Creating new tmux session '$SESSION_NAME'..."

# Create tmux session
tmux new-session -d -s $SESSION_NAME -c "$SCRIPT_DIR"

# Setup and run
tmux send-keys -t $SESSION_NAME "echo '=== K=6, K=8, K=10 Benchmark (Run 3) ==='" C-m
tmux send-keys -t $SESSION_NAME "echo 'Output: outputs_run3/'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting at: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# Activate virtual environment
tmux send-keys -t $SESSION_NAME "source /home/albert/learn/l-vllm/.venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "echo 'Virtual environment activated'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# Run the experiment
tmux send-keys -t $SESSION_NAME "mkdir -p outputs_run3 && python comprehensive_benchmark_k2k4.py 2>&1 | tee outputs_run3/comprehensive_log.txt" C-m

echo "✓ Experiment started in tmux session '$SESSION_NAME'"
echo ""
echo "Commands:"
echo "  - View live: tmux attach-session -t $SESSION_NAME"
echo "  - Detach: Ctrl+B, then D"
echo "  - Check: pgrep -f comprehensive_benchmark_k2k4.py"
echo "  - View log: tail -f $SCRIPT_DIR/outputs_run3/comprehensive_log.txt"
echo ""
echo "Results will be saved to outputs_run3/"
