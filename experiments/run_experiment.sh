#!/bin/bash
# Run comprehensive benchmark in tmux to prevent interruption on SSH disconnect

set -e

SESSION_NAME="benchmark-experiment"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Comprehensive Benchmark Runner ==="
echo "Running in tmux session to prevent SSH disconnection issues"
echo ""

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists."
    echo ""
    echo "Options:"
    echo "  1. View progress: tmux attach-session -t $SESSION_NAME"
    echo "  2. Kill and restart: tmux kill-session -t $SESSION_NAME && $0"
    echo ""
    echo "Checking if experiment is still running..."
    if pgrep -f comprehensive_benchmark.py > /dev/null; then
        echo "✓ Experiment is still running"
    else
        echo "✗ Experiment has finished or stopped"
        echo "  View results: cat $SCRIPT_DIR/comprehensive_log.txt"
    fi
    exit 0
fi

echo "Creating new tmux session '$SESSION_NAME'..."
echo ""

# Create tmux session and run experiment
tmux new-session -d -s $SESSION_NAME -c "$SCRIPT_DIR"

# Send commands to tmux session
tmux send-keys -t $SESSION_NAME "echo '=== Starting Comprehensive Benchmark ==='" C-m
tmux send-keys -t $SESSION_NAME "echo 'This will run 300 tests (100 easy + 100 hard prompts × 3 methods)'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Estimated time: 30-60 minutes depending on network and GPU'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting at: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# Activate virtual environment
tmux send-keys -t $SESSION_NAME "source /home/albert/learn/l-vllm/.venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "echo 'Virtual environment activated'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# Run the experiment
tmux send-keys -t $SESSION_NAME "python3 comprehensive_benchmark.py 2>&1 | tee comprehensive_log.txt" C-m

echo "✓ Experiment started in tmux session '$SESSION_NAME'"
echo ""
echo "Useful commands:"
echo "  - View live progress: tmux attach-session -t $SESSION_NAME"
echo "  - Detach from session: Ctrl+B, then D"
echo "  - Stop experiment: tmux kill-session -t $SESSION_NAME"
echo "  - Check if running: pgrep -f comprehensive_benchmark.py"
echo "  - View log: tail -f $SCRIPT_DIR/comprehensive_log.txt"
echo "  - List all sessions: tmux list-sessions"
echo ""
echo "The experiment will continue running even if you disconnect from SSH."
echo "Results will be saved to:"
echo "  - $SCRIPT_DIR/comprehensive_results.json"
echo "  - $SCRIPT_DIR/comprehensive_log.txt"
