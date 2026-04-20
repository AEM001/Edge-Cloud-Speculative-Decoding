#!/usr/bin/env bash
# ============================================================
# Run the Async Pipeline Experiment in a tmux session
# Usage: bash experiments/run_experiment_async.sh
# ============================================================

set -euo pipefail

DRAFT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="async-experiment"
LOG_DIR="$DRAFT_DIR/experiments/outputs_async"
VENV="$DRAFT_DIR/../ubuntu-verify/.venv"

mkdir -p "$LOG_DIR"

# Activate the right python environment
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
    echo "Using venv: $VENV"
else
    echo "WARNING: venv not found at $VENV — using system python"
fi

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Starting experiment in tmux session: $SESSION"
echo "Draft dir: $DRAFT_DIR"
echo "Output dir: $LOG_DIR"
echo ""

# Verify the verify server is reachable
if ! curl -sf http://localhost:6006/health > /dev/null; then
    echo "ERROR: Cloud verify server not reachable at localhost:6006"
    echo "Start it first with: ./start_background.sh verify 6006"
    exit 1
fi
echo "Verify server: OK"

# Start experiment in tmux
tmux new-session -d -s "$SESSION" \
    "cd '$DRAFT_DIR' && \
     PYTHONPATH='$DRAFT_DIR' python experiments/experiment_async_pipeline.py \
     2>&1 | tee '$LOG_DIR/async_log.txt'; \
     echo ''; echo '=== DONE ==='; read -r"

echo ""
echo "Experiment running in tmux session: $SESSION"
echo ""
echo "Commands:"
echo "  Monitor:  tmux attach -t $SESSION"
echo "  Log:      tail -f $LOG_DIR/async_log.txt"
echo "  Results:  cat $LOG_DIR/async_results.json"
echo "  Detach:   Ctrl+B then D"
