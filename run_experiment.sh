#!/usr/bin/env bash
# Run an experiment script in a tmux session (survives SSH disconnect).
# Automatically uses GPU 1 for the draft model (GPU 0 is for verify server).
#
# Usage:
#   bash scripts/run_experiment.sh [experiment_script] [--verify-port 6006]
#
# Examples:
#   bash scripts/run_experiment.sh                         # run Run-3 experiment
#   bash scripts/run_experiment.sh experiments/experiment_async_pipeline.py
#   bash scripts/run_experiment.sh quick_test.py

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${1:-experiments/experiment_async_pipeline_run3.py}"
VERIFY_PORT="${VERIFY_PORT:-6006}"
SESSION="picospec-exp"
LOG="/tmp/experiment.log"

echo "=== PicoSpec Experiment Runner ==="
echo "Script  : $SCRIPT"
echo "Repo    : $REPO_DIR"
echo "Draft GPU: 1  |  Verify server: localhost:$VERIFY_PORT"
echo ""

# Sanity check: verify server must be up
if ! curl -sf "http://localhost:$VERIFY_PORT/health" > /dev/null 2>&1; then
  echo "ERROR: Verify server not reachable at localhost:$VERIFY_PORT"
  echo "Start it first with: bash scripts/start_verify.sh"
  exit 1
fi
echo "Verify server: OK"
echo ""

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" \
  "cd '$REPO_DIR' && \
   CUDA_VISIBLE_DEVICES=1 \
   PYTHONPATH='$REPO_DIR' \
   python3 '$SCRIPT' \
   2>&1 | tee '$LOG'; \
   echo ''; echo '=== EXPERIMENT DONE ==='; exec bash"

echo "Experiment running in tmux session: $SESSION"
echo ""
echo "Commands:"
echo "  Watch logs: tail -f $LOG"
echo "  Attach    : tmux attach -t $SESSION"
echo "  Detach    : Ctrl-B then D"
