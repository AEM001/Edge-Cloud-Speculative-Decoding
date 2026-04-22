#!/usr/bin/env bash
# Start the verify server (Qwen2.5-7B-AWQ) in a tmux session.
# The server occupies GPU 0 by default.
#
# Usage:
#   bash scripts/start_verify.sh [--port 6006]
#
# Stop with:
#   tmux kill-session -t verify-server

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT=${PORT:-6006}
SESSION="verify-server"

# Parse optional --port arg
while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== Starting verify server on port $PORT ==="
echo "Repo: $REPO_DIR"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" \
  "cd '$REPO_DIR' && \
   PYTHONPATH='$REPO_DIR' \
   python3 -m server.verify_server --host 0.0.0.0 --port $PORT \
   2>&1 | tee /tmp/verify_server.log; \
   echo ''; echo '=== SERVER STOPPED ==='; exec bash"

echo ""
echo "Verify server starting in tmux session: $SESSION"
echo ""
echo "Commands:"
echo "  Watch logs  : tail -f /tmp/verify_server.log"
echo "  Attach      : tmux attach -t $SESSION"
echo "  Health check: curl http://localhost:$PORT/health"
echo "  Stop        : tmux kill-session -t $SESSION"
echo ""
echo "Waiting for server to be healthy..."
for i in $(seq 1 60); do
  sleep 5
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "  Server ready after $((i*5))s"
    break
  fi
  echo "  Still loading... $((i*5))s"
done
