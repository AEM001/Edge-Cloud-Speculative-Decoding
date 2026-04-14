#!/bin/bash
# Start model download with monitoring in tmux

set -e

PYTHON_BIN="/home/albert/learn/l-vllm/.venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_with_mirror.py"
MONITOR_SCRIPT="$SCRIPT_DIR/download_monitor.py"

echo "🤖 Qwen2.5-1.5B Model Download Setup"
echo "===================================="
echo "Python: $PYTHON_BIN"
echo "Download script: $DOWNLOAD_SCRIPT"
echo "Monitor script: $MONITOR_SCRIPT"
echo ""

# Check if python exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ Python not found at: $PYTHON_BIN"
    echo "Please update PYTHON_BIN in this script"
    exit 1
fi

# Check if scripts exist
if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "❌ Download script not found: $DOWNLOAD_SCRIPT"
    exit 1
fi

if [ ! -f "$MONITOR_SCRIPT" ]; then
    echo "❌ Monitor script not found: $MONITOR_SCRIPT"
    exit 1
fi

# Make scripts executable
chmod +x "$DOWNLOAD_SCRIPT" "$MONITOR_SCRIPT"

# Kill existing sessions if they exist
tmux kill-session -t model-download 2>/dev/null || true
tmux kill-session -t model-monitor 2>/dev/null || true

echo "🚀 Starting download in tmux session 'model-download'..."
tmux new-session -d -s model-download "$PYTHON_BIN $DOWNLOAD_SCRIPT"

echo "📊 Starting monitor in tmux session 'model-monitor'..."
tmux new-session -d -s model-monitor "$PYTHON_BIN $MONITOR_SCRIPT"

echo ""
echo "✅ Sessions started!"
echo ""
echo "📥 To view download progress:"
echo "   tmux attach -t model-download"
echo ""
echo "📊 To view monitor (recommended):"
echo "   tmux attach -t model-monitor"
echo ""
echo "💡 To detach from tmux: Press Ctrl+B, then D"
echo "💡 To list sessions: tmux ls"
echo "💡 To kill sessions: tmux kill-session -t model-download"
echo ""
echo "🔄 Attaching to monitor in 2 seconds..."
sleep 2
tmux attach -t model-monitor
