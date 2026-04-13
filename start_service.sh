#!/bin/bash
# Start the edge service in tmux to prevent process termination on SSH disconnect

set -e

PROJECT_DIR="/home/albert/code/AInfra/mac-draft"
SESSION_NAME="edge-service"
PORT=6006

echo "=== Starting Speculative Decoding Edge Service ==="

# Check if model exists
MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please run ./download_model.sh first"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "Error: Virtual environment not found"
    echo "Please run ./setup_env.sh first"
    exit 1
fi

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists."
    echo "Attaching to existing session..."
    tmux attach-session -t $SESSION_NAME
else
    echo "Creating new tmux session '$SESSION_NAME'..."
    tmux new-session -d -s $SESSION_NAME -c $PROJECT_DIR
    
    # Activate virtual environment and start service
    tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESSION_NAME "source .venv/bin/activate" C-m
    
    tmux send-keys -t $SESSION_NAME "echo '=== Starting Edge Service on port $PORT ==='" C-m
    tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=0" C-m
    tmux send-keys -t $SESSION_NAME "python main.py" C-m
    
    echo "Edge service started in tmux session '$SESSION_NAME'."
    echo "Service running on port $PORT"
    echo ""
    echo "Useful commands:"
    echo "  - View service: tmux attach-session -t $SESSION_NAME"
    echo "  - Detach from session: Ctrl+B, then D"
    echo "  - Stop service: tmux kill-session -t $SESSION_NAME"
    echo "  - List all sessions: tmux list-sessions"
    echo ""
    echo "The service will continue running even if you disconnect from SSH."
fi
