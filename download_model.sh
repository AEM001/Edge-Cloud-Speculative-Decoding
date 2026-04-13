#!/bin/bash
# Download Qwen2.5 3B model to local directory (avoiding cache)

set -e

MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
LOCAL_DIR="$HOME/models/Qwen2.5-3B-Instruct"
SESSION_NAME="model-download"

echo "=== Downloading Qwen2.5 3B model to local directory ==="
echo "Model: $MODEL_NAME"
echo "Local directory: $LOCAL_DIR"

# Create local directory
mkdir -p "$LOCAL_DIR"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Installing huggingface-hub..."
    pip install huggingface-hub
fi

# Create tmux session for download
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists. Attaching..."
    tmux attach-session -t $SESSION_NAME
else
    echo "Creating new tmux session '$SESSION_NAME' for model download..."
    tmux new-session -d -s $SESSION_NAME
    
    # Run download in tmux
    tmux send-keys -t $SESSION_NAME "echo '=== Starting model download ==='" C-m
    tmux send-keys -t $SESSION_NAME "huggingface-cli download $MODEL_NAME --local-dir $LOCAL_DIR --local-dir-use-symlinks False" C-m
    
    tmux send-keys -t $SESSION_NAME "echo '=== Download complete! ==='" C-m
    tmux send-keys -t $SESSION_NAME "ls -lh $LOCAL_DIR" C-m
    
    echo "Model download started in tmux session '$SESSION_NAME'."
    echo "Run 'tmux attach-session -t $SESSION_NAME' to view progress."
    echo "The download will continue even if you disconnect from SSH."
fi
