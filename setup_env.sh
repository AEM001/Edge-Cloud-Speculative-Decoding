#!/bin/bash
# Environment setup script with tmux to prevent process termination on SSH disconnect

set -e

PROJECT_DIR="/home/albert/code/AInfra/mac-draft"
SESSION_NAME="speculative-decoding"

echo "=== Setting up environment for Speculative Decoding Edge Node ==="

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux not found. Installing tmux..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create a new tmux session or attach to existing one
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists. Attaching..."
    tmux attach-session -t $SESSION_NAME
else
    echo "Creating new tmux session '$SESSION_NAME'..."
    tmux new-session -d -s $SESSION_NAME -c $PROJECT_DIR
    
    # Run setup commands in tmux
    tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESSION_NAME "echo '=== Step 1: Creating virtual environment ==='" C-m
    tmux send-keys -t $SESSION_NAME "uv venv" C-m
    tmux send-keys -t $SESSION_NAME "source .venv/bin/activate" C-m
    
    tmux send-keys -t $SESSION_NAME "echo '=== Step 2: Installing dependencies ==='" C-m
    tmux send-keys -t $SESSION_NAME "uv pip install -r requirements.txt" C-m
    
    tmux send-keys -t $SESSION_NAME "echo '=== Step 3: Verifying installation ==='" C-m
    tmux send-keys -t $SESSION_NAME "python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'" C-m
    
    tmux send-keys -t $SESSION_NAME "echo '=== Environment setup complete! ==='" C-m
    tmux send-keys -t $SESSION_NAME "echo 'To download the model, run: ./download_model.sh'" C-m
    tmux send-keys -t $SESSION_NAME "echo 'To start the service, run: ./start_service.sh'" C-m
    
    echo "Tmux session created. Run 'tmux attach-session -t $SESSION_NAME' to view progress."
    echo "Or run 'tmux list-sessions' to see all sessions."
fi
