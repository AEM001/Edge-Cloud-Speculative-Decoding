#!/bin/bash
# Start SSH tunnel with auto-reconnect and keepalive to prevent disconnection
# This creates a persistent tunnel to the remote verification server

set -e

SSH_PORT=20514
REMOTE_HOST="connect.westd.seetacloud.com"
REMOTE_USER="root"
LOCAL_PORT=6006
REMOTE_PORT=6006
SESSION_NAME="ssh-tunnel"

echo "=== SSH Tunnel Manager with Auto-Reconnect ==="
echo "Local port: $LOCAL_PORT -> Remote: $REMOTE_HOST:$REMOTE_PORT"
echo "SSH port: $SSH_PORT"
echo ""

# Function to check if tunnel is working
check_tunnel() {
    curl -s -m 5 http://localhost:$LOCAL_PORT/health > /dev/null 2>&1
    return $?
}

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists."
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach-session -t $SESSION_NAME"
    echo "  2. Kill and restart: tmux kill-session -t $SESSION_NAME && $0"
    echo ""
    
    # Check if tunnel is working
    if check_tunnel; then
        echo "✓ Tunnel appears to be working (health check passed)"
    else
        echo "✗ Tunnel may not be working (health check failed)"
        echo "  Consider restarting: tmux kill-session -t $SESSION_NAME && $0"
    fi
    exit 0
fi

echo "Creating new tmux session '$SESSION_NAME' with auto-reconnect SSH tunnel..."
echo ""

# Create tmux session and start SSH tunnel with auto-reconnect
tmux new-session -d -s $SESSION_NAME

# Send commands to tmux session
tmux send-keys -t $SESSION_NAME "echo '=== SSH Tunnel Auto-Reconnect Loop ==='" C-m
tmux send-keys -t $SESSION_NAME "echo 'Press Ctrl+C to stop'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# Create auto-reconnect loop
tmux send-keys -t $SESSION_NAME "while true; do" C-m
tmux send-keys -t $SESSION_NAME "  echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Starting SSH tunnel...\"" C-m
tmux send-keys -t $SESSION_NAME "  ssh -p $SSH_PORT \\" C-m
tmux send-keys -t $SESSION_NAME "      -L $LOCAL_PORT:localhost:$REMOTE_PORT \\" C-m
tmux send-keys -t $SESSION_NAME "      -o ServerAliveInterval=30 \\" C-m
tmux send-keys -t $SESSION_NAME "      -o ServerAliveCountMax=3 \\" C-m
tmux send-keys -t $SESSION_NAME "      -o ExitOnForwardFailure=yes \\" C-m
tmux send-keys -t $SESSION_NAME "      -o TCPKeepAlive=yes \\" C-m
tmux send-keys -t $SESSION_NAME "      $REMOTE_USER@$REMOTE_HOST -N" C-m
tmux send-keys -t $SESSION_NAME "  echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Tunnel disconnected. Reconnecting in 5 seconds...\"" C-m
tmux send-keys -t $SESSION_NAME "  sleep 5" C-m
tmux send-keys -t $SESSION_NAME "done" C-m

echo "✓ SSH tunnel started in tmux session '$SESSION_NAME'"
echo ""
echo "Waiting for tunnel to establish..."
sleep 3

# Check if tunnel is working
if check_tunnel; then
    echo "✓ Tunnel is working! Health check passed."
else
    echo "⚠ Tunnel may still be connecting. Check with: curl http://localhost:$LOCAL_PORT/health"
fi

echo ""
echo "Useful commands:"
echo "  - View tunnel status: tmux attach-session -t $SESSION_NAME"
echo "  - Detach from session: Ctrl+B, then D"
echo "  - Stop tunnel: tmux kill-session -t $SESSION_NAME"
echo "  - Check tunnel: curl http://localhost:$LOCAL_PORT/health"
echo "  - List all sessions: tmux list-sessions"
echo ""
echo "The tunnel will auto-reconnect if disconnected."
echo "It will continue running even if you close this terminal."
