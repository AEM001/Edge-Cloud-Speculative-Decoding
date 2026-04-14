# Model Download Guide

## Quick Start

### Option 1: Automated Download with Monitoring (Recommended)

```bash
cd /home/albert/code/AInfra/draft
./start_download.sh
```

This will:
- Start download in tmux session `model-download`
- Start monitor in tmux session `model-monitor`
- Auto-attach to monitor view

### Option 2: Manual Download

```bash
# Start download
/home/albert/learn/l-vllm/.venv/bin/python download_with_mirror.py

# In another terminal, start monitor
/home/albert/learn/l-vllm/.venv/bin/python download_monitor.py
```

## Files

- `download_with_mirror.py` - Main downloader with HF mirror support
- `download_monitor.py` - Real-time progress monitor
- `start_download.sh` - Convenience script to start both in tmux

## Download Details

- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Size**: ~3 GB
- **Location**: `~/models/Qwen--Qwen2.5-1.5B-Instruct/`
- **Mirrors**: hf-mirror.com (primary), huggingface.co (fallback)

## Features

- ✅ Resume capability (can restart if interrupted)
- ✅ Automatic mirror failover
- ✅ Real-time progress monitoring
- ✅ Speed and ETA calculation
- ✅ Stall detection

## Tmux Commands

```bash
# List sessions
tmux ls

# Attach to download
tmux attach -t model-download

# Attach to monitor
tmux attach -t model-monitor

# Detach from session
Ctrl+B, then D

# Kill sessions
tmux kill-session -t model-download
tmux kill-session -t model-monitor
```

## Testing After Download

Once download completes, test the model:

```bash
/home/albert/learn/l-vllm/.venv/bin/python main.py
```
