# Draft Edge (Mac MLX)

This folder contains the **edge/draft-side** code for the SpecExtend system.
Run this folder on the Mac that hosts the small draft model.

## Target Mac Setup

- **Device**: Mac with Apple Silicon (MLX required)
- **Draft Model**: `Qwen/Qwen3-0.6B-AWQ` loaded via MLX
- **Cloud Target**: This machine connects to the cloud verifier over the phone hotspot local network.

## Network Setup (Phone Hotspot)

1. Enable phone hotspot.
2. Connect both this Mac **and** the cloud GPU machine to the same hotspot.
3. Find the cloud GPU machine's local IP (e.g., `192.168.43.100`).
4. Set `VERIFY_SERVER_URL` on the Mac to `http://<cloud-ip>:6007`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt
```

## Draft Backend

`scripts/core/mlx_qwen_backend.py` is the Mac draft backend. It implements
`SpecExtendDraftBackend` with `mlx` and `mlx-lm`, keeps the selected sparse
context plus recent tail, and builds the draft tree sent to the Ubuntu verifier.

`scripts/core/qwen_specextend_backend.py` is a compatibility shim for older
imports and points to the MLX backend. The Ubuntu/cloud target backend is not
part of this Mac folder.

## Running a Quick Test

```bash
export VERIFY_SERVER_URL="http://<cloud-ip>:6007"
export DRAFT_MODEL_PATH="/path/to/Qwen3-0.6B-AWQ"
bash quick.sh
```

## Files

| File | Purpose |
|------|---------|
| `scripts/client/http_cloud_client.py` | HTTP client talking to cloud verifier |
| `scripts/client/specextend_edge_client.py` | SpecExtend edge orchestration loop |
| `scripts/core/protocol.py` | Wire protocol dataclasses |
| `scripts/core/specextend_backend.py` | Backend protocols (`SpecExtendDraftBackend`) |
| `scripts/core/mlx_qwen_backend.py` | Mac MLX draft backend |
| `scripts/core/specextend_retrieval.py` | Edge-side retrieval chunk selection |
| `scripts/experiments/quick_test.py` | End-to-end quick test runner |
| `scripts/experiments/network_conditions.py` | Network throttling simulation |
| `scripts/experiments/prompt_loader.py` | Prompt dataset loader |
| `quick.sh` | Quick test launcher |
