# Draft Edge (Mac MLX)

This folder contains the **edge/draft-side** code for the SpecExtend system.
Copy this entire folder to your Mac that will run the small draft model.

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

## You Must Implement

The current `scripts/core/qwen_specextend_backend.py` is the **CUDA/Transformers** backend.
On Mac you need to replace it with an **MLX draft backend** that implements:

```python
from core.specextend_backend import SpecExtendDraftBackend

class MLXQwenDraftBackend(SpecExtendDraftBackend):
    tokenizer: object

    def build_draft_tree(self, verified_prefix, correction_token_id, nodes, threshold, max_depth, retrieval_token_indices=None):
        ...
```

Key libraries to use on Mac:
- `mlx` and `mlx-lm` for Qwen3 0.6B AWQ inference
- Keep `core/protocol.py`, `core/specextend_retrieval.py`, `client/http_cloud_client.py`, and `client/specextend_edge_client.py` unchanged.

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
| `scripts/core/specextend_retrieval.py` | Edge-side retrieval chunk selection |
| `scripts/experiments/quick_test.py` | End-to-end quick test runner |
| `scripts/experiments/network_conditions.py` | Network throttling simulation |
| `scripts/experiments/prompt_loader.py` | Prompt dataset loader |
| `quick.sh` | Quick test launcher |
