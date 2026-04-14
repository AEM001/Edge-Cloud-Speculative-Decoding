#!/usr/bin/env python3
"""Download Qwen2.5-1.5B-Instruct in HuggingFace format for vLLM."""
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_DIR = Path.home() / "models" / "Qwen--Qwen2.5-1.5B-Instruct"

print(f"Downloading {MODEL_NAME} to {LOCAL_DIR}...")
print("This will take 5-10 minutes (~3GB)")

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=str(LOCAL_DIR),
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"\n✓ Model downloaded to: {LOCAL_DIR}")
print(f"Size: {sum(f.stat().st_size for f in LOCAL_DIR.rglob('*') if f.is_file()) / 1e9:.2f} GB")
