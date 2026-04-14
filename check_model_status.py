#!/usr/bin/env python3
"""Check if the 1.5B draft model is ready for the experiment."""
import os
import sys
import time
from pathlib import Path

MODEL_PATH = Path("/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-1.5B-Instruct")
EXPECTED_SIZE_GB = 3.0  # Approximate expected size in GB
REQUIRED_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]


def check_model_exists():
    """Check if model directory exists and has required files."""
    if not MODEL_PATH.exists():
        print(f"❌ Model directory does not exist: {MODEL_PATH}")
        return False
    
    print(f"✓ Model directory exists: {MODEL_PATH}")
    
    # Check for required files
    missing = []
    for file in REQUIRED_FILES:
        file_path = MODEL_PATH / file
        if not file_path.exists():
            # Check for sharded model files
            if file == "model.safetensors":
                shard_files = list(MODEL_PATH.glob("model*.safetensors*"))
                if not shard_files:
                    missing.append(file)
            else:
                missing.append(file)
    
    if missing:
        print(f"⚠ Missing files: {missing}")
    else:
        print("✓ All required files present")
    
    # Calculate total size
    total_bytes = 0
    for f in MODEL_PATH.rglob("*"):
        if f.is_file():
            total_bytes += f.stat().st_size
    
    total_gb = total_bytes / (1024**3)
    print(f"📦 Current model size: {total_gb:.2f} GB")
    
    if total_gb < EXPECTED_SIZE_GB * 0.8:
        print(f"⚠ Model size seems incomplete (expected ~{EXPECTED_SIZE_GB} GB)")
        return False
    
    print(f"✓ Model size looks good (expected ~{EXPECTED_SIZE_GB} GB)")
    return len(missing) == 0


def wait_for_model(timeout_seconds=3600, check_interval=30):
    """Wait for model to be fully uploaded/downloaded."""
    print(f"Waiting for model at: {MODEL_PATH}")
    print(f"Timeout: {timeout_seconds}s, Check interval: {check_interval}s")
    print("-" * 60)
    
    start_time = time.time()
    last_size = 0
    stable_count = 0
    
    while time.time() - start_time < timeout_seconds:
        if MODEL_PATH.exists():
            # Calculate current size
            total_bytes = sum(f.stat().st_size for f in MODEL_PATH.rglob("*") if f.is_file())
            total_gb = total_bytes / (1024**3)
            
            if total_bytes > last_size:
                print(f"[{time.strftime('%H:%M:%S')}] Model downloading... {total_gb:.2f} GB")
                last_size = total_bytes
                stable_count = 0
            else:
                stable_count += 1
                if stable_count >= 2:  # Size stable for 2 checks
                    if check_model_exists():
                        print("-" * 60)
                        print("✅ Model is ready!")
                        return True
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting for model directory to be created...")
        
        time.sleep(check_interval)
    
    print("-" * 60)
    print("❌ Timeout waiting for model")
    return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--wait":
        success = wait_for_model()
        sys.exit(0 if success else 1)
    else:
        if check_model_exists():
            print("-" * 60)
            print("✅ Model is ready to use!")
            sys.exit(0)
        else:
            print("-" * 60)
            print("❌ Model is not ready")
            print("Run with --wait to wait for the model")
            sys.exit(1)
