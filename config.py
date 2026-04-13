"""Configuration settings for speculative decoding on Ubuntu 3060."""
from pathlib import Path

# Model configuration - Qwen2.5 3B quantized version
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_PATH = Path.home() / "models" / "Qwen2.5-3B-Instruct"

# Server configuration
# Edge client port (local)
EDGE_PORT = 6006
# Cloud server port (remote)
CLOUD_PORT = 6008
# Cloud server address
CLOUD_SERVER = "connect.westd.seetacloud.com"
CLOUD_SSH_PORT = 12272
CLOUD_URL = f"http://{CLOUD_SERVER}:{CLOUD_PORT}"

# Generation settings
MAX_DRAFT_TOKENS = 5
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 128

# Device settings
DEVICE = "cuda"  # PyTorch uses CUDA on NVIDIA GPUs

# Logging
VERBOSE = True
