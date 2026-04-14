"""Configuration settings for speculative decoding with vLLM."""
from pathlib import Path

# Model configuration - Qwen2.5 1.5B in HuggingFace format
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_PATH = Path.home() / "models" / "Qwen--Qwen2.5-1.5B-Instruct"

# vLLM settings
GPU_MEMORY_UTILIZATION = 0.6  # GPU memory fraction for draft model
MAX_MODEL_LEN = 32768  # Maximum sequence length
TENSOR_PARALLEL_SIZE = 1  # Number of GPUs for tensor parallelism

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
