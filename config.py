"""Configuration settings for speculative decoding with vLLM."""
import os
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


# Model configuration - defaults to Qwen2.5 3B in HuggingFace format.
MODEL_NAME = os.getenv("AINFRA_DRAFT_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
MODEL_PATH = Path(
    os.getenv(
        "AINFRA_DRAFT_MODEL_PATH",
        str(Path.home() / "models" / "Qwen--Qwen2.5-3B-Instruct"),
    )
)

# vLLM settings
GPU_MEMORY_UTILIZATION = _env_float("AINFRA_GPU_MEMORY_UTILIZATION", 0.7)
MAX_MODEL_LEN = _env_int("AINFRA_MAX_MODEL_LEN", 8192)
TENSOR_PARALLEL_SIZE = _env_int("AINFRA_TENSOR_PARALLEL_SIZE", 1)

# Server configuration
# Edge client port (local)
EDGE_PORT = 6006
# Cloud server port (remote)
CLOUD_PORT = 6008
# Cloud server address
CLOUD_SERVER = "connect.westd.seetacloud.com"
CLOUD_SSH_PORT = 20514
CLOUD_URL = f"http://{CLOUD_SERVER}:{CLOUD_PORT}"

# Generation settings
MAX_DRAFT_TOKENS = _env_int("AINFRA_MAX_DRAFT_TOKENS", 5)
TEMPERATURE = _env_float("AINFRA_TEMPERATURE", 0.8)
TOP_P = _env_float("AINFRA_TOP_P", 0.95)
MAX_NEW_TOKENS = _env_int("AINFRA_MAX_NEW_TOKENS", 128)

# Device settings
DEVICE = "cuda"  # PyTorch uses CUDA on NVIDIA GPUs

# Logging
VERBOSE = True
