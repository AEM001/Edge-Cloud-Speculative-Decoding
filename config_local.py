"""Local configuration settings for speculative decoding with vLLM on AutoDL."""
import os
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


# Model configuration - Qwen2.5 1.5B as draft model
MODEL_NAME = os.getenv("AINFRA_DRAFT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL_PATH = Path(
    os.getenv(
        "AINFRA_DRAFT_MODEL_PATH",
        "/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-1.5B-Instruct",
    )
)

# vLLM settings - optimized for 7B AWQ + 1.5B on same GPU
GPU_MEMORY_UTILIZATION = _env_float("AINFRA_GPU_MEMORY_UTILIZATION", 0.30)
MAX_MODEL_LEN = _env_int("AINFRA_MAX_MODEL_LEN", 4096)
TENSOR_PARALLEL_SIZE = _env_int("AINFRA_TENSOR_PARALLEL_SIZE", 1)

# Server configuration - Local only (both draft and verify on same machine)
# Edge client port (local)
EDGE_PORT = 6006
# Cloud server port (remote) - now localhost since both are on same machine
CLOUD_PORT = 6006
# Cloud server address - localhost for local execution
CLOUD_SERVER = "localhost"
CLOUD_SSH_PORT = None  # No SSH tunnel needed for local execution
CLOUD_URL = f"http://{CLOUD_SERVER}:{CLOUD_PORT}"

# Generation settings
MAX_DRAFT_TOKENS = _env_int("AINFRA_MAX_DRAFT_TOKENS", 5)
TEMPERATURE = _env_float("AINFRA_TEMPERATURE", 0.0)  # Greedy for benchmarking
TOP_P = _env_float("AINFRA_TOP_P", 0.95)
MAX_NEW_TOKENS = _env_int("AINFRA_MAX_NEW_TOKENS", 128)

# Device settings
DEVICE = "cuda"

# Logging
VERBOSE = True
