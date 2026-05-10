"""
Central configuration for PicoSpec.

All values can be overridden via environment variables.
The defaults assume both models are stored under /root/code/.
"""
import os
from pathlib import Path


def _float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None else default


def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None else default


# ---------------------------------------------------------------------------
# Draft model  (Qwen2.5-1.5B-Instruct-AWQ — runs on the edge / GPU 1)
# ---------------------------------------------------------------------------
DRAFT_MODEL_NAME: str = os.getenv(
    "DRAFT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"
)
DRAFT_MODEL_PATH: Path = Path(
    os.getenv("DRAFT_MODEL_PATH", "/root/code/draft/models/Qwen2.5-1.5B-Instruct-AWQ")
)
DRAFT_GPU_MEM: float = _float("DRAFT_GPU_MEM", 0.25)
DRAFT_MAX_LEN: int = _int("DRAFT_MAX_LEN", 4096)
DRAFT_GPU_ID: int = _int("DRAFT_GPU_ID", 1)

# ---------------------------------------------------------------------------
# Verify model  (Qwen2.5-32B-Instruct-AWQ — runs on GPU 0 via verify server)
# ---------------------------------------------------------------------------
VERIFY_MODEL_PATH: Path = Path(
    os.getenv("VERIFY_MODEL_PATH", "/root/code/draft/models/Qwen2.5-32B-Instruct-AWQ")
)
VERIFY_GPU_MEM: float = _float("VERIFY_GPU_MEM", 0.55)
VERIFY_MAX_LEN: int = _int("VERIFY_MAX_LEN", 4096)
VERIFY_QUANTIZATION: str = os.getenv("VERIFY_QUANTIZATION", "awq")
VERIFY_TENSOR_PARALLEL_SIZE: int = _int("VERIFY_TENSOR_PARALLEL_SIZE", 2)

# ---------------------------------------------------------------------------
# Verify server  (HTTP endpoint)
# ---------------------------------------------------------------------------
VERIFY_SERVER_URL: str = os.getenv("VERIFY_SERVER_URL", "http://localhost:6006")

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS: int = _int("MAX_NEW_TOKENS", 128)
TEMPERATURE: float = _float("TEMPERATURE", 0.0)
TOP_P: float = _float("TOP_P", 0.95)
