"""Configuration settings for speculative decoding."""
from pathlib import Path

# Model configuration
MODEL_NAME = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MODEL_PATH = Path.home() / ".models" / "mlx-draft-model"

# Generation settings
MAX_DRAFT_TOKENS = 5
TEMPERATURE = 0.8
TOP_P = 0.95

# Device settings
DEVICE = "metal"  # MLX uses Metal on Apple Silicon

# Logging
VERBOSE = True
