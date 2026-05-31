"""Compatibility import for the Mac draft backend.

The Ubuntu/cloud verifier owns the Transformers target backend. The Mac edge
runtime uses MLX, so old imports of ``QwenSpecExtendDraftBackend`` resolve to
the MLX implementation here.
"""

from core.mlx_qwen_backend import MLXBackendConfig as QwenBackendConfig
from core.mlx_qwen_backend import MLXQwenDraftBackend as QwenSpecExtendDraftBackend

__all__ = ["QwenBackendConfig", "QwenSpecExtendDraftBackend"]
