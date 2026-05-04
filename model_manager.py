"""Model loading and management with vLLM for HuggingFace models."""
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VLLMModelManager:
    """Manages the draft model using vLLM with GGUF support."""

    def __init__(
        self,
        model_path: Path,
        gpu_memory_utilization: float = 0.6,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
        gpu_id: int = 0
    ):
        """
        Initialize vLLM model manager.

        Args:
            model_path: Path to HuggingFace model directory
            gpu_memory_utilization: GPU memory fraction to use
            max_model_len: Maximum sequence length
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_id: Which GPU to use (default 0)
        """
        self.model_path = Path(model_path)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_id = gpu_id

        self.llm = None
        self.tokenizer = None
        
    def load(self):
        """Load model using vLLM."""
        try:
            from vllm import LLM
            from transformers import AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "vLLM not installed. Run: pip install vllm"
            )

        logger.info(f"Loading vLLM model from: {self.model_path} on GPU {self.gpu_id}")

        # Load HuggingFace format model with vLLM
        self.llm = LLM(
            model=str(self.model_path),
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            enable_prefix_caching=True,
            disable_log_stats=True,
        )
        
        # Get tokenizer from vLLM
        self.tokenizer = self.llm.get_tokenizer()
        
        logger.info("vLLM model loaded successfully")
        return self.llm, self.tokenizer
    
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.llm is not None and self.tokenizer is not None
    
    def get_model_info(self) -> dict:
        """Get model information."""
        if not self.is_loaded():
            return {}
        
        vocab_size = 0
        if self.tokenizer:
            vocab_size = getattr(self.tokenizer, 'vocab_size', 0)
        
        return {
            "model_path": str(self.model_path),
            "vocab_size": vocab_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
