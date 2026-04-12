"""Model loading and management with custom storage path."""
from pathlib import Path
from typing import Tuple, Optional
import mlx.core as mx
from mlx_lm import load, generate
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the draft model with custom storage location."""
    
    def __init__(self, model_name: str, model_path: Path):
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        
    def setup_storage(self) -> None:
        """Ensure model storage directory exists."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model storage path: {self.model_path}")
    
    def load(self) -> Tuple[Optional[object], Optional[object]]:
        """Load model to custom path, avoiding cache."""
        self.setup_storage()
        
        try:
            logger.info(f"Loading model {self.model_name}...")
            
            # Set environment to use custom path instead of cache
            import os
            os.environ["HF_HOME"] = str(self.model_path.parent)
            os.environ["TRANSFORMERS_CACHE"] = str(self.model_path)
            
            # Load with MLX - this downloads to the specified location
            self.model, self.tokenizer = load(self.model_name)
            
            logger.info("Model loaded successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> dict:
        """Get model information."""
        if not self.is_loaded():
            return {}
        
        vocab_size = 0
        if self.tokenizer:
            # TokenizerWrapper uses vocab_size property
            vocab_size = getattr(self.tokenizer, 'vocab_size', 0)
        
        return {
            "name": self.model_name,
            "storage_path": str(self.model_path),
            "vocab_size": vocab_size,
        }
