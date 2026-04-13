"""Model loading and management with custom storage path for PyTorch."""
from pathlib import Path
from typing import Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the draft model with custom storage location using PyTorch."""
    
    def __init__(self, model_name: str, model_path: Path, quantization: bool = True):
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.quantization = quantization
        
    def setup_storage(self) -> None:
        """Ensure model storage directory exists."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model storage path: {self.model_path}")
    
    def load(self) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """Load model from local custom path, avoiding HuggingFace cache."""
        self.setup_storage()
        
        try:
            logger.info(f"Loading model from local path: {self.model_path}")
            
            # Configure quantization if enabled
            quantization_config = None
            if self.quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load tokenizer from local path
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model from local path
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )
            
            self.model.eval()
            
            device = next(self.model.parameters()).device
            logger.info(f"Model loaded successfully from {self.model_path} on {device}")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
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
            vocab_size = getattr(self.tokenizer, 'vocab_size', 0)
        
        device = next(self.model.parameters()).device
        
        return {
            "name": self.model_name,
            "storage_path": str(self.model_path),
            "vocab_size": vocab_size,
            "device": str(device),
            "quantization": self.quantization,
        }
