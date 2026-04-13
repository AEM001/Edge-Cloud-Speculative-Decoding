"""Draft token generation with confidence statistics."""
from typing import List, Tuple
import mlx.core as mx
import numpy as np
from protocol import TokenInfo, DraftRequest, DraftResponse
import logging
from mlx_lm import generate

logger = logging.getLogger(__name__)


class DraftGenerator:
    """Generates draft tokens autoregressively with confidence stats."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = None  # KV cache for efficient generation
        
    def compute_confidence_stats(self, logits: mx.array) -> Tuple[float, float, float, float]:
        """
        Compute confidence statistics from logits.
        
        Returns:
            - probability of selected token
            - max probability in distribution
            - entropy of distribution
            - top-1 minus top-2 margin
        """
        # Convert logits to probabilities
        probs = mx.softmax(logits, axis=-1)
        probs_np = np.array(probs)
        
        # Get top 2 probabilities
        top_2_indices = np.argpartition(probs_np, -2)[-2:]
        top_2_probs = probs_np[top_2_indices]
        top_2_probs.sort()
        
        max_prob = np.max(probs_np)
        top_margin = top_2_probs[1] - top_2_probs[0]
        
        # Compute entropy: -sum(p * log(p))
        # Avoid log(0) by masking
        log_probs = np.log(probs_np + 1e-10)
        entropy = -np.sum(probs_np * log_probs)
        
        # Get probability of the sampled token
        token_prob = max_prob  # Will be updated with actual sampled token
        
        return float(token_prob), float(max_prob), float(entropy), float(top_margin)
    
    def generate_draft_tokens(
        self, 
        request: DraftRequest,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> DraftResponse:
        """
        Generate K draft tokens autoregressively with confidence stats.
        
        Args:
            request: DraftRequest with verified prefix and num tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            DraftResponse with token IDs and confidence stats
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        prefix = request.verified_prefix
        k = request.num_draft_tokens
        
        # Use mlx_lm.generate which uses KV cache internally
        # Convert prefix to text
        prompt_text = self.tokenizer.decode(prefix, skip_special_tokens=False)
        
        # Generate K tokens (mlx_lm.generate uses KV cache internally)
        generated_text = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt_text,
            max_tokens=k,
            verbose=False
        )
        
        # Extract generated tokens
        generated_only = generated_text[len(prompt_text):]
        generated_ids = self.tokenizer.encode(generated_only)[:k]
        
        if len(generated_ids) < k:
            logger.warning(f"Generated only {len(generated_ids)} tokens, expected {k}")
        
        # Get logprobs and stats via one forward pass
        full_sequence = prefix + generated_ids
        input_ids = mx.array([full_sequence])
        logits = self.model(input_ids)
        
        draft_token_ids = []
        draft_logprobs = []
        draft_probs = []
        max_probs = []
        entropies = []
        top_margins = []
        
        prefix_len = len(prefix)
        for i, token_id in enumerate(generated_ids):
            pos = prefix_len + i
            token_logits = logits[0, pos - 1, :]
            
            probs = mx.softmax(token_logits, axis=-1)
            probs_np = np.array(probs)
            
            token_prob = float(probs[token_id].item())
            token_logprob = float(mx.log(probs[token_id] + 1e-10).item())
            max_prob = float(np.max(probs_np))
            
            log_probs = np.log(probs_np + 1e-10)
            entropy = float(-np.sum(probs_np * log_probs))
            
            top_2_indices = np.argpartition(probs_np, -2)[-2:]
            top_2_probs = probs_np[top_2_indices]
            top_2_probs.sort()
            top_margin = float(top_2_probs[1] - top_2_probs[0]) if len(top_2_probs) == 2 else 0.0
            
            draft_token_ids.append(token_id)
            draft_logprobs.append(token_logprob)
            draft_probs.append(token_prob)
            max_probs.append(max_prob)
            entropies.append(entropy)
            top_margins.append(top_margin)
        
        return DraftResponse(
            draft_token_ids=draft_token_ids,
            logprobs=draft_logprobs,
            probabilities=draft_probs,
            confidence_stats={
                "max_probs": max_probs,
                "entropies": entropies,
                "top_margins": top_margins
            }
        )
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)
