"""Draft token generation with confidence statistics."""
from typing import List, Tuple
import mlx.core as mx
import numpy as np
from protocol import TokenInfo, DraftRequest, DraftResponse
import logging

logger = logging.getLogger(__name__)


class DraftGenerator:
    """Generates draft tokens autoregressively with confidence stats."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
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
        
        draft_tokens: List[TokenInfo] = []
        current_tokens = list(prefix)  # Working copy
        
        # Generate K tokens autoregressively
        for step in range(k):
            # Prepare input
            input_ids = mx.array([current_tokens])
            
            # Get logits from model
            logits = self.model(input_ids)
            next_token_logits = logits[0, -1, :]  # Last position logits
            
            # Sample next token with temperature
            if temperature > 0:
                scaled_logits = next_token_logits / temperature
                probs = mx.softmax(scaled_logits, axis=-1)
                
                # Simple top-p filtering
                sorted_probs = mx.sort(probs)[::-1]
                cumsum_probs = mx.cumsum(sorted_probs, axis=0)
                threshold = sorted_probs[mx.argmax(cumsum_probs > top_p)]
                probs = mx.where(probs < threshold, 0, probs)
                probs = probs / mx.sum(probs)  # Renormalize
                
                # Sample
                next_token = mx.random.categorical(mx.log(probs + 1e-10))
            else:
                next_token = mx.argmax(next_token_logits)
            
            next_token_id = int(next_token.item())
            
            # Compute confidence stats
            token_prob, max_prob, entropy, top_margin = self.compute_confidence_stats(next_token_logits)
            
            # Get actual probability of sampled token
            probs = mx.softmax(next_token_logits, axis=-1)
            actual_prob = float(probs[next_token_id].item())
            actual_logprob = float(mx.log(probs[next_token_id] + 1e-10).item())
            
            # Store token info
            token_info = TokenInfo(
                token_id=next_token_id,
                logprob=actual_logprob,
                probability=actual_prob,
                max_prob=max_prob,
                entropy=entropy,
                top_margin=top_margin
            )
            draft_tokens.append(token_info)
            
            # Append to current tokens for next iteration
            current_tokens.append(next_token_id)
            
            if logger.isEnabledFor(logging.DEBUG):
                token_text = self.tokenizer.decode([next_token_id])
                logger.debug(f"Step {step}: token={token_text!r}, prob={actual_prob:.4f}, entropy={entropy:.4f}")
        
        # Package response with lightweight transmission format
        return DraftResponse(
            draft_token_ids=[t.token_id for t in draft_tokens],
            logprobs=[t.logprob for t in draft_tokens],
            probabilities=[t.probability for t in draft_tokens],
            confidence_stats={
                "max_probs": [t.max_prob for t in draft_tokens],
                "entropies": [t.entropy for t in draft_tokens],
                "top_margins": [t.top_margin for t in draft_tokens]
            }
        )
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)
