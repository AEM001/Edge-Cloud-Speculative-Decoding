"""Draft token generation with confidence statistics using PyTorch."""
from typing import List, Tuple
import torch
import numpy as np
from protocol import TokenInfo, DraftRequest, DraftResponse
import logging

logger = logging.getLogger(__name__)


class DraftGenerator:
    """Generates draft tokens autoregressively with confidence stats using PyTorch."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def compute_confidence_stats(self, logits: torch.Tensor) -> Tuple[float, float, float, float]:
        """
        Compute confidence statistics from logits.
        
        Returns:
            - probability of selected token
            - max probability in distribution
            - entropy of distribution
            - top-1 minus top-2 margin
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        probs_np = probs.detach().cpu().numpy()
        
        # Get top 2 probabilities
        top_2_indices = np.argpartition(probs_np, -2)[-2:]
        top_2_probs = probs_np[top_2_indices]
        top_2_probs.sort()
        
        max_prob = np.max(probs_np)
        top_margin = top_2_probs[1] - top_2_probs[0] if len(top_2_probs) == 2 else 0.0
        
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
        Generate K draft tokens autoregressively with confidence stats using PyTorch.
        
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
        
        # Convert prefix to text
        prompt_text = self.tokenizer.decode(prefix, skip_special_tokens=False)
        
        # Generate K tokens using PyTorch
        input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=k,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Extract generated tokens
        full_sequence = outputs.sequences[0].cpu().tolist()
        generated_ids = full_sequence[len(prefix):]
        generated_ids = generated_ids[:k]
        
        if len(generated_ids) < k:
            logger.warning(f"Generated only {len(generated_ids)} tokens, expected {k}")
        
        # Get logprobs and stats from outputs.scores
        draft_token_ids = []
        draft_logprobs = []
        draft_probs = []
        max_probs = []
        entropies = []
        top_margins = []
        
        for i, token_id in enumerate(generated_ids):
            if i < len(outputs.scores):
                logits = outputs.scores[i][0]  # [vocab_size]
                
                probs = torch.softmax(logits, dim=-1)
                probs_np = probs.detach().cpu().numpy()
                
                token_prob = float(probs[token_id].item())
                token_logprob = float(torch.log(probs[token_id] + 1e-10).item())
                max_prob = float(probs_np.max())
                
                log_probs = np.log(probs_np + 1e-10)
                entropy = float(-(probs_np * log_probs).sum())
                
                top_2_indices = np.argpartition(probs_np, -2)[-2:]
                top_2_probs = np.sort(probs_np[top_2_indices])
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
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
