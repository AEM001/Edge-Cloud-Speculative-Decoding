"""Draft token generation with vLLM for GGUF models."""
from typing import List, Tuple
import numpy as np
import logging

from protocol import TokenInfo, DraftRequest, DraftResponse

logger = logging.getLogger(__name__)


class VLLMDraftGenerator:
    """Generates draft tokens using vLLM with confidence statistics."""
    
    def __init__(self, llm, tokenizer):
        """
        Initialize vLLM draft generator.
        
        Args:
            llm: vLLM LLM instance
            tokenizer: Tokenizer instance
        """
        self.llm = llm
        self.tokenizer = tokenizer
        
    def compute_confidence_stats(self, logprobs_dict: dict) -> Tuple[float, float, float]:
        """
        Compute confidence statistics from vLLM logprobs.
        
        Args:
            logprobs_dict: Dict mapping token_id to logprob (from vLLM)
        
        Returns:
            Tuple of (max_prob, entropy, top_margin)
        """
        # Convert logprobs to probabilities
        probs = {}
        for token_id, logprob_obj in logprobs_dict.items():
            # vLLM returns LogProb objects with .logprob attribute
            if hasattr(logprob_obj, 'logprob'):
                logprob = logprob_obj.logprob
            else:
                logprob = logprob_obj
            probs[token_id] = np.exp(logprob)
        
        # Normalize probabilities
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v / total_prob for k, v in probs.items()}
        
        probs_array = np.array(list(probs.values()))
        
        # Max probability
        max_prob = float(np.max(probs_array))
        
        # Entropy
        log_probs = np.log(probs_array + 1e-10)
        entropy = float(-np.sum(probs_array * log_probs))
        
        # Top margin (top-1 minus top-2)
        sorted_probs = np.sort(probs_array)[::-1]
        top_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else 0.0
        
        return max_prob, entropy, top_margin
    
    def generate_draft_tokens(
        self, 
        request: DraftRequest,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> DraftResponse:
        """
        Generate K draft tokens using vLLM with confidence stats.
        
        Args:
            request: DraftRequest with verified prefix and num tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            DraftResponse with token IDs and confidence stats
        """
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        from vllm import SamplingParams
        
        prefix = request.verified_prefix
        k = request.num_draft_tokens
        
        # Decode prefix to text
        prompt_text = self.tokenizer.decode(prefix, skip_special_tokens=False)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            max_tokens=k,
            logprobs=5,  # Get top 5 logprobs for each position
            prompt_logprobs=k,  # Get logprobs for draft positions
        )
        
        # Generate with vLLM
        outputs = self.llm.generate(
            prompts=[prompt_text],
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        output = outputs[0]
        
        # Extract generated tokens
        if output.outputs:
            generated_ids = output.outputs[0].token_ids[:k]
        else:
            generated_ids = []
        
        if len(generated_ids) < k:
            logger.warning(f"Generated only {len(generated_ids)} tokens, expected {k}")
        
        # Get prompt logprobs for confidence stats
        prompt_logprobs = output.prompt_logprobs or []
        
        # Build response with confidence stats
        draft_token_ids = []
        draft_logprobs = []
        draft_probs = []
        max_probs = []
        entropies = []
        top_margins = []
        
        prefix_len = len(prefix)
        
        for i, token_id in enumerate(generated_ids):
            # Position in the full sequence (after prefix)
            pos = prefix_len + i
            
            # Get logprobs at this position
            if pos < len(prompt_logprobs) and prompt_logprobs[pos] is not None:
                logprobs_dict = prompt_logprobs[pos]
                
                # Get probability of the selected token
                token_logprob = logprobs_dict.get(token_id, None)
                if token_logprob is not None:
                    if hasattr(token_logprob, 'logprob'):
                        token_prob = np.exp(token_logprob.logprob)
                        token_logprob_val = token_logprob.logprob
                    else:
                        token_prob = np.exp(token_logprob)
                        token_logprob_val = token_logprob
                else:
                    token_prob = 0.0
                    token_logprob_val = -float('inf')
                
                # Compute confidence stats
                max_prob, entropy, top_margin = self.compute_confidence_stats(logprobs_dict)
            else:
                # Fallback if no logprobs available
                token_prob = 1.0 / self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 0.0
                token_logprob_val = np.log(token_prob + 1e-10)
                max_prob = token_prob
                entropy = np.log(self.tokenizer.vocab_size) if hasattr(self.tokenizer, 'vocab_size') else 0.0
                top_margin = 0.0
            
            draft_token_ids.append(token_id)
            draft_logprobs.append(float(token_logprob_val))
            draft_probs.append(float(token_prob))
            max_probs.append(float(max_prob))
            entropies.append(float(entropy))
            top_margins.append(float(top_margin))
        
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
