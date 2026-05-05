"""Draft token generation with vLLM for GGUF models."""
from typing import List, Tuple
import numpy as np
import logging

from protocol import DraftRequest, DraftResponse

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

        # Set up sampling parameters — no prompt_logprobs to avoid
        # re-computing the full prefix at every draft step
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            max_tokens=k,
            logprobs=1,
        )

        # Pass token IDs directly to avoid decode → re-tokenize round-trip
        from vllm import TokensPrompt
        outputs = self.llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=list(prefix))],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        
        output = outputs[0]
        
        # Extract generated tokens
        if output.outputs:
            generated_ids = output.outputs[0].token_ids[:k]
        else:
            generated_ids = []
        
        if len(generated_ids) < k:
            logger.warning(f"Generated only {len(generated_ids)} tokens, expected {k}")

        # Use per-generated-token logprobs (output.outputs[0].logprobs)
        out_logprobs = output.outputs[0].logprobs or []

        # Build response with confidence stats
        draft_token_ids = []
        draft_logprobs = []
        draft_probs = []
        max_probs = []
        entropies = []
        top_margins = []

        for i, token_id in enumerate(generated_ids):
            if i < len(out_logprobs) and out_logprobs[i] is not None:
                logprobs_dict = out_logprobs[i]

                token_logprob_obj = logprobs_dict.get(token_id, None)
                if token_logprob_obj is not None:
                    token_logprob_val = token_logprob_obj.logprob if hasattr(token_logprob_obj, 'logprob') else float(token_logprob_obj)
                    token_prob = np.exp(token_logprob_val)
                else:
                    token_logprob_val = -float('inf')
                    token_prob = 0.0

                max_prob, entropy, top_margin = self.compute_confidence_stats(logprobs_dict)
            else:
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

    def generate_draft_tokens_batch(
        self,
        requests: List[DraftRequest],
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[DraftResponse]:
        if not requests:
            return []
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        from vllm import SamplingParams, TokensPrompt

        k = requests[0].num_draft_tokens
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            max_tokens=k,
            logprobs=1,
        )
        outputs = self.llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=list(req.verified_prefix)) for req in requests],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        responses: List[DraftResponse] = []
        for output, req in zip(outputs, requests):
            generated_ids = output.outputs[0].token_ids[:req.num_draft_tokens] if output.outputs else []
            out_logprobs = output.outputs[0].logprobs or [] if output.outputs else []
            draft_token_ids = []
            draft_logprobs = []
            draft_probs = []
            max_probs = []
            entropies = []
            top_margins = []

            for i, token_id in enumerate(generated_ids):
                if i < len(out_logprobs) and out_logprobs[i] is not None:
                    logprobs_dict = out_logprobs[i]
                    token_logprob_obj = logprobs_dict.get(token_id, None)
                    if token_logprob_obj is not None:
                        token_logprob_val = token_logprob_obj.logprob if hasattr(token_logprob_obj, 'logprob') else float(token_logprob_obj)
                        token_prob = np.exp(token_logprob_val)
                    else:
                        token_logprob_val = -float('inf')
                        token_prob = 0.0
                    max_prob, entropy, top_margin = self.compute_confidence_stats(logprobs_dict)
                else:
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

            responses.append(DraftResponse(
                draft_token_ids=draft_token_ids,
                logprobs=draft_logprobs,
                probabilities=draft_probs,
                confidence_stats={
                    "max_probs": max_probs,
                    "entropies": entropies,
                    "top_margins": top_margins
                }
            ))

        return responses
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
