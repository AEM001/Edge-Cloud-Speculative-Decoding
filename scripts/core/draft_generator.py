import logging
import threading
from typing import Iterator, List, Tuple

from core.protocol import DraftRequest, DraftResponse

logger = logging.getLogger(__name__)


class VLLMDraftGenerator:
    
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self._generate_lock = threading.Lock()
        
    def generate_draft_tokens(
        self, 
        request: DraftRequest,
        temperature: float = 0.0,
        top_p: float = 0.95
    ) -> DraftResponse:

        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        from vllm import SamplingParams
        
        prefix = request.verified_prefix
        k = request.num_draft_tokens

        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            max_tokens=k,
            logprobs=1,
        )

        from vllm import TokensPrompt
        with self._generate_lock:
            outputs = self.llm.generate(
                prompts=[TokensPrompt(prompt_token_ids=list(prefix))],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        
        output = outputs[0]
        
        if output.outputs:
            generated_ids = output.outputs[0].token_ids[:k]
        else:
            generated_ids = []
        
        if len(generated_ids) < k:
            logger.warning(f"Generated only {len(generated_ids)} tokens, expected {k}")

        out_logprobs = output.outputs[0].logprobs or []

        draft_token_ids = []
        draft_logprobs = []

        for i, token_id in enumerate(generated_ids):
            if i < len(out_logprobs) and out_logprobs[i] is not None:
                logprobs_dict = out_logprobs[i]
                token_logprob_obj = logprobs_dict.get(token_id, None)
                if token_logprob_obj is not None:
                    token_logprob_val = token_logprob_obj.logprob if hasattr(token_logprob_obj, 'logprob') else float(token_logprob_obj)
                else:
                    token_logprob_val = -float('inf')
            else:
                token_logprob_val = -float('inf')

            draft_token_ids.append(token_id)
            draft_logprobs.append(float(token_logprob_val))
        
        return DraftResponse(
            draft_token_ids=draft_token_ids,
            logprobs=draft_logprobs
        )

    def generate_draft_tokens_batch(
        self,
        requests: List[DraftRequest],
        temperature: float = 0.0,
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
        with self._generate_lock:
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

            for i, token_id in enumerate(generated_ids):
                if i < len(out_logprobs) and out_logprobs[i] is not None:
                    logprobs_dict = out_logprobs[i]
                    token_logprob_obj = logprobs_dict.get(token_id, None)
                    if token_logprob_obj is not None:
                        token_logprob_val = token_logprob_obj.logprob if hasattr(token_logprob_obj, 'logprob') else float(token_logprob_obj)
                    else:
                        token_logprob_val = -float('inf')
                else:
                    token_logprob_val = -float('inf')

                draft_token_ids.append(token_id)
                draft_logprobs.append(float(token_logprob_val))

            responses.append(DraftResponse(
                draft_token_ids=draft_token_ids,
                logprobs=draft_logprobs
            ))

        return responses

    def generate_next_token_candidates_batch(
        self,
        prefixes: List[List[int]],
        num_candidates: int,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> List[List[Tuple[int, float]]]:
        """Return top next-token candidates for each prefix.

        This exposes the local draft model's next-token distribution so callers
        can build proactive branches from likely correction/continuation tokens
        instead of choosing branch offsets by a fixed heuristic.
        """
        if not prefixes:
            return []
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        from vllm import SamplingParams, TokensPrompt

        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            max_tokens=1,
            logprobs=max(1, num_candidates),
        )
        with self._generate_lock:
            outputs = self.llm.generate(
                prompts=[TokensPrompt(prompt_token_ids=list(prefix)) for prefix in prefixes],
                sampling_params=sampling_params,
                use_tqdm=False,
            )

        all_candidates: List[List[Tuple[int, float]]] = []
        for output in outputs:
            if not output.outputs:
                all_candidates.append([])
                continue

            generated_ids = output.outputs[0].token_ids[:1]
            out_logprobs = output.outputs[0].logprobs or []
            logprobs_dict = out_logprobs[0] if out_logprobs else {}
            candidates = []
            for token_id, token_logprob_obj in logprobs_dict.items():
                logprob = (
                    token_logprob_obj.logprob
                    if hasattr(token_logprob_obj, "logprob")
                    else float(token_logprob_obj)
                )
                candidates.append((int(token_id), float(logprob)))

            if generated_ids and all(token_id != generated_ids[0] for token_id, _ in candidates):
                candidates.append((int(generated_ids[0]), -float("inf")))

            candidates.sort(key=lambda item: item[1], reverse=True)
            all_candidates.append(candidates[:num_candidates])

        return all_candidates

    def stream_draft_tokens(
        self,
        request: DraftRequest,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> Iterator[Tuple[int, float]]:
        """Yield draft tokens as soon as each token is available.

        vLLM's offline ``LLM.generate`` API returns a complete request, so this
        uses one-token decode steps to expose a streaming interface to callers
        that can benefit from partial draft work.
        """
        prefix = list(request.verified_prefix)
        for _ in range(request.num_draft_tokens):
            response = self.generate_draft_tokens(
                DraftRequest(verified_prefix=list(prefix), num_draft_tokens=1),
                temperature=temperature,
                top_p=top_p,
            )
            if not response.draft_token_ids:
                break

            token_id = response.draft_token_ids[0]
            logprob = response.logprobs[0] if response.logprobs else -float("inf")
            prefix.append(token_id)
            yield token_id, logprob
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
