"""Custom Qwen backend for the SpecExtend cloud target.

This module contains only the target/verifier backend. Draft-side code lives in
the separate `draft-edge/` package (intended for the Mac MLX runtime).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.protocol import SpecExtendTreeRequest, SpecExtendTreeResponse
from core.specextend_retrieval import build_chunks, select_chunks_by_attention

logger = logging.getLogger(__name__)


@dataclass
class QwenBackendConfig:
    model_path: Path
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    max_model_len: int = 6000
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"
    gpu_memory_fraction: Optional[float] = None


class QwenModelRuntime:
    def __init__(self, config: QwenBackendConfig):
        path = Path(config.model_path)
        if not path.exists() and os.sep in str(path):
            raise FileNotFoundError(f"Model not found: {path}")

        self.config = config
        self.model_path = str(path)
        self.device = torch.device(config.device)
        if config.gpu_memory_fraction is not None and self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(
                config.gpu_memory_fraction, self.device.index
            )
            logger.info(
                "Set GPU memory fraction to %.2f on %s",
                config.gpu_memory_fraction,
                self.device,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=config.dtype,
            trust_remote_code=config.trust_remote_code,
            attn_implementation=config.attn_implementation,
        ).to(self.device)
        self.model.eval()
        self._cache_token_ids: List[int] = []
        self._cache = None
        self._cache_next_logits: Optional[torch.Tensor] = None

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @torch.inference_mode()
    def next_logits(self, token_ids: Sequence[int]) -> torch.Tensor:
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        outputs = self.model(input_ids=input_ids, use_cache=False)
        return outputs.logits[:, -1, :]

    def reset_cache(self) -> None:
        self._cache_token_ids = []
        self._cache = None
        self._cache_next_logits = None

    def _rewind_cache_for_prefix(self, common_length: int) -> int:
        if common_length <= 1 or self._cache is None:
            self.reset_cache()
            return 0

        replay_from = common_length - 1
        self._cache.crop(replay_from)
        self._cache_token_ids = self._cache_token_ids[:replay_from]
        self._cache_next_logits = None
        return replay_from

    @staticmethod
    def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
        common = 0
        limit = min(len(left), len(right))
        while common < limit and left[common] == right[common]:
            common += 1
        return common

    @torch.inference_mode()
    def _forward_cache_tokens(self, token_ids: Sequence[int]) -> torch.Tensor:
        if not token_ids:
            if self._cache_next_logits is None:
                raise ValueError("Cannot compute next logits for an empty cache.")
            return self._cache_next_logits

        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=self._cache,
            use_cache=True,
            return_dict=True,
        )
        self._cache = outputs.past_key_values
        self._cache_token_ids.extend(int(token_id) for token_id in token_ids)
        self._cache_next_logits = outputs.logits[:, -1, :]
        return outputs.logits

    @torch.inference_mode()
    def ensure_cache(self, token_ids: Sequence[int]) -> torch.Tensor:
        token_ids = list(token_ids)
        common = self._common_prefix_len(self._cache_token_ids, token_ids)
        if common < len(self._cache_token_ids):
            common = self._rewind_cache_for_prefix(common)

        missing = token_ids[common:]
        if missing:
            self._forward_cache_tokens(missing)

        if self._cache_next_logits is None:
            raise ValueError("Prompt cache is empty; at least one token is required.")
        return self._cache_next_logits

    @torch.inference_mode()
    def generate_token_ids_cached(self, token_ids: Sequence[int], max_new_tokens: int) -> List[int]:
        generated: List[int] = []
        self.ensure_cache(token_ids)
        for _ in range(max_new_tokens):
            next_token = int(torch.argmax(self._cache_next_logits, dim=-1).item())
            generated.append(next_token)
            self._forward_cache_tokens([next_token])
            if self.eos_token_id is not None and next_token == self.eos_token_id:
                break
        return generated

    @torch.inference_mode()
    def generate_token_ids(self, token_ids: Sequence[int], max_new_tokens: int) -> List[int]:
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        return output_ids[0, input_ids.shape[1] :].tolist()

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)


class QwenSpecExtendTargetBackend:
    def __init__(self, config: QwenBackendConfig):
        self.runtime = QwenModelRuntime(config)
        self.tokenizer = self.runtime.tokenizer

    def runtime_debug_info(self) -> Dict[str, object]:
        return {
            "backend": "custom_qwen3",
            "specextend_tree_verify": True,
            "attention_scores": True,
            "tree_attention": True,
            "kv_cache": "transformers_dynamic_cache",
            "draft_tree_default": "linear",
        }

    def get_info(self) -> Dict[str, object]:
        return {
            "model_path": self.runtime.model_path,
            "device": str(self.runtime.device),
            "dtype": str(self.runtime.config.dtype).replace("torch.", ""),
            "max_model_len": self.runtime.config.max_model_len,
            "attn_implementation": self.runtime.config.attn_implementation,
        }

    @torch.inference_mode()
    def verify_tree(self, request: SpecExtendTreeRequest) -> SpecExtendTreeResponse:
        start = time.perf_counter()
        paths = self._paths_from_tree(request.tree_input_ids, request.parent_indices)

        accepted_indices: List[int] = []
        correction_token_id: Optional[int] = None
        best_accept_len = -1
        max_path_len = max((len(path) for path in paths), default=0)

        if self._is_linear_tree(request.parent_indices):
            accepted_len, correction_token_id = self._verify_linear_path(
                request.prefix_ids,
                list(request.tree_input_ids),
            )
            best_accept_len = accepted_len
            accepted_indices = list(range(accepted_len))
        elif max_path_len > 0:
            target_tokens = self.runtime.generate_token_ids_cached(request.prefix_ids, max_path_len + 1)
            for node_idx, path in enumerate(paths):
                accepted_len, correction = self._verify_path_against_tokens(path, target_tokens)
                if accepted_len > best_accept_len:
                    best_accept_len = accepted_len
                    accepted_indices = self._indices_for_path(node_idx, request.parent_indices)
                    accepted_indices = accepted_indices[:accepted_len]
                    correction_token_id = correction
        else:
            target_tokens = self.runtime.generate_token_ids_cached(request.prefix_ids, 1)
            correction_token_id = target_tokens[0] if target_tokens else None

        if best_accept_len < 0:
            best_accept_len = 0

        target_attn_scores = None
        selected_chunk_ids = None
        if request.retrieve_attn_scores:
            accepted_tokens = [
                request.tree_input_ids[idx]
                for idx in accepted_indices
                if 0 <= idx < len(request.tree_input_ids)
            ]
            scoring_ids = list(request.prefix_ids) + accepted_tokens
            if correction_token_id is not None:
                scoring_ids.append(correction_token_id)
            scores = self._last_query_attention_scores_cached(scoring_ids)
            return_full_scores = bool((request.metadata or {}).get("return_attention_scores", False))
            if return_full_scores:
                target_attn_scores = scores
            else:
                chunks = build_chunks(
                    total_seq_len=max(0, len(scores)),
                    chunk_size=request.retrieval_chunk_size,
                )
                selected = select_chunks_by_attention(
                    chunks,
                    scores,
                    top_k_chunks=request.retrieve_top_k,
                )
                selected_chunk_ids = [chunk.chunk_id for chunk in selected]

        elapsed_ms = (time.perf_counter() - start) * 1000
        return SpecExtendTreeResponse(
            request_id=request.request_id,
            accepted_len=max(0, best_accept_len),
            correction_token_id=correction_token_id,
            accepted_tree_indices=accepted_indices,
            server_verify_time_ms=elapsed_ms,
            target_attn_scores=target_attn_scores,
            selected_chunk_ids=selected_chunk_ids,
            model_time_ms=elapsed_ms,
            http_overhead_ms=0.0,
        )

    def generate_text(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> Dict[str, object]:
        start = time.perf_counter()
        input_ids = self.tokenizer.encode(prompt)
        if temperature and temperature > 0:
            generated = self._sample(input_ids, max_tokens=max_tokens, temperature=temperature)
        else:
            generated = self.runtime.generate_token_ids(input_ids, max_tokens)
        return {
            "text": self.runtime.decode(generated),
            "tokens_generated": len(generated),
            "generation_time_ms": (time.perf_counter() - start) * 1000,
        }

    def _sample(self, token_ids: List[int], max_tokens: int, temperature: float) -> List[int]:
        generated: List[int] = []
        current = list(token_ids)
        for _ in range(max_tokens):
            logits = self.runtime.next_logits(current) / max(temperature, 1e-6)
            probs = torch.softmax(logits[0], dim=-1)
            next_token = int(torch.multinomial(probs, 1).item())
            generated.append(next_token)
            current.append(next_token)
            if self.runtime.eos_token_id is not None and next_token == self.runtime.eos_token_id:
                break
        return generated

    @staticmethod
    def _is_linear_tree(parent_indices: Sequence[int]) -> bool:
        return all(parent_idx == idx - 1 for idx, parent_idx in enumerate(parent_indices))

    @torch.inference_mode()
    def _verify_linear_path(self, prefix_ids: List[int], path: List[int]) -> tuple[int, Optional[int]]:
        next_logits = self.runtime.ensure_cache(prefix_ids)
        if not path:
            correction = int(torch.argmax(next_logits, dim=-1).item())
            return 0, correction

        path_logits = self.runtime._forward_cache_tokens(path)
        accepted = 0
        correction = None
        for idx, draft_token in enumerate(path):
            logits = next_logits if idx == 0 else path_logits[:, idx - 1, :]
            target_token = int(torch.argmax(logits, dim=-1).item())
            if target_token != draft_token:
                correction = target_token
                break
            accepted += 1

        if accepted == len(path):
            correction = int(torch.argmax(path_logits[:, -1, :], dim=-1).item())
        return accepted, correction

    @staticmethod
    def _verify_path_against_tokens(path: List[int], target_tokens: List[int]) -> tuple[int, Optional[int]]:
        accepted = 0
        correction = None
        for idx, draft_token in enumerate(path):
            if idx >= len(target_tokens):
                break
            target_token = int(target_tokens[idx])
            if target_token != draft_token:
                correction = target_token
                break
            accepted += 1
        if accepted == len(path) and accepted < len(target_tokens):
            correction = int(target_tokens[accepted])
        return accepted, correction

    @staticmethod
    def _paths_from_tree(tree_input_ids: Sequence[int], parent_indices: Sequence[int]) -> List[List[int]]:
        paths: List[List[int]] = []
        for idx in range(len(tree_input_ids)):
            reverse_path = []
            current = idx
            seen = set()
            while current >= 0 and current not in seen:
                seen.add(current)
                reverse_path.append(int(tree_input_ids[current]))
                current = int(parent_indices[current])
            paths.append(list(reversed(reverse_path)))
        return paths

    @staticmethod
    def _indices_for_path(node_idx: int, parent_indices: Sequence[int]) -> List[int]:
        indices = []
        current = node_idx
        seen = set()
        while current >= 0 and current not in seen:
            seen.add(current)
            indices.append(current)
            current = int(parent_indices[current])
        return list(reversed(indices))

    @torch.inference_mode()
    def _last_query_attention_scores_cached(self, token_ids: Sequence[int]) -> List[float]:
        if len(token_ids) < 2:
            return [1.0 for _ in token_ids]

        prefix_ids = list(token_ids[:-1])
        query_id = int(token_ids[-1])
        self.runtime.ensure_cache(prefix_ids)
        input_ids = torch.tensor([[query_id]], dtype=torch.long, device=self.runtime.device)
        outputs = self.runtime.model(
            input_ids=input_ids,
            past_key_values=self.runtime._cache,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        self.runtime._cache = outputs.past_key_values
        self.runtime._cache_token_ids.append(query_id)
        self.runtime._cache_next_logits = outputs.logits[:, -1, :]

        attentions = outputs.attentions
        if not attentions:
            return [0.0 for _ in token_ids]
        last_layer = attentions[-1][0]
        scores = last_layer[:, -1, :].mean(dim=0)
        return [float(value) for value in scores.detach().float().cpu().tolist()]


def dtype_from_env(value: str) -> torch.dtype | None:
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"auto", "none"}:
        return None
    return torch.float16
