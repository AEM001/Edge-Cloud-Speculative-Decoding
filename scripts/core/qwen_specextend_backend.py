"""Custom Qwen backend for the SpecExtend protocol.

It uses Hugging Face/Transformers models directly so the implementation can
control draft trees, inspect target attention scores, and keep draft-side cache
state visible to SpecExtend.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.protocol import SpecExtendTreeRequest, SpecExtendTreeResponse
from core.specextend_backend import DraftTree, DraftTreeResult, SpecExtendDraftBackend

logger = logging.getLogger(__name__)


@dataclass
class QwenBackendConfig:
    model_path: Path
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    max_model_len: int = 32768
    trust_remote_code: bool = True


class VisibleTokenCache:
    """Token-level cache bookkeeping for SpecExtend retrieval decisions.

    Transformers owns the tensor KV cache internally for this correctness-first
    implementation. This class keeps the full draft-cache token positions and a
    working-cache view selected by retrieval, which is the backend-visible state
    needed by the edge SpecExtend loop.
    """

    def __init__(self) -> None:
        self.full_token_ids: List[int] = []
        self.working_token_indices: List[int] = []

    def append_missing_prefix(self, token_ids: Sequence[int]) -> int:
        common = 0
        limit = min(len(self.full_token_ids), len(token_ids))
        while common < limit and self.full_token_ids[common] == token_ids[common]:
            common += 1

        if common < len(self.full_token_ids):
            self.full_token_ids = self.full_token_ids[:common]

        new_tokens = list(token_ids[common:])
        self.full_token_ids.extend(new_tokens)
        if not self.working_token_indices:
            self.working_token_indices = list(range(len(self.full_token_ids)))
        return len(new_tokens)

    def select_working_tokens(self, indices: Optional[Iterable[int]]) -> None:
        if indices is None:
            self.working_token_indices = list(range(len(self.full_token_ids)))
            return

        valid = sorted({idx for idx in indices if 0 <= idx < len(self.full_token_ids)})
        self.working_token_indices = valid or list(range(len(self.full_token_ids)))


class QwenModelRuntime:
    def __init__(self, config: QwenBackendConfig):
        path = Path(config.model_path)
        if not path.exists() and os.sep in str(path):
            raise FileNotFoundError(f"Model not found: {path}")

        self.config = config
        self.model_path = str(path)
        self.device = torch.device(config.device)
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
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

        # KV cache for efficient autoregressive generation
        # Maps request_id -> (past_key_values, cached_seq)
        self._kv_cache: Optional[tuple] = None
        self._kv_cache_seq: List[int] = []
        self._request_kv_cache: dict = {}
        self._request_kv_seq: dict = {}

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @torch.inference_mode()
    def next_logits(self, token_ids: Sequence[int]) -> torch.Tensor:
        token_ids = list(token_ids)

        # Fast path: new tokens extend cached sequence
        if (
            self._kv_cache is not None
            and len(token_ids) > len(self._kv_cache_seq)
            and token_ids[: len(self._kv_cache_seq)] == self._kv_cache_seq
        ):
            new_ids = token_ids[len(self._kv_cache_seq) :]
            input_ids = torch.tensor([new_ids], dtype=torch.long, device=self.device)
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=self._kv_cache,
                use_cache=True,
            )
            self._kv_cache = outputs.past_key_values
            self._kv_cache_seq = token_ids
            return outputs.logits[:, -1, :]

        # Slow path: prefix changed, recompute from scratch
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        outputs = self.model(input_ids=input_ids, use_cache=True)
        self._kv_cache = outputs.past_key_values
        self._kv_cache_seq = token_ids
        return outputs.logits[:, -1, :]

    def clear_kv_cache(self) -> None:
        self._kv_cache = None
        self._kv_cache_seq = []

    def clear_request_kv_cache(self, request_id: str) -> None:
        self._request_kv_cache.pop(request_id, None)
        self._request_kv_seq.pop(request_id, None)

    @torch.inference_mode()
    def next_logits_for_request(
        self, request_id: str, token_ids: Sequence[int]
    ) -> torch.Tensor:
        """next_logits with per-request KV cache reuse."""
        token_ids = list(token_ids)
        cached_seq = self._request_kv_seq.get(request_id, [])
        cached_kv = self._request_kv_cache.get(request_id)

        if (
            cached_kv is not None
            and len(token_ids) > len(cached_seq)
            and token_ids[: len(cached_seq)] == cached_seq
        ):
            new_ids = token_ids[len(cached_seq) :]
            input_ids = torch.tensor([new_ids], dtype=torch.long, device=self.device)
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=cached_kv,
                use_cache=True,
            )
        else:
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            outputs = self.model(input_ids=input_ids, use_cache=True)

        self._request_kv_cache[request_id] = outputs.past_key_values
        self._request_kv_seq[request_id] = token_ids
        return outputs.logits[:, -1, :]

    @torch.inference_mode()
    def generate_token_ids(self, token_ids: Sequence[int], max_new_tokens: int) -> List[int]:
        generated = list(token_ids)
        for _ in range(max_new_tokens):
            next_token = int(torch.argmax(self.next_logits(generated), dim=-1).item())
            generated.append(next_token)
            if self.eos_token_id is not None and next_token == self.eos_token_id:
                break
        return generated[len(token_ids):]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)


class QwenSpecExtendDraftBackend(SpecExtendDraftBackend):
    def __init__(self, config: QwenBackendConfig):
        self.runtime = QwenModelRuntime(config)
        self.tokenizer = self.runtime.tokenizer
        self.cache = VisibleTokenCache()

    def build_draft_tree(
        self,
        verified_prefix: List[int],
        correction_token_id: Optional[int],
        nodes: int,
        threshold: float,
        max_depth: int,
        retrieval_token_indices: Optional[List[int]] = None,
    ) -> DraftTreeResult:
        start = time.perf_counter()
        if correction_token_id is not None and (
            not verified_prefix or verified_prefix[-1] != correction_token_id
        ):
            verified_prefix = list(verified_prefix) + [correction_token_id]

        appended = self.cache.append_missing_prefix(verified_prefix)
        self.cache.select_working_tokens(retrieval_token_indices)

        tree = self._grow_tree(verified_prefix, nodes=max(1, nodes), max_depth=max(1, max_depth))
        # Purge per-path KV cache entries from this build to free GPU memory
        stale = [k for k in self.runtime._request_kv_cache if k not in self.runtime._request_kv_seq]
        for k in list(self.runtime._request_kv_cache):
            del self.runtime._request_kv_cache[k]
            self.runtime._request_kv_seq.pop(k, None)
        return DraftTreeResult(
            tree=tree,
            draft_time_ms=(time.perf_counter() - start) * 1000,
            appended_kv_tokens=appended,
        )

    def _grow_tree(self, prefix: List[int], nodes: int, max_depth: int) -> DraftTree:
        device = self.runtime.device
        # Use a unique tree-build session id so per-path caches don't collide
        # across calls but DO reuse within a single tree build.
        session = uuid.uuid4().hex
        frontier = [([], -1, 0.0)]
        records: List[Dict[str, object]] = []
        branch_width = max(1, min(nodes, int(nodes**0.5) or 1))

        for depth in range(max_depth):
            candidates = []
            for path, parent_idx, score in frontier:
                # Cache key per path so the prefix+path KV cache is reused
                path_key = f"{session}:{'_'.join(map(str, path))}"
                logits = self.runtime.next_logits_for_request(path_key, prefix + path)
                logprobs = F.log_softmax(logits[0], dim=-1)
                width = min(branch_width, nodes - len(records), logprobs.numel())
                values, token_ids = torch.topk(logprobs, k=width)
                for token_id, logprob in zip(token_ids.tolist(), values.tolist()):
                    candidates.append((path + [int(token_id)], parent_idx, score + float(logprob)))

            if not candidates:
                break

            candidates.sort(key=lambda item: item[2], reverse=True)
            chosen = candidates[: max(1, min(nodes - len(records), len(candidates)))]
            next_frontier = []
            for path, _old_parent_idx, score in chosen:
                parent_idx = self._find_parent_index(records, path[:-1])
                node_idx = len(records)
                records.append(
                    {
                        "token_id": path[-1],
                        "position_id": len(prefix) + len(path) - 1,
                        "parent_idx": parent_idx,
                        "path": path,
                    }
                )
                next_frontier.append((path, node_idx, score))
            frontier = next_frontier

            if len(records) >= nodes:
                break

        if not records:
            logits = self.runtime.next_logits_for_request(f"{session}:", prefix)
            token_id = int(torch.argmax(logits, dim=-1).item())
            records.append(
                {
                    "token_id": token_id,
                    "position_id": len(prefix),
                    "parent_idx": -1,
                    "path": [token_id],
                }
            )

        records = records[:nodes]
        input_ids = [int(record["token_id"]) for record in records]
        position_ids = [int(record["position_id"]) for record in records]
        parent_indices = [int(record["parent_idx"]) for record in records]
        attention_mask = self._tree_attention_mask(parent_indices)

        return DraftTree(
            input_ids=input_ids,
            position_ids=position_ids,
            parent_indices=parent_indices,
            attention_mask=attention_mask,
        )

    @staticmethod
    def _find_parent_index(records: List[Dict[str, object]], parent_path: List[int]) -> int:
        if not parent_path:
            return -1
        for idx in range(len(records) - 1, -1, -1):
            if records[idx]["path"] == parent_path:
                return idx
        return -1

    @staticmethod
    def _tree_attention_mask(parent_indices: Sequence[int]) -> List[List[int]]:
        size = len(parent_indices)
        mask = [[0 for _ in range(size)] for _ in range(size)]
        for row in range(size):
            idx = row
            while idx >= 0:
                mask[row][idx] = 1
                idx = parent_indices[idx]
        return mask


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
            "kv_cache": "transformers_cache_plus_visible_token_cache",
        }

    def get_info(self) -> Dict[str, object]:
        return {
            "model_path": self.runtime.model_path,
            "device": str(self.runtime.device),
            "dtype": str(self.runtime.config.dtype).replace("torch.", ""),
            "max_model_len": self.runtime.config.max_model_len,
        }

    @torch.inference_mode()
    def verify_tree(self, request: SpecExtendTreeRequest) -> SpecExtendTreeResponse:
        start = time.perf_counter()
        rid = request.request_id
        leaf_indices = self._leaf_indices(request.parent_indices)
        paths = self._paths_from_tree(request.tree_input_ids, request.parent_indices)

        accepted_indices: List[int] = []
        correction_token_id: Optional[int] = None
        best_accept_len = -1

        for node_idx in leaf_indices:
            path = paths[node_idx]
            accepted_len, correction = self._verify_path(rid, request.prefix_ids, path)
            if accepted_len > best_accept_len:
                best_accept_len = accepted_len
                accepted_indices = self._indices_for_path(node_idx, request.parent_indices)
                accepted_indices = accepted_indices[:accepted_len]
                correction_token_id = correction

        if best_accept_len < 0:
            best_accept_len = 0
            correction_token_id = int(
                torch.argmax(
                    self.runtime.next_logits_for_request(rid, request.prefix_ids), dim=-1
                ).item()
            )

        target_attn_scores = None
        if request.retrieve_attn_scores:
            accepted_tokens = [
                request.tree_input_ids[idx]
                for idx in accepted_indices
                if 0 <= idx < len(request.tree_input_ids)
            ]
            scoring_ids = list(request.prefix_ids) + accepted_tokens
            if correction_token_id is not None:
                scoring_ids.append(correction_token_id)
            target_attn_scores = self._last_query_attention_scores(scoring_ids)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return SpecExtendTreeResponse(
            request_id=request.request_id,
            accepted_len=max(0, best_accept_len),
            correction_token_id=correction_token_id,
            accepted_tree_indices=accepted_indices,
            server_verify_time_ms=elapsed_ms,
            target_attn_scores=target_attn_scores,
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

    def _verify_path(
        self, request_id: str, prefix_ids: List[int], path: List[int]
    ) -> tuple[int, Optional[int]]:
        current = list(prefix_ids)
        accepted = 0
        correction = None
        for draft_token in path:
            next_token = int(
                torch.argmax(
                    self.runtime.next_logits_for_request(request_id, current), dim=-1
                ).item()
            )
            if next_token != draft_token:
                correction = next_token
                break
            current.append(draft_token)
            accepted += 1
        if accepted == len(path):
            correction = int(
                torch.argmax(
                    self.runtime.next_logits_for_request(request_id, current), dim=-1
                ).item()
            )
        return accepted, correction

    @staticmethod
    def _leaf_indices(parent_indices: Sequence[int]) -> List[int]:
        """Return indices of nodes that are not a parent of any other node (leaves)."""
        parents = set(parent_indices)
        return [i for i in range(len(parent_indices)) if i not in parents]

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
    def _last_query_attention_scores(self, token_ids: Sequence[int]) -> List[float]:
        if not token_ids:
            return []
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.runtime.device)
        outputs = self.runtime.model(
            input_ids=input_ids,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        attentions = outputs.attentions
        if not attentions:
            return [0.0 for _ in token_ids]
        last_layer = attentions[-1][0]
        scores = last_layer[:, -1, :].mean(dim=0)
        return [float(value) for value in scores.detach().float().cpu().tolist()]


def dtype_from_env(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return torch.float16
