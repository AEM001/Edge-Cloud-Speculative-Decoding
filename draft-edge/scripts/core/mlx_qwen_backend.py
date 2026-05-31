"""MLX backend for SpecExtend draft model on Apple Silicon.

Key design:
- Uses mlx_lm KVCache with cache.offset for RoPE position tracking.
- Sparse KV cache: only prefix tokens selected by retrieval + recent window
  are forwarded into the cache. Position is set via cache.offset manipulation.
- Linear draft tree (default): greedily decode `nodes` tokens autoregressively.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import mlx.core as mx
import mlx.nn as mnn
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.utils import load

from core.specextend_backend import DraftTree, DraftTreeResult, SpecExtendDraftBackend

logger = logging.getLogger(__name__)


@dataclass
class MLXBackendConfig:
    model_path: Path
    max_model_len: int = 6000


class MLXKVCacheState:
    """Manages a full-context KVCache with sparse selection support.

    The underlying KVCache accumulates KV entries sequentially with positions
    tracked by cache.offset. For sparse mode, we rebuild the cache from scratch
    using only the selected token indices + recent window, forwarding each
    contiguous run in a single pass with the correct offset.
    """

    def __init__(self, model: mnn.Module) -> None:
        self.model = model
        self._cache: List[KVCache] = make_prompt_cache(model)
        self._cached_token_ids: List[int] = []
        self._cached_positions: List[int] = []  # positions stored in cache
        self._next_logits: Optional[mx.array] = None

    def _reset(self) -> None:
        self._cache = make_prompt_cache(self.model)
        self._cached_token_ids = []
        self._cached_positions = []
        self._next_logits = None

    def _common_prefix_len(self, token_ids: List[int], positions: List[int]) -> int:
        limit = min(len(self._cached_token_ids), len(token_ids))
        i = 0
        while i < limit and self._cached_token_ids[i] == token_ids[i] and self._cached_positions[i] == positions[i]:
            i += 1
        return i

    def _trim_to(self, length: int) -> None:
        """Trim cache to `length` entries (using KVCache.trim)."""
        trim_n = len(self._cached_token_ids) - length
        if trim_n <= 0:
            return
        for c in self._cache:
            c.trim(trim_n)
        self._cached_token_ids = self._cached_token_ids[:length]
        self._cached_positions = self._cached_positions[:length]
        self._next_logits = None

    def _forward_segment(self, token_ids: List[int], positions: List[int]) -> mx.array:
        """Forward a contiguous-position segment; returns all logits [1, T, V]."""
        assert len(token_ids) == len(positions)
        # Set cache offset to the starting position of this segment
        start_pos = positions[0]
        for c in self._cache:
            c.offset = start_pos
        x = mx.array(token_ids)[None]  # [1, T]
        logits = self.model(x, cache=self._cache)
        mx.eval(logits)
        self._cached_token_ids.extend(token_ids)
        self._cached_positions.extend(positions)
        self._next_logits = logits[0, -1]  # [V]
        return logits

    def ensure_sparse(self, token_ids: List[int], positions: List[int]) -> mx.array:
        """Build/reuse cache for a sparse (token_ids, positions) context.

        Returns last-position logits [V].
        """
        assert len(token_ids) == len(positions)
        common = self._common_prefix_len(token_ids, positions)

        if common < len(self._cached_token_ids):
            self._trim_to(common)

        missing_tokens = token_ids[common:]
        missing_positions = positions[common:]

        if not missing_tokens:
            if self._next_logits is None:
                # Re-run last token to restore logits
                if self._cached_token_ids:
                    last_t = self._cached_token_ids[-1:]
                    last_p = self._cached_positions[-1:]
                    self._trim_to(len(self._cached_token_ids) - 1)
                    self._forward_segment(last_t, last_p)
                else:
                    raise ValueError("Empty cache, cannot restore logits.")
            return self._next_logits

        # Forward missing tokens in contiguous-position runs
        # (positions may have gaps for sparse context)
        run_t: List[int] = []
        run_p: List[int] = []
        for t, p in zip(missing_tokens, missing_positions):
            if run_p and p != run_p[-1] + 1:
                # Gap detected: flush current run then reset cache offset
                self._forward_segment(run_t, run_p)
                run_t, run_p = [], []
            run_t.append(t)
            run_p.append(p)
        if run_t:
            self._forward_segment(run_t, run_p)

        return self._next_logits  # type: ignore[return-value]

    def decode_one(self, token_id: int, position: int) -> mx.array:
        """Append a single token at `position`, return logits [V]."""
        return self._forward_segment([token_id], [position])


class MLXQwenDraftBackend(SpecExtendDraftBackend):
    """Apple Silicon MLX draft backend for SpecExtend.

    Implements the same interface as QwenSpecExtendDraftBackend but uses
    mlx_lm instead of PyTorch/Transformers.
    """

    def __init__(self, config: MLXBackendConfig) -> None:
        path = Path(config.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        logger.info("Loading MLX model from %s", path)
        model, tokenizer = load(str(path))
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._kv = MLXKVCacheState(model)
        logger.info("MLX model loaded.")

    # ------------------------------------------------------------------
    # SpecExtendDraftBackend interface
    # ------------------------------------------------------------------

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

        sparse_ids, sparse_pos = self._make_sparse_context(verified_prefix, retrieval_token_indices)

        length = max(1, min(nodes, max_depth))
        input_ids = self._greedy_decode(
            sparse_ids, sparse_pos,
            next_position=len(verified_prefix),
            max_new_tokens=length,
        )

        position_ids = list(range(len(verified_prefix), len(verified_prefix) + len(input_ids)))
        parent_indices = [idx - 1 for idx in range(len(input_ids))]
        attention_mask = self._tree_attention_mask(parent_indices)

        return DraftTreeResult(
            tree=DraftTree(
                input_ids=input_ids,
                position_ids=position_ids,
                parent_indices=parent_indices,
                attention_mask=attention_mask,
            ),
            draft_time_ms=(time.perf_counter() - start) * 1000,
            appended_kv_tokens=len(input_ids),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_sparse_context(
        self,
        prefix: List[int],
        retrieval_token_indices: Optional[List[int]],
    ):
        """Select sparse (token_ids, position_ids) from prefix."""
        recent_tokens = int(os.getenv("DRAFT_RECENT_TOKENS", "128"))
        if not retrieval_token_indices:
            return list(prefix), list(range(len(prefix)))

        selected = {idx for idx in retrieval_token_indices if 0 <= idx < len(prefix)}
        if recent_tokens > 0:
            selected.update(range(max(0, len(prefix) - recent_tokens), len(prefix)))
        if not selected:
            return list(prefix), list(range(len(prefix)))

        ordered = sorted(selected)
        return [prefix[i] for i in ordered], ordered

    def _greedy_decode(
        self,
        context_ids: List[int],
        context_positions: List[int],
        next_position: int,
        max_new_tokens: int,
    ) -> List[int]:
        """Greedily decode `max_new_tokens` tokens using sparse KV cache.

        After decoding, trims draft tokens from the cache so that the next
        call to `ensure_sparse` can cleanly reuse the context KV entries.
        """
        logits = self._kv.ensure_sparse(context_ids, context_positions)
        context_kv_len = len(self._kv._cached_token_ids)

        generated: List[int] = []
        eos = getattr(self.tokenizer, "eos_token_id", None)
        current_logits = logits

        for offset in range(max_new_tokens):
            next_token = int(mx.argmax(current_logits).item())
            generated.append(next_token)
            if eos is not None and next_token == eos:
                break
            current_logits = self._kv.decode_one(next_token, next_position + offset)

        # Trim draft tokens from cache, keep only context KV
        self._kv._trim_to(context_kv_len)

        return generated

    @staticmethod
    def _tree_attention_mask(parent_indices: Sequence[int]) -> List[List[int]]:
        size = len(parent_indices)
        mask = [[0] * size for _ in range(size)]
        for row in range(size):
            idx = row
            while idx >= 0:
                mask[row][idx] = 1
                idx = parent_indices[idx]
        return mask
