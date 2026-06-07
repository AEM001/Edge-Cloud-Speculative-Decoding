"""Custom Qwen backend for the SpecExtend protocol.

The target verifier can use the standard Transformers runtime, but the edge
draft path uses explicit SpecExtend-style KV cache tensors: a full draft cache
plus a retrieval-selected working cache.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.protocol import SpecExtendRequest, SpecExtendResponse
from core.modeling_qwen3_kv import Qwen3ForCausalLM
from core.qwen_kv_cache import KVCache
from core.specextend_backend import DraftSequence, DraftResult, SpecExtendDraftBackend
from core.specextend_retrieval import RetrievalChunk, build_chunks, select_chunks_by_attention
from core.tiered_kv_store import KVLoadMetrics, KVTier, TieredKVStore

logger = logging.getLogger(__name__)


@dataclass
class QwenBackendConfig:
    model_path: Path
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    max_model_len: int = 32768
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"


@dataclass(frozen=True)
class DraftContext:
    token_ids: List[int]
    position_ids: List[int]
    past_key_values: object = None

    def with_path(self, path: Sequence[int], position_start: int) -> "DraftContext":
        path = list(path)
        return DraftContext(
            token_ids=self.token_ids + path,
            position_ids=self.position_ids + list(range(position_start, position_start + len(path))),
        )


class SpecExtendDraftKVCache:
    """SpecExtend draft cache: full KV store plus retrieval-selected working KV.

    This mirrors ``SpecExtend/specextend/application/model_classic.py``:
    ``update_full_draft_cache`` appends newly forwarded K/V tensors to a full
    cache, and ``update_working_cache_retrieval`` rebuilds ``draft_stable_kv``
    by indexing selected chunks out of that full cache.
    """

    def __init__(self, model, max_length: int, device: torch.device) -> None:
        self.model = model
        self.max_length = max_length
        self.device = device
        self.dtype = model.dtype
        self.full_token_ids: List[int] = []
        self.working_token_indices: List[int] = []
        self.last_prefix_logits: Optional[torch.Tensor] = None
        self.full_draft_kv = self._allocate_tensor_kv(max_length)
        self.working_cache = self._allocate_kv_cache(max_length)
        self.chunk_size: int = 32
        self.chunks: List[RetrievalChunk] = []
        self.selected_chunks: List[RetrievalChunk] = []
        self.tiered_store = TieredKVStore(self.full_draft_kv, chunk_size=self.chunk_size)
        self.last_load_metrics: Optional[KVLoadMetrics] = None
        self._retrieval_base_seq_len: int = 0

    @property
    def total_seq_len(self) -> int:
        return len(self.full_token_ids)

    def _allocate_tensor_kv(self, max_length: int) -> List[tuple[torch.Tensor, torch.Tensor]]:
        config = self.model.config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        kv = []
        for layer in self.model.model.layers:
            layer_device = layer.self_attn.q_proj.weight.device
            key = torch.zeros(
                [1, config.num_key_value_heads, max_length, head_dim],
                dtype=self.dtype,
                device=layer_device,
            )
            value = torch.zeros_like(key)
            kv.append((key, value))
        return kv

    def _allocate_kv_cache(self, max_length: int):
        with torch.inference_mode(False):
            current_length_data = torch.zeros(self.model.config.num_hidden_layers * 2, dtype=torch.long, device="cpu")
            cache = []
            for layer_idx, layer in enumerate(self.model.model.layers):
                layer_device = layer.self_attn.q_proj.weight.device
                full_key, _ = self.full_draft_kv[layer_idx]
                data_shape = [1, self.model.config.num_key_value_heads, max_length, full_key.shape[-1]]
                key_data = torch.zeros(data_shape, dtype=self.dtype, device=layer_device)
                value_data = torch.zeros(data_shape, dtype=self.dtype, device=layer_device)
                cache.append(
                    [
                        KVCache(key_data, current_length_data[layer_idx * 2]),
                        KVCache(value_data, current_length_data[layer_idx * 2 + 1]),
                    ]
                )
        return cache

    @staticmethod
    def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
        common = 0
        limit = min(len(left), len(right))
        while common < limit and left[common] == right[common]:
            common += 1
        return common

    def reset(self) -> None:
        self.full_token_ids = []
        self.working_token_indices = []
        self.last_prefix_logits = None
        self.chunks = []
        self.selected_chunks = []
        self.last_load_metrics = None
        self._retrieval_base_seq_len = 0
        self.tiered_store.reset()
        for key, value in self.full_draft_kv:
            key.zero_()
            value.zero_()
        self.working_cache = self._allocate_kv_cache(self.max_length)

    def select_chunks(
        self,
        chunk_ids: Optional[List[int]] = None,
        retrieval_selection_updated: bool = False,
        retrieval_selection_base_len: Optional[int] = None,
    ) -> None:
        """Select working cache by chunk IDs, mirroring sparse-kv's chunk-level retrieval.

        The working cache is the cloud-selected sparse chunks plus the suffix
        generated since that cloud selection was made. At chunk granularity,
        include every chunk overlapping that suffix.
        """
        if chunk_ids is None:
            self.selected_chunks = list(self.chunks)
            self._retrieval_base_seq_len = self.total_seq_len
        else:
            wanted = set(chunk_ids)
            base = (
                retrieval_selection_base_len
                if retrieval_selection_updated and retrieval_selection_base_len is not None
                else self._retrieval_base_seq_len
            )
            selected = [
                chunk for chunk in self.chunks
                if chunk.chunk_id in wanted or chunk.end > base
            ]
            if not selected:
                selected = list(self.chunks)
            self.selected_chunks = selected
            if retrieval_selection_updated:
                self._retrieval_base_seq_len = base

        self._update_working_indices_from_chunks()
        self._rebuild_working_cache()

    def _update_working_indices_from_chunks(self) -> None:
        """Expand selected_chunks into working_token_indices (sorted, unique)."""
        indices = []
        for chunk in self.selected_chunks:
            indices.extend(range(chunk.start, chunk.end))
        self.working_token_indices = sorted(set(indices))

    def _update_chunks_from_full(self) -> bool:
        """Rebuild chunk list from full_token_ids. Return True if new chunks were added."""
        old_count = len(self.chunks)
        self.chunks = build_chunks(len(self.full_token_ids), self.chunk_size)
        return len(self.chunks) > old_count

    def _refresh_selected_tail(self) -> None:
        """Keep the selected tail chunk in sync with the latest full cache, matching sparse-kv."""
        if not self.chunks or not self.selected_chunks:
            return
        selected_ids = {chunk.chunk_id for chunk in self.selected_chunks}
        last_chunk = self.chunks[-1]
        if last_chunk.chunk_id in selected_ids:
            self.selected_chunks = [
                last_chunk if chunk.chunk_id == last_chunk.chunk_id else chunk
                for chunk in self.selected_chunks
            ]

    def append_missing_prefix(self, token_ids: Sequence[int]) -> int:
        token_ids = list(token_ids)
        common = self._common_prefix_len(self.full_token_ids, token_ids)

        if common < len(self.full_token_ids):
            self.reset()
            common = 0

        new_tokens = list(token_ids[common:])
        if not self.working_token_indices:
            self.select_chunks(None)
        return len(new_tokens)

    @torch.inference_mode()
    def append_prefix(self, runtime: "QwenModelRuntime", token_ids: Sequence[int]) -> int:
        token_ids = list(token_ids)
        common = self._common_prefix_len(self.full_token_ids, token_ids)
        if common < len(self.full_token_ids):
            self.reset()
            common = 0

        new_tokens = token_ids[common:]
        if not new_tokens:
            return 0
        if self.total_seq_len + len(new_tokens) > self.max_length:
            raise RuntimeError(
                f"Full draft KV budget exceeded: {self.total_seq_len} + {len(new_tokens)} > {self.max_length}"
            )

        start = self.total_seq_len
        working_start = len(self.working_token_indices)
        input_ids = torch.tensor([new_tokens], dtype=torch.long, device=runtime.device)
        position_ids = torch.arange(start, start + len(new_tokens), dtype=torch.long, device=runtime.device).unsqueeze(0)
        outputs = runtime.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=self.working_cache,
            use_cache=True,
            return_dict=True,
        )
        self.last_prefix_logits = outputs.logits[:, -1, :]
        self._copy_working_tail_to_full(working_start=working_start, dest_start=start, length=len(new_tokens))
        self.full_token_ids.extend(new_tokens)

        # Maintain chunk bookkeeping (mirrors sparse-kv's update_chunks)
        is_new_chunks = self._update_chunks_from_full()
        if not self.selected_chunks:
            # First call: select all chunks
            self.selected_chunks = list(self.chunks)
            self._retrieval_base_seq_len = self.total_seq_len
            self._update_working_indices_from_chunks()
            self._rebuild_working_cache()
        else:
            if is_new_chunks:
                selected_ids = {chunk.chunk_id for chunk in self.selected_chunks}
                last_chunk = self.chunks[-1]
                if last_chunk.chunk_id not in selected_ids:
                    self.selected_chunks.append(last_chunk)
            self._refresh_selected_tail()
            self._update_working_indices_from_chunks()
            self._rebuild_working_cache()

        return len(new_tokens)

    def _rebuild_working_cache_all(self) -> None:
        """Rebuild working cache to cover all full_token_ids (used on first call)."""
        self.working_token_indices = list(range(len(self.full_token_ids)))
        self._rebuild_working_cache()

    def _copy_working_tail_to_full(self, working_start: int, dest_start: int, length: int) -> None:
        dest_end = dest_start + length
        for layer_idx, (full_key, full_value) in enumerate(self.full_draft_kv):
            working_key = self.working_cache[layer_idx][0].data
            working_value = self.working_cache[layer_idx][1].data
            full_key[:, :, dest_start:dest_end, :].copy_(
                working_key[:, :, working_start : working_start + length, :],
                non_blocking=True,
            )
            full_value[:, :, dest_start:dest_end, :].copy_(
                working_value[:, :, working_start : working_start + length, :],
                non_blocking=True,
            )

    def _rebuild_working_cache(self) -> None:
        n = len(self.working_token_indices)
        with torch.inference_mode(False):
            for layer_idx in range(len(self.full_draft_kv)):
                kv_k = self.working_cache[layer_idx][0]
                kv_v = self.working_cache[layer_idx][1]
                kv_k.current_length.fill_(0)
                kv_v.current_length.fill_(0)
        if n == 0:
            self.last_load_metrics = KVLoadMetrics()
            return

        all_gpu = all(
            self.tiered_store.tier_of(chunk.chunk_id) == KVTier.GPU
            for chunk in self.selected_chunks
        )
        if all_gpu:
            metrics = KVLoadMetrics()
            t0 = time.perf_counter()
            for layer_idx, (full_key, full_value) in enumerate(self.full_draft_kv):
                kv_k = self.working_cache[layer_idx][0]
                kv_v = self.working_cache[layer_idx][1]
                index = torch.tensor(
                    self.working_token_indices, dtype=torch.long, device=full_key.device
                )
                key_slice = full_key.index_select(dim=2, index=index)
                value_slice = full_value.index_select(dim=2, index=index)
                kv_k.data[:, :, :n, :].copy_(key_slice, non_blocking=True)
                kv_v.data[:, :, :n, :].copy_(value_slice, non_blocking=True)
                with torch.inference_mode(False):
                    kv_k.current_length.fill_(n)
                    kv_v.current_length.fill_(n)
            torch.cuda.synchronize()
            metrics.gpu_chunks = len(self.selected_chunks)
            metrics.gpu_load_ms = (time.perf_counter() - t0) * 1000
            self.last_load_metrics = metrics
        else:
            working_key_list = [self.working_cache[i][0].data for i in range(len(self.full_draft_kv))]
            working_val_list = [self.working_cache[i][1].data for i in range(len(self.full_draft_kv))]
            chunk_ids = [chunk.chunk_id for chunk in self.selected_chunks]
            metrics = self.tiered_store.load_chunks(
                chunk_ids=chunk_ids,
                total_seq_len=self.total_seq_len,
                token_indices=self.working_token_indices,
                working_key_list=working_key_list,
                working_val_list=working_val_list,
            )
            actual_n = metrics.tokens_written if metrics.tokens_written > 0 else n
            if actual_n != n:
                self.working_token_indices = self.working_token_indices[:actual_n]
            for layer_idx in range(len(self.full_draft_kv)):
                with torch.inference_mode(False):
                    self.working_cache[layer_idx][0].current_length.fill_(actual_n)
                    self.working_cache[layer_idx][1].current_length.fill_(actual_n)
            self.last_load_metrics = metrics

    def context(self) -> DraftContext:
        return DraftContext(
            token_ids=[self.full_token_ids[idx] for idx in self.working_token_indices],
            position_ids=list(self.working_token_indices),
            past_key_values=self.working_cache,
        )

    def evict_to_cpu(self, chunk_id: int) -> None:
        """Evict a chunk's KV data to CPU pinned memory (tier = CPU)."""
        self.tiered_store.evict_to_cpu(chunk_id, self.total_seq_len)

    def evict_to_ssd(self, chunk_id: int) -> None:
        """Serialise a chunk's KV data to SSD (tier = SSD)."""
        self.tiered_store.evict_to_ssd(chunk_id, self.total_seq_len)

    def chunk_tier_summary(self) -> Dict[str, int]:
        """Return {tier_name: chunk_count} across all known chunks."""
        return self.tiered_store.tier_summary(len(self.chunks))

    def _working_cache_seq_len(self) -> int:
        """Current sequence length of working_cache (KVCache)."""
        if not self.working_cache:
            return 0
        return self.working_cache[0][0].current_length.item()

    @torch.inference_mode()
    def next_logits(self, runtime: "QwenModelRuntime") -> torch.Tensor:
        if self.last_prefix_logits is None:
            raise ValueError("Draft prefix logits are unavailable; append the verified prefix before drafting.")
        return self.last_prefix_logits

    @torch.inference_mode()
    def forward_draft_token(
        self,
        runtime: "QwenModelRuntime",
        token_id: int,
        position_id: int,
    ) -> torch.Tensor:
        input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=runtime.device)
        position_ids = torch.tensor([[int(position_id)]], dtype=torch.long, device=runtime.device)
        outputs = runtime.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=self.working_cache,
            use_cache=True,
            return_dict=True,
        )
        return outputs.logits[:, -1, :]


class QwenModelRuntime:
    def __init__(self, config: QwenBackendConfig, use_custom_kv_model: bool = False):
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

        is_awq = bool(
            getattr(getattr(self.tokenizer, "config", None), "quantization_config", None)
            and self.tokenizer.config.quantization_config.get("quant_method") == "awq"
        )
        # Fallback: check model config if tokenizer config didn't have it
        if not is_awq:
            try:
                from transformers import AutoConfig
                mcfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=config.trust_remote_code)
                is_awq = bool(
                    getattr(mcfg, "quantization_config", None)
                    and mcfg.quantization_config.get("quant_method") == "awq"
                )
            except Exception:
                pass

        if use_custom_kv_model:
            self.model = Qwen3ForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=config.dtype,
            ).to(self.device)
            self.model.config._attn_implementation = config.attn_implementation
        else:
            load_kwargs = {
                "trust_remote_code": config.trust_remote_code,
                "attn_implementation": config.attn_implementation,
            }
            if is_awq:
                # AWQ models must use device_map="auto" and should not specify torch_dtype or call .to()
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = config.dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
            if not is_awq:
                self.model = self.model.to(self.device)
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

    @torch.inference_mode()
    def next_logits_positioned(
        self,
        token_ids: Sequence[int],
        position_ids: Sequence[int],
    ) -> torch.Tensor:
        if len(token_ids) != len(position_ids):
            raise ValueError("token_ids and position_ids must have the same length")
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        pos_ids = torch.tensor([list(position_ids)], dtype=torch.long, device=self.device)
        outputs = self.model(input_ids=input_ids, position_ids=pos_ids, use_cache=False)
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


class QwenSpecExtendDraftBackend(SpecExtendDraftBackend):
    supports_pipeline_candidates = False

    def __init__(self, config: QwenBackendConfig):
        self.runtime = QwenModelRuntime(config, use_custom_kv_model=True)
        self.tokenizer = self.runtime.tokenizer
        self.cache = SpecExtendDraftKVCache(
            self.runtime.model,
            max_length=config.max_model_len,
            device=self.runtime.device,
        )
        self.kv_tier_policy = "gpu"
        self.kv_tier_keep_recent_chunks = 0

    def set_kv_tier_policy(self, policy: str = "gpu", keep_recent_chunks: int = 0) -> None:
        """Set a synthetic KV hierarchy policy for retrieval-update benchmarks.

        ``gpu`` keeps the original all-HBM path. ``cpu`` and ``ssd`` are applied
        only when target attention returns a new sparse chunk selection. This
        measures the cost of selecting chunks that live outside GPU memory
        without charging every ordinary tail refresh.
        """
        policy = (policy or "gpu").strip().lower()
        if policy not in {"gpu", "cpu", "ssd"}:
            raise ValueError(f"Unsupported KV tier policy: {policy}")
        self.kv_tier_policy = policy
        self.kv_tier_keep_recent_chunks = max(0, int(keep_recent_chunks))

    def _apply_kv_tier_policy(self) -> None:
        if self.kv_tier_policy == "gpu" or not self.cache.chunks:
            return
        protected = set()
        if self.kv_tier_keep_recent_chunks:
            protected = {
                chunk.chunk_id
                for chunk in self.cache.chunks[-self.kv_tier_keep_recent_chunks :]
            }
        for chunk in self.cache.chunks:
            if chunk.chunk_id in protected:
                continue
            if self.kv_tier_policy == "cpu":
                self.cache.evict_to_cpu(chunk.chunk_id)
            elif self.kv_tier_policy == "ssd":
                self.cache.evict_to_ssd(chunk.chunk_id)

    def build_draft(
        self,
        verified_prefix: List[int],
        correction_token_id: Optional[int],
        draft_length: int,
        retrieval_chunk_ids: Optional[List[int]] = None,
        retrieval_selection_updated: bool = False,
        retrieval_selection_base_len: Optional[int] = None,
    ) -> DraftResult:
        start = time.perf_counter()
        if correction_token_id is not None and (
            not verified_prefix or verified_prefix[-1] != correction_token_id
        ):
            verified_prefix = list(verified_prefix) + [correction_token_id]

        appended = self.cache.append_prefix(self.runtime, verified_prefix)
        if retrieval_selection_updated and retrieval_chunk_ids is not None:
            self._apply_kv_tier_policy()
        self.cache.select_chunks(
            retrieval_chunk_ids,
            retrieval_selection_updated=retrieval_selection_updated,
            retrieval_selection_base_len=retrieval_selection_base_len,
        )
        kv_load_metrics = (
            self.cache.last_load_metrics
            if retrieval_selection_updated and retrieval_chunk_ids is not None
            else KVLoadMetrics()
        )
        draft_context = self.cache.context()

        draft_seq = self._generate_draft_sequence(
            draft_context,
            draft_length=max(1, draft_length),
            position_start=len(verified_prefix),
        )
        return DraftResult(
            draft=draft_seq,
            draft_time_ms=(time.perf_counter() - start) * 1000,
            appended_kv_tokens=appended,
            kv_load_metrics=kv_load_metrics,
        )

    def _generate_draft_sequence(
        self,
        prefix: DraftContext,
        draft_length: int,
        position_start: int,
    ) -> DraftSequence:
        length = max(1, draft_length)
        if prefix.past_key_values is None:
            raise ValueError("Draft generation requires the explicit SpecExtend working KV cache.")
        input_ids = self._generate_linear_from_working_cache(
            next_position_id=position_start,
            max_new_tokens=length,
        )
        position_ids = list(range(position_start, position_start + len(input_ids)))
        return DraftSequence(
            input_ids=input_ids,
            position_ids=position_ids,
        )

    def _snapshot_working_lengths(self) -> List[int]:
        """Snapshot current_length for every KVCache slot in working_cache."""
        return [
            self.cache.working_cache[i][j].current_length.item()
            for i in range(len(self.cache.working_cache))
            for j in range(2)
        ]

    def _restore_working_lengths(self, snapshot: List[int]) -> None:
        """Restore current_length from a snapshot (no data copy needed)."""
        idx = 0
        with torch.inference_mode(False):
            for i in range(len(self.cache.working_cache)):
                for j in range(2):
                    self.cache.working_cache[i][j].current_length.fill_(snapshot[idx])
                    idx += 1

    def _generate_linear_from_working_cache(self, next_position_id: int, max_new_tokens: int) -> List[int]:
        generated: List[int] = []
        length_snapshot = self._snapshot_working_lengths()
        logits = self.cache.next_logits(self.runtime)
        try:
            for offset in range(max_new_tokens):
                next_token = int(torch.argmax(logits, dim=-1).item())
                generated.append(next_token)
                logits = self.cache.forward_draft_token(
                    self.runtime,
                    token_id=next_token,
                    position_id=next_position_id + offset,
                )
                if self.runtime.eos_token_id is not None and next_token == self.runtime.eos_token_id:
                    break
        finally:
            self._restore_working_lengths(length_snapshot)
        return generated

    @torch.inference_mode()
    def build_prefetch_draft(
        self,
        verified_prefix: List[int],
        draft_length: int,
        retrieval_chunk_ids: Optional[List[int]] = None,
    ) -> "DraftResult":
        """Build a pipeline-prefetch draft without committing to the main cache.

        Runs under _cache_lock so it cannot race with build_draft.
        Works on the fixed-size chunk-only working cache: forward any candidate
        tokens beyond full_token_ids temporarily, generate the draft, then
        restore current_length.  Does NOT write to full_draft_kv or
        full_token_ids.
        """
        start = time.perf_counter()
        self.cache.select_chunks(retrieval_chunk_ids)
        full_ids = list(self.cache.full_token_ids)
        last_prefix_logits = self.cache.last_prefix_logits
        if last_prefix_logits is None:
            return None

        candidate_tokens = verified_prefix[len(full_ids):]
        length_snap = self._snapshot_working_lengths()
        try:
            logits = last_prefix_logits
            for i, tok in enumerate(candidate_tokens):
                pos = len(full_ids) + i
                input_ids_t = torch.tensor([[tok]], dtype=torch.long, device=self.runtime.device)
                pos_ids_t = torch.tensor([[pos]], dtype=torch.long, device=self.runtime.device)
                out = self.runtime.model(
                    input_ids=input_ids_t,
                    position_ids=pos_ids_t,
                    past_key_values=self.cache.working_cache,
                    use_cache=True,
                    return_dict=True,
                )
                logits = out.logits[:, -1, :]
            self.cache.last_prefix_logits = logits

            position_start = len(verified_prefix)
            length = max(1, draft_length)
            input_ids = self._generate_linear_from_working_cache(
                next_position_id=position_start,
                max_new_tokens=length,
            )
        finally:
            self._restore_working_lengths(length_snap)
            self.cache.last_prefix_logits = last_prefix_logits

        position_ids = list(range(position_start, position_start + len(input_ids)))
        draft_seq = DraftSequence(
            input_ids=input_ids,
            position_ids=position_ids,
        )
        return DraftResult(
            draft=draft_seq,
            draft_time_ms=(time.perf_counter() - start) * 1000,
            appended_kv_tokens=0,
        )


class QwenSpecExtendTargetBackend:
    def __init__(self, config: QwenBackendConfig):
        self.runtime = QwenModelRuntime(config)
        self.tokenizer = self.runtime.tokenizer

    def runtime_debug_info(self) -> Dict[str, object]:
        return {
            "backend": "custom_qwen3",
            "draft_mode": "linear",
            "attention_scores": True,
            "kv_cache": "transformers_dynamic_cache_plus_visible_token_cache",
            "draft_cache": "explicit_full_kv_plus_retrieval_working_kv",
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
    def verify(self, request: SpecExtendRequest) -> SpecExtendResponse:
        start = time.perf_counter()
        accepted_len, correction_token_id = self._verify_linear_path(
            request.prefix_ids,
            list(request.draft_ids),
        )
        accepted_indices = list(range(accepted_len))

        target_attn_scores = None
        selected_chunk_ids = None
        if request.retrieve_attn_scores:
            accepted_tokens = [
                request.draft_ids[idx]
                for idx in accepted_indices
                if 0 <= idx < len(request.draft_ids)
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
        return SpecExtendResponse(
            request_id=request.request_id,
            accepted_len=max(0, accepted_len),
            correction_token_id=correction_token_id,
            accepted_indices=accepted_indices,
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
