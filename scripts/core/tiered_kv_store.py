"""Tiered KV storage abstraction for SpecExtend research.

Simulates a three-tier memory hierarchy (GPU HBM / CPU pinned / SSD) for
studying the trade-off between sparse KV selection and load-latency overhead.

Usage
-----
At construction time every chunk lives on GPU (tier = "gpu").  To simulate
different storage policies call ``evict_to_cpu`` or ``evict_to_ssd`` on
specific chunk IDs before running inference.  ``load_chunks_to_gpu`` is called
automatically by ``SpecExtendDraftKVCache._rebuild_working_cache`` and records
precise load times into a ``KVLoadMetrics`` object.

Design notes
------------
- GPU tier  : slice of the pre-allocated ``full_draft_kv`` tensor – zero-copy.
- CPU tier  : chunk data lives in CPU pinned memory; loaded via cudaMemcpyAsync
              (``tensor.cuda(non_blocking=True)``).
- SSD tier  : chunk data serialised with ``torch.save``; loaded via disk read
              then H2D copy.  The SSD path is suitable for controlled latency
              experiments on NVMe storage.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class KVTier(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    SSD = "ssd"


@dataclass
class KVLoadMetrics:
    """Per-rebuild-call latency breakdown.

    All times are wall-clock milliseconds measured around the actual data
    movement operations only (excludes index_select / copy_ for chunks already
    on GPU).
    """

    gpu_chunks: int = 0
    cpu_chunks: int = 0
    ssd_chunks: int = 0

    gpu_load_ms: float = 0.0
    cpu_load_ms: float = 0.0
    ssd_load_ms: float = 0.0

    tokens_written: int = 0

    @property
    def total_load_ms(self) -> float:
        return self.gpu_load_ms + self.cpu_load_ms + self.ssd_load_ms

    @property
    def total_chunks(self) -> int:
        return self.gpu_chunks + self.cpu_chunks + self.ssd_chunks

    def as_dict(self) -> dict:
        return {
            "gpu_chunks": self.gpu_chunks,
            "cpu_chunks": self.cpu_chunks,
            "ssd_chunks": self.ssd_chunks,
            "gpu_load_ms": round(self.gpu_load_ms, 3),
            "cpu_load_ms": round(self.cpu_load_ms, 3),
            "ssd_load_ms": round(self.ssd_load_ms, 3),
            "total_load_ms": round(self.total_load_ms, 3),
        }


@dataclass
class _ChunkMeta:
    tier: KVTier = KVTier.GPU
    cpu_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ssd_path: Optional[Path] = None


class TieredKVStore:
    """Manages per-chunk tier assignments for a single ``full_draft_kv`` store.

    Parameters
    ----------
    full_draft_kv:
        The list of ``(key, value)`` tensors produced by
        ``SpecExtendDraftKVCache._allocate_tensor_kv``.  Shape per tensor is
        ``[1, num_kv_heads, max_length, head_dim]``.
    chunk_size:
        Token granularity of a retrieval chunk (must match the cache's
        ``chunk_size``).
    ssd_dir:
        Directory for SSD-tier serialised files.  Created automatically if
        absent.  Pass ``None`` to use a temp directory.
    """

    def __init__(
        self,
        full_draft_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        chunk_size: int = 32,
        ssd_dir: Optional[Path] = None,
    ) -> None:
        self.full_draft_kv = full_draft_kv
        self.chunk_size = chunk_size

        if ssd_dir is None:
            self._ssd_dir = Path(tempfile.mkdtemp(prefix="tiered_kv_"))
            self._owns_ssd_dir = True
        else:
            self._ssd_dir = Path(ssd_dir)
            self._ssd_dir.mkdir(parents=True, exist_ok=True)
            self._owns_ssd_dir = False

        self._chunks: Dict[int, _ChunkMeta] = {}

    def reset(self) -> None:
        """Clear all tier assignments (called when the KV cache is reset)."""
        for meta in self._chunks.values():
            if meta.ssd_path is not None and meta.ssd_path.exists():
                meta.ssd_path.unlink(missing_ok=True)
        self._chunks.clear()

    def tier_of(self, chunk_id: int) -> KVTier:
        return self._chunks.get(chunk_id, _ChunkMeta()).tier

    def set_tier(self, chunk_id: int, tier: KVTier) -> None:
        if chunk_id not in self._chunks:
            self._chunks[chunk_id] = _ChunkMeta(tier=KVTier.GPU)
        self._chunks[chunk_id].tier = tier

    def evict_to_cpu(self, chunk_id: int, total_seq_len: int) -> None:
        """Copy chunk data from GPU to CPU pinned memory and mark as CPU tier.

        Safe to call even if the chunk is already on CPU.
        """
        meta = self._chunks.setdefault(chunk_id, _ChunkMeta())
        if meta.tier == KVTier.CPU:
            return
        start, end = self._chunk_slice(chunk_id, total_seq_len)
        cpu_slices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for gpu_key, gpu_val in self.full_draft_kv:
            k = gpu_key[:, :, start:end, :].cpu().pin_memory()
            v = gpu_val[:, :, start:end, :].cpu().pin_memory()
            cpu_slices.append((k, v))
        meta.cpu_data = cpu_slices
        meta.tier = KVTier.CPU

    def evict_to_ssd(self, chunk_id: int, total_seq_len: int) -> None:
        """Serialise chunk data to SSD and mark as SSD tier.

        If already on CPU, re-uses that copy; otherwise reads from GPU.
        """
        meta = self._chunks.setdefault(chunk_id, _ChunkMeta())
        if meta.tier == KVTier.SSD:
            return

        start, end = self._chunk_slice(chunk_id, total_seq_len)
        if meta.tier == KVTier.CPU and meta.cpu_data is not None:
            layer_data = meta.cpu_data
        else:
            layer_data = [
                (gpu_key[:, :, start:end, :].cpu(), gpu_val[:, :, start:end, :].cpu())
                for gpu_key, gpu_val in self.full_draft_kv
            ]

        ssd_path = self._ssd_dir / f"chunk_{chunk_id:06d}.pt"
        torch.save(layer_data, ssd_path)

        meta.ssd_path = ssd_path
        meta.cpu_data = None
        meta.tier = KVTier.SSD

    def load_chunk_to_gpu(
        self,
        chunk_id: int,
        total_seq_len: int,
        target_key_list: List[torch.Tensor],
        target_val_list: List[torch.Tensor],
        target_start: int,
    ) -> Tuple[float, float, float, int]:
        """Ensure chunk data is accessible on GPU and copy into target buffers.

        ``target_key_list[layer_idx]`` and ``target_val_list[layer_idx]`` are
        the destination slices (pre-sized by the caller).  Returns
        ``(gpu_ms, cpu_ms, ssd_ms, actual_len)`` where ``actual_len`` is the
        number of tokens actually written (may be less than the current chunk
        slice for CPU/SSD chunks evicted when the tail chunk was smaller).

        For GPU-tier chunks the data is already in ``full_draft_kv``; this
        method still copies it into the target buffer for uniform downstream
        handling.
        """
        meta = self._chunks.get(chunk_id, _ChunkMeta())
        start, end = self._chunk_slice(chunk_id, total_seq_len)
        length = end - start
        gpu_ms = cpu_ms = ssd_ms = 0.0

        if meta.tier == KVTier.GPU:
            t0 = time.perf_counter()
            for layer_idx, (gpu_key, gpu_val) in enumerate(self.full_draft_kv):
                target_key_list[layer_idx][:, :, target_start:target_start + length, :].copy_(
                    gpu_key[:, :, start:end, :], non_blocking=True
                )
                target_val_list[layer_idx][:, :, target_start:target_start + length, :].copy_(
                    gpu_val[:, :, start:end, :], non_blocking=True
                )
            torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - t0) * 1000

        elif meta.tier == KVTier.CPU:
            assert meta.cpu_data is not None, f"chunk {chunk_id} marked CPU but cpu_data is None"
            t0 = time.perf_counter()
            for layer_idx, (cpu_k, cpu_v) in enumerate(meta.cpu_data):
                # Use actual stored size: the chunk may have grown since eviction (tail chunk).
                stored_len = cpu_k.shape[2]
                gpu_k = cpu_k.cuda(non_blocking=True)
                gpu_v = cpu_v.cuda(non_blocking=True)
                torch.cuda.synchronize()
                target_key_list[layer_idx][:, :, target_start:target_start + stored_len, :].copy_(gpu_k)
                target_val_list[layer_idx][:, :, target_start:target_start + stored_len, :].copy_(gpu_v)
            length = stored_len
            cpu_ms = (time.perf_counter() - t0) * 1000

        elif meta.tier == KVTier.SSD:
            assert meta.ssd_path is not None and meta.ssd_path.exists(), (
                f"chunk {chunk_id} marked SSD but file missing: {meta.ssd_path}"
            )
            t0 = time.perf_counter()
            layer_data: List[Tuple[torch.Tensor, torch.Tensor]] = torch.load(
                meta.ssd_path, map_location="cpu", weights_only=True
            )
            for layer_idx, (cpu_k, cpu_v) in enumerate(layer_data):
                # Use actual stored size: the chunk may have grown since eviction (tail chunk).
                stored_len = cpu_k.shape[2]
                gpu_k = cpu_k.cuda(non_blocking=True)
                gpu_v = cpu_v.cuda(non_blocking=True)
                torch.cuda.synchronize()
                target_key_list[layer_idx][:, :, target_start:target_start + stored_len, :].copy_(gpu_k)
                target_val_list[layer_idx][:, :, target_start:target_start + stored_len, :].copy_(gpu_v)
            length = stored_len
            ssd_ms = (time.perf_counter() - t0) * 1000

        return gpu_ms, cpu_ms, ssd_ms, length

    def load_chunks(
        self,
        chunk_ids: List[int],
        total_seq_len: int,
        token_indices: List[int],
        working_key_list: List[torch.Tensor],
        working_val_list: List[torch.Tensor],
    ) -> KVLoadMetrics:
        """Load all requested chunks into working cache buffers.

        ``working_key_list[layer_idx]`` has shape
        ``[1, num_kv_heads, max_length, head_dim]``.  Data is written starting
        at position ``token_indices[0]`` relative to the chunk boundaries.

        Returns a ``KVLoadMetrics`` summarising load times per tier.
        """
        metrics = KVLoadMetrics()

        target_pos = 0
        for chunk_id in sorted(chunk_ids):
            gpu_ms, cpu_ms, ssd_ms, actual_len = self.load_chunk_to_gpu(
                chunk_id,
                total_seq_len,
                working_key_list,
                working_val_list,
                target_pos,
            )
            tier = self.tier_of(chunk_id)
            if tier == KVTier.GPU:
                metrics.gpu_chunks += 1
                metrics.gpu_load_ms += gpu_ms
            elif tier == KVTier.CPU:
                metrics.cpu_chunks += 1
                metrics.cpu_load_ms += cpu_ms
            else:
                metrics.ssd_chunks += 1
                metrics.ssd_load_ms += ssd_ms
            target_pos += actual_len

        metrics.tokens_written = target_pos
        return metrics

    def tier_summary(self, total_chunks: int) -> Dict[str, int]:
        """Return counts of chunks per tier for diagnostics."""
        counts: Dict[str, int] = {KVTier.GPU: 0, KVTier.CPU: 0, KVTier.SSD: 0}
        for cid in range(total_chunks):
            counts[self.tier_of(cid)] += 1
        return {t.value: v for t, v in counts.items()}

    def _chunk_slice(self, chunk_id: int, total_seq_len: int) -> Tuple[int, int]:
        start = chunk_id * self.chunk_size
        end = min(start + self.chunk_size, total_seq_len)
        return start, end
