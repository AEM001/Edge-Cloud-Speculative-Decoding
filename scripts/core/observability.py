"""Small helpers for SpecExtend observability records."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


def now_ms() -> float:
    return time.time() * 1000.0


def json_size_bytes(data: Dict[str, Any]) -> int:
    return len(json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def jaccard(left: Sequence[int], right: Sequence[int]) -> Optional[float]:
    a = set(int(v) for v in left)
    b = set(int(v) for v in right)
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_count_for_chunks(chunk_ids: Iterable[int], chunk_size: int, total_tokens: int) -> int:
    total = 0
    for chunk_id in set(int(v) for v in chunk_ids):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, total_tokens)
        if end > start:
            total += end - start
    return total


def attention_mass_for_chunks(scores: Sequence[float], chunk_ids: Iterable[int], chunk_size: int) -> float:
    mass = 0.0
    for chunk_id in set(int(v) for v in chunk_ids):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, len(scores))
        if end > start:
            mass += sum(float(v) for v in scores[start:end])
    return mass


def logits_uncertainty_from_probs(probs) -> Dict[str, Optional[float]]:
    """Return uncertainty metrics from a 1-D torch probability tensor."""
    top_values = probs.topk(k=min(2, probs.numel())).values.detach().float().cpu().tolist()
    top1 = float(top_values[0]) if top_values else None
    top2 = float(top_values[1]) if len(top_values) > 1 else None
    margin = (top1 - top2) if top1 is not None and top2 is not None else None
    entropy = float((-(probs.float() * probs.float().clamp_min(1e-12).log()).sum()).detach().cpu().item())
    return {
        "draft_entropy": entropy if math.isfinite(entropy) else None,
        "draft_top1_confidence": top1,
        "draft_top1_top2_margin": margin,
    }


@dataclass
class EdgeObservability:
    prefix_len: int = 0
    generated_token_offset: int = 0
    generation_phase: str = "unknown"
    selected_chunk_ids: List[int] = field(default_factory=list)
    selected_token_count: int = 0
    full_kv_tokens: int = 0
    working_kv_tokens: int = 0
    selected_full_ratio: Optional[float] = None
    appended_kv_tokens: int = 0
    draft_compute_ms: float = 0.0
    kv_append_ms: float = 0.0
    kv_select_ms: float = 0.0
    tree_construct_ms: float = 0.0
    postprocess_ms: float = 0.0
    draft_entropy: Optional[float] = None
    draft_top1_confidence: Optional[float] = None
    draft_top1_top2_margin: Optional[float] = None
    draft_tree_nodes: int = 0
    draft_tree_actual_depth: int = 0
    cuda_memory_allocated_mb: Optional[float] = None
    cuda_memory_reserved_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CloudObservability:
    cloud_receive_ts_ms: float = 0.0
    cloud_finish_ts_ms: float = 0.0
    prefix_len: int = 0
    draft_tokens: int = 0
    accepted_len: int = 0
    rejected_position: Optional[int] = None
    accepted_indices: List[int] = field(default_factory=list)
    correction_token_id: Optional[int] = None
    target_verify_time_ms: float = 0.0
    guidance_generation_time_ms: float = 0.0
    top_attention_token_indices: List[int] = field(default_factory=list)
    selected_chunk_ids: List[int] = field(default_factory=list)
    selected_count: int = 0
    selected_token_count: int = 0
    total_attention_tokens: int = 0
    attention_mass_covered: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
