"""Backend interfaces for edge-cloud SpecExtend.

SpecExtend uses backend-visible KV caches, linear draft sequences, and
target-attention-driven retrieval selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Protocol


@dataclass
class DraftSequence:
    input_ids: List[int]
    position_ids: List[int]


@dataclass
class DraftResult:
    draft: DraftSequence
    draft_time_ms: float
    appended_kv_tokens: int
    kv_load_metrics: Optional[Any] = field(default=None)


class SpecExtendDraftBackend(Protocol):
    tokenizer: object

    def build_draft(
        self,
        verified_prefix: List[int],
        correction_token_id: Optional[int],
        nodes: int,
        threshold: float,
        max_depth: int,
        retrieval_chunk_ids: Optional[List[int]] = None,
    ) -> DraftResult:
        """Build a linear draft sequence for the current verified prefix."""


class SpecExtendTargetBackend(Protocol):
    def verify(self, request):
        """Verify a SpecExtend request and return a SpecExtendResponse."""
