"""Backend interfaces for edge-cloud SpecExtend.

SpecExtend uses backend-visible KV caches, linear draft sequences, and
target-attention-driven retrieval selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class DraftSequence:
    input_ids: List[int]
    position_ids: List[int]
    parent_indices: Optional[List[int]] = None
    attention_mask: Optional[List[List[int]]] = None


@dataclass
class DraftResult:
    draft: DraftSequence
    draft_time_ms: float
    appended_kv_tokens: int


class SpecExtendDraftBackend(Protocol):
    tokenizer: object

    def build_draft(
        self,
        verified_prefix: List[int],
        correction_token_id: Optional[int],
        draft_length: int,
        retrieval_chunk_ids: Optional[List[int]] = None,
        retrieval_selection_updated: bool = False,
        retrieval_selection_base_len: Optional[int] = None,
    ) -> DraftResult:
        """Build a linear draft sequence for the current verified prefix."""


class SpecExtendTargetBackend(Protocol):
    def verify(self, request):
        """Verify a SpecExtend request and return a SpecExtendResponse."""
