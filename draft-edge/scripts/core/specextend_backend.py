"""Backend interfaces for edge-cloud SpecExtend.

Full SpecExtend needs backend-visible KV caches, tree masks, and target
attention scores, so this module defines the contract expected by the
edge-cloud glue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class DraftTree:
    input_ids: List[int]
    position_ids: List[int]
    parent_indices: List[int]
    attention_mask: List[List[int]]


@dataclass
class DraftTreeResult:
    tree: DraftTree
    draft_time_ms: float
    appended_kv_tokens: int


class SpecExtendDraftBackend(Protocol):
    tokenizer: object

    def build_draft_tree(
        self,
        verified_prefix: List[int],
        correction_token_id: Optional[int],
        nodes: int,
        threshold: float,
        max_depth: int,
        retrieval_token_indices: Optional[List[int]] = None,
    ) -> DraftTreeResult:
        """Grow a SpecExtend draft tree for the current verified prefix."""


class SpecExtendTargetBackend(Protocol):
    def verify_tree(self, request):
        """Verify a SpecExtend tree request and return SpecExtendTreeResponse."""
