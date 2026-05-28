"""Edge client for the full SpecExtend protocol.

This client is backend-agnostic. It coordinates draft-tree construction,
cloud tree verification, and target-attention driven retrieval selection. A
real SpecExtend backend must implement ``SpecExtendDraftBackend``; the current
vLLM draft generator does not expose enough cache control for this path.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from core.protocol import SpecExtendTreeRequest, SpecExtendTreeResponse
from core.specextend_backend import SpecExtendDraftBackend
from core.specextend_retrieval import SpecExtendRetrievalState

logger = logging.getLogger(__name__)


@dataclass
class SpecExtendRequestMetrics:
    request_id: str
    prompt: str
    total_latency_ms: float = 0.0
    generated_tokens: int = 0
    total_rounds: int = 0
    total_edge_draft_time_ms: float = 0.0
    total_server_verify_time_ms: float = 0.0
    total_network_time_ms: float = 0.0
    total_accepted_tokens: int = 0
    retrieval_updates: int = 0
    selected_chunk_ids: List[int] = field(default_factory=list)
    round_details: List[Dict[str, object]] = field(default_factory=list)


class SpecExtendEdgeClient:
    def __init__(
        self,
        draft_backend: SpecExtendDraftBackend,
        cloud_verify_tree: Callable[[SpecExtendTreeRequest], SpecExtendTreeResponse],
        max_new_tokens: int = 128,
        nodes: int = 50,
        threshold: float = 0.7,
        max_depth: int = 10,
        retrieval_chunk_size: int = 32,
        retrieve_top_k: int = 32,
        retrieve_every_n_steps: int = 8,
        eos_token_id: Optional[int] = None,
    ):
        self.draft_backend = draft_backend
        self.cloud_verify_tree = cloud_verify_tree
        self.max_new_tokens = max_new_tokens
        self.nodes = nodes
        self.threshold = threshold
        self.max_depth = max_depth
        self.retrieve_every_n_steps = retrieve_every_n_steps
        self.retrieval = SpecExtendRetrievalState(
            chunk_size=retrieval_chunk_size,
            top_k_chunks=retrieve_top_k,
        )
        self.tokenizer = draft_backend.tokenizer
        self.eos_token_id = eos_token_id
        if self.eos_token_id is None and self.tokenizer:
            self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

    def generate(self, prompt: str) -> SpecExtendRequestMetrics:
        request_id = str(uuid.uuid4())
        metrics = SpecExtendRequestMetrics(request_id=request_id, prompt=prompt)
        prompt_ids = list(self.tokenizer.encode(prompt))
        verified_prefix = list(prompt_ids)
        correction_token_id: Optional[int] = None
        wall_start = time.perf_counter()

        while len(verified_prefix) - len(prompt_ids) < self.max_new_tokens:
            if self.eos_token_id and self.eos_token_id in verified_prefix[len(prompt_ids) :]:
                break

            retrieve_this_round = (
                metrics.total_rounds % self.retrieve_every_n_steps == 0
                and metrics.total_rounds > 0
            )
            retrieval_indices = self.retrieval.selected_token_indices()
            draft_result = self.draft_backend.build_draft_tree(
                verified_prefix=list(verified_prefix),
                correction_token_id=correction_token_id,
                nodes=self.nodes,
                threshold=self.threshold,
                max_depth=self.max_depth,
                retrieval_token_indices=retrieval_indices,
            )
            self.retrieval.append_tokens(draft_result.appended_kv_tokens)

            request = SpecExtendTreeRequest(
                request_id=request_id,
                prefix_ids=list(verified_prefix),
                tree_input_ids=draft_result.tree.input_ids,
                tree_position_ids=draft_result.tree.position_ids,
                parent_indices=draft_result.tree.parent_indices,
                tree_attention_mask=draft_result.tree.attention_mask,
                retrieve_attn_scores=retrieve_this_round,
                retrieval_chunk_size=self.retrieval.chunk_size,
                retrieve_top_k=self.retrieval.top_k_chunks,
                metadata={"round": metrics.total_rounds},
            )

            verify_start = time.perf_counter()
            response = self.cloud_verify_tree(request)
            verify_elapsed_ms = (time.perf_counter() - verify_start) * 1000

            accepted_tokens = [
                draft_result.tree.input_ids[idx]
                for idx in response.accepted_tree_indices
                if 0 <= idx < len(draft_result.tree.input_ids)
            ]
            verified_prefix.extend(accepted_tokens)
            correction_token_id = response.correction_token_id
            if correction_token_id is not None:
                verified_prefix.append(correction_token_id)

            if response.target_attn_scores is not None:
                selected = self.retrieval.select_by_attention(response.target_attn_scores)
                metrics.retrieval_updates += 1
                metrics.selected_chunk_ids = [chunk.chunk_id for chunk in selected]
            elif response.selected_chunk_ids is not None:
                self.retrieval.set_selected_chunk_ids(response.selected_chunk_ids)
                metrics.selected_chunk_ids = list(response.selected_chunk_ids)

            metrics.total_rounds += 1
            metrics.total_edge_draft_time_ms += draft_result.draft_time_ms
            metrics.total_server_verify_time_ms += response.server_verify_time_ms
            metrics.total_network_time_ms += max(0.0, verify_elapsed_ms - response.server_verify_time_ms)
            metrics.total_accepted_tokens += response.accepted_len
            metrics.round_details.append(
                {
                    "round": metrics.total_rounds - 1,
                    "tree_nodes": len(draft_result.tree.input_ids),
                    "accepted_len": response.accepted_len,
                    "retrieval": retrieve_this_round,
                    "selected_chunk_ids": list(metrics.selected_chunk_ids),
                }
            )

            if not accepted_tokens and correction_token_id is None:
                logger.warning("SpecExtend made no progress; stopping request %s", request_id)
                break

        metrics.total_latency_ms = (time.perf_counter() - wall_start) * 1000
        metrics.generated_tokens = len(verified_prefix) - len(prompt_ids)
        return metrics
