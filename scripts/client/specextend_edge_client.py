"""Edge client for the full SpecExtend protocol.

This client is backend-agnostic. It coordinates draft-tree construction,
cloud tree verification, and target-attention driven retrieval selection. A
real SpecExtend backend must implement ``SpecExtendDraftBackend``.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from core.protocol import SpecExtendTreeRequest, SpecExtendTreeResponse
from core.specextend_backend import DraftTree, DraftTreeResult, SpecExtendDraftBackend
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
        self.pipeline_enabled = os.getenv("SPECEXTEND_ASYNC_PIPELINE", "1").strip() not in {"0", "false", "False"}
        self.pipeline_offsets = self._parse_pipeline_offsets(os.getenv("SPECEXTEND_PIPELINE_OFFSETS", "full,half"))

    def generate(self, prompt: str) -> SpecExtendRequestMetrics:
        request_id = str(uuid.uuid4())
        metrics = SpecExtendRequestMetrics(request_id=request_id, prompt=prompt)
        prompt_ids = list(self.tokenizer.encode(prompt))
        verified_prefix = list(prompt_ids)
        correction_token_id: Optional[int] = None
        prefetched = None
        retrieval_recorded_len = 0
        wall_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=1) as executor:
            while len(verified_prefix) - len(prompt_ids) < self.max_new_tokens:
                if self.eos_token_id and self.eos_token_id in verified_prefix[len(prompt_ids) :]:
                    break

                new_prefix_tokens = len(verified_prefix) - retrieval_recorded_len
                if new_prefix_tokens > 0:
                    self.retrieval.append_tokens(new_prefix_tokens)
                    retrieval_recorded_len = len(verified_prefix)

                retrieve_this_round = (
                    self.retrieve_every_n_steps > 0
                    and metrics.total_rounds % self.retrieve_every_n_steps == 0
                    and metrics.total_rounds > 0
                )
                retrieval_indices = self.retrieval.selected_token_indices()
                pipeline_hit = False
                pipeline_built = 0
                pipeline_wait_ms = 0.0

                if prefetched and prefetched["prefix"] == verified_prefix:
                    draft_result = prefetched["draft"]
                    correction_token_id = None
                    pipeline_hit = True
                else:
                    draft_result = self.draft_backend.build_draft_tree(
                        verified_prefix=list(verified_prefix),
                        correction_token_id=correction_token_id,
                        nodes=self.nodes,
                        threshold=self.threshold,
                        max_depth=self.max_depth,
                        retrieval_token_indices=retrieval_indices,
                    )
                prefetched = None

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
                if self.pipeline_enabled:
                    future = executor.submit(self.cloud_verify_tree, request)
                    candidates, pipeline_built = self._build_pipeline_candidates(
                        future=future,
                        base_prefix=verified_prefix,
                        draft_result=draft_result,
                        retrieval_indices=retrieval_indices,
                    )
                    wait_start = time.perf_counter()
                    response = future.result()
                    pipeline_wait_ms = (time.perf_counter() - wait_start) * 1000
                else:
                    candidates = []
                    response = self.cloud_verify_tree(request)
                verify_elapsed_ms = (time.perf_counter() - verify_start) * 1000

                accepted_tokens = [
                    draft_result.tree.input_ids[idx]
                    for idx in response.accepted_tree_indices
                    if 0 <= idx < len(draft_result.tree.input_ids)
                ]
                previous_prefix = list(verified_prefix)
                verified_prefix.extend(accepted_tokens)
                correction_token_id = response.correction_token_id
                if correction_token_id is not None:
                    verified_prefix.append(correction_token_id)

                new_prefix_tokens = len(verified_prefix) - retrieval_recorded_len
                if new_prefix_tokens > 0:
                    self.retrieval.append_tokens(new_prefix_tokens)
                    retrieval_recorded_len = len(verified_prefix)

                prefetched = self._select_pipeline_candidate(
                    candidates=candidates,
                    previous_prefix=previous_prefix,
                    accepted_len=response.accepted_len,
                    correction_token_id=correction_token_id,
                    verified_prefix=verified_prefix,
                )

                if response.target_attn_scores is not None:
                    selected = self.retrieval.select_by_attention(response.target_attn_scores)
                    metrics.retrieval_updates += 1
                    metrics.selected_chunk_ids = [chunk.chunk_id for chunk in selected]
                elif response.selected_chunk_ids is not None:
                    self.retrieval.set_selected_chunk_ids(response.selected_chunk_ids)
                    metrics.selected_chunk_ids = self.retrieval.selected_chunk_ids()

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
                        "pipeline_hit": pipeline_hit,
                        "pipeline_built": pipeline_built,
                        "pipeline_wait_ms": pipeline_wait_ms,
                        "pipeline_reuse": bool(prefetched),
                    }
                )

                if not accepted_tokens and correction_token_id is None:
                    logger.warning("SpecExtend made no progress; stopping request %s", request_id)
                    break

        metrics.total_latency_ms = (time.perf_counter() - wall_start) * 1000
        metrics.generated_tokens = len(verified_prefix) - len(prompt_ids)
        return metrics

    @staticmethod
    def _parse_pipeline_offsets(value: str) -> List[str]:
        offsets = []
        for item in value.split(","):
            item = item.strip().lower()
            if item:
                offsets.append(item)
        return offsets or ["full"]

    def _candidate_offsets(self, draft_len: int) -> List[int]:
        offsets = []
        for item in self.pipeline_offsets:
            if item == "full":
                offset = draft_len
            elif item == "half":
                offset = max(0, draft_len // 2)
            elif item == "zero":
                offset = 0
            else:
                try:
                    offset = int(item)
                except ValueError:
                    continue
            offsets.append(max(0, min(draft_len, offset)))
        return sorted(set(offsets), reverse=True)

    def _build_pipeline_candidates(
        self,
        future,
        base_prefix: List[int],
        draft_result: DraftTreeResult,
        retrieval_indices: List[int],
    ):
        if not getattr(self.draft_backend, "supports_pipeline_candidates", True):
            return [], 0
        if not hasattr(self.draft_backend, "build_draft_candidate"):
            return [], 0

        candidates = []
        for offset in self._candidate_offsets(len(draft_result.tree.input_ids)):
            if future.done():
                break
            candidate_prefix = list(base_prefix) + draft_result.tree.input_ids[:offset]
            candidate = self.draft_backend.build_draft_candidate(
                verified_prefix=candidate_prefix,
                nodes=self.nodes,
                max_depth=self.max_depth,
                retrieval_token_indices=retrieval_indices,
            )
            candidates.append(
                {
                    "offset": offset,
                    "prefix": candidate_prefix,
                    "draft": candidate,
                }
            )
        return candidates, len(candidates)

    def _select_pipeline_candidate(
        self,
        candidates,
        previous_prefix: List[int],
        accepted_len: int,
        correction_token_id: Optional[int],
        verified_prefix: List[int],
    ):
        for candidate in candidates:
            if candidate["offset"] != accepted_len:
                continue
            candidate_draft = candidate["draft"]
            input_ids = list(candidate_draft.tree.input_ids)
            if correction_token_id is None:
                return {"prefix": list(verified_prefix), "draft": candidate_draft}
            if not input_ids or input_ids[0] != correction_token_id:
                continue
            adjusted_ids = input_ids[1:]
            if not adjusted_ids:
                continue
            parent_indices = [idx - 1 for idx in range(len(adjusted_ids))]
            adjusted_tree = DraftTree(
                input_ids=adjusted_ids,
                position_ids=list(range(len(verified_prefix), len(verified_prefix) + len(adjusted_ids))),
                parent_indices=parent_indices,
                attention_mask=self.draft_backend._tree_attention_mask(parent_indices),
            )
            return {
                "prefix": list(verified_prefix),
                "draft": DraftTreeResult(
                    tree=adjusted_tree,
                    draft_time_ms=0.0,
                    appended_kv_tokens=0,
                ),
            }
        return None
