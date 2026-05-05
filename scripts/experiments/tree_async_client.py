"""Tree-based speculative pipeline client used by quick_test.

This module keeps the tree-specific pipeline logic isolated from the
benchmark driver so it can be imported elsewhere without dragging in
all of quick_test's orchestration code.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable, List

from client.async_edge_client import AsyncEdgeClient, AsyncRequestMetrics, _VerifyResult
from core.protocol import DraftRequest, EdgeRequest


logger = logging.getLogger(__name__)


@dataclass
class TreeBranch:
    branch_id: int
    offset: int
    prefix: List[int]
    draft_ids: List[int]
    draft_logprobs: List[float]
    draft_time_ms: float


class TreeAsyncEdgeClient(AsyncEdgeClient):
    """Async client that explores multiple speculative branches per round."""

    HISTORY_WINDOW: int = 50
    BRANCH_WIDTH: int = 3

    def __init__(self, *args, branch_width: int = BRANCH_WIDTH, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch_width = max(1, branch_width)
        self._history: deque[int] = deque(maxlen=self.HISTORY_WINDOW)

    def generate(
        self,
        prompt: str,
        policy: Callable,
        policy_name: str = "Tree",
    ) -> AsyncRequestMetrics:
        request_id = str(uuid.uuid4())
        metrics = AsyncRequestMetrics(request_id=request_id, prompt=prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        committed_prefix: List[int] = list(prompt_ids)
        speculative_prefix: List[int] = list(prompt_ids)
        result_queue: queue.Queue = queue.Queue()
        wall_start = time.perf_counter()

        try:
            self._tree_pipeline_loop(
                prompt_ids=prompt_ids,
                committed_prefix=committed_prefix,
                speculative_prefix=speculative_prefix,
                result_queue=result_queue,
                policy=policy,
                policy_name=policy_name,
                request_id=request_id,
                metrics=metrics,
            )
        finally:
            metrics.total_latency_ms = (time.perf_counter() - wall_start) * 1000
            metrics.generated_tokens = len(committed_prefix) - len(prompt_ids)
            metrics.compute_derived()

        logger.info(
            "Tree generation done: %d tokens in %.0fms  %.2f tok/s  "
            "pipeline_eff=%.1f%%  rollbacks=%d  bubble=%.1fms",
            metrics.generated_tokens,
            metrics.total_latency_ms,
            metrics.tokens_per_second,
            100 * metrics.pipeline_efficiency,
            metrics.total_rollbacks,
            metrics.avg_bubble_ms,
        )
        return metrics

    def _tree_offsets(self, k: int, spec_ahead: int) -> List[int]:
        if spec_ahead <= 0:
            return [0]
        max_offset = min(k, spec_ahead)
        if not self._history:
            seeds = [max(0, k - 2), max(0, k - 1), k]
        else:
            sorted_h = sorted(self._history)
            seeds = []
            for p in (0.10, 0.50, 0.90):
                idx = int(math.floor(p * (len(sorted_h) - 1)))
                seeds.append(sorted_h[max(0, min(idx, len(sorted_h) - 1))])

        offsets: List[int] = []
        for offset in seeds:
            clipped = max(0, min(max_offset, int(offset)))
            if clipped not in offsets:
                offsets.append(clipped)

        center = max(0, min(max_offset, int(round(sum(seeds) / len(seeds)))))
        delta = 1
        while len(offsets) < self.branch_width and delta <= k:
            for candidate in (center - delta, center + delta):
                clipped = max(0, min(max_offset, candidate))
                if clipped not in offsets:
                    offsets.append(clipped)
                if len(offsets) >= self.branch_width:
                    break
            delta += 1

        return offsets[:self.branch_width]

    def _tree_pipeline_loop(
        self,
        prompt_ids,
        committed_prefix,
        speculative_prefix,
        result_queue,
        policy,
        policy_name,
        request_id,
        metrics,
    ):
        max_tokens = self.max_new_tokens
        eos = self.eos_token_id
        next_tree_id = 0

        def _done():
            if len(committed_prefix) - len(prompt_ids) >= max_tokens:
                return True
            if eos and eos in committed_prefix[len(prompt_ids):]:
                return True
            return False

        while not _done():
            if metrics.total_rounds >= max_tokens * 4:
                logger.error(
                    "Tree exceeded safety round cap: rounds=%d committed_generated=%d max_tokens=%d",
                    metrics.total_rounds,
                    len(committed_prefix) - len(prompt_ids),
                    max_tokens,
                )
                break

            tree_id = next_tree_id
            next_tree_id += 1
            k = policy(tree_id, [])
            branches: List[TreeBranch] = []
            base_t0 = time.perf_counter()
            base_resp = self.draft_generator.generate_draft_tokens(
                DraftRequest(verified_prefix=list(committed_prefix), num_draft_tokens=k),
                temperature=self.temperature,
            )
            base_draft_ms = (time.perf_counter() - base_t0) * 1000
            if not base_resp.draft_token_ids:
                break
            base_draft = list(base_resp.draft_token_ids)
            branches.append(
                TreeBranch(
                    branch_id=tree_id * 100,
                    offset=0,
                    prefix=list(committed_prefix),
                    draft_ids=base_draft,
                    draft_logprobs=base_resp.logprobs,
                    draft_time_ms=base_draft_ms,
                )
            )

            offsets = self._tree_offsets(k, len(base_draft))
            branch_offsets = [o for o in offsets if o > 0][: max(0, self.branch_width - 1)]
            branch_prefixes = []
            for offset in branch_offsets:
                prefix = list(committed_prefix)
                prefix.extend(base_draft[:offset])
                branch_prefixes.append((offset, prefix))

            batch_t0 = time.perf_counter()
            if branch_prefixes and hasattr(self.draft_generator, "generate_draft_tokens_batch"):
                draft_resps = self.draft_generator.generate_draft_tokens_batch(
                    [
                        DraftRequest(verified_prefix=list(prefix), num_draft_tokens=k)
                        for _, prefix in branch_prefixes
                    ],
                    temperature=self.temperature,
                )
                batch_draft_ms = (time.perf_counter() - batch_t0) * 1000
            else:
                draft_resps = []
                for _, prefix in branch_prefixes:
                    draft_resps.append(
                        self.draft_generator.generate_draft_tokens(
                            DraftRequest(verified_prefix=list(prefix), num_draft_tokens=k),
                            temperature=self.temperature,
                        )
                    )
                batch_draft_ms = (time.perf_counter() - batch_t0) * 1000

            per_branch_draft_ms = batch_draft_ms / len(draft_resps) if draft_resps else 0.0
            for branch_index, ((offset, prefix), draft_resp) in enumerate(
                zip(branch_prefixes, draft_resps),
                start=1,
            ):
                if not draft_resp.draft_token_ids:
                    continue
                branches.append(
                    TreeBranch(
                        branch_id=tree_id * 100 + branch_index,
                        offset=offset,
                        prefix=list(prefix),
                        draft_ids=draft_resp.draft_token_ids,
                        draft_logprobs=draft_resp.logprobs,
                        draft_time_ms=per_branch_draft_ms,
                    )
                )

            if not branches:
                break

            metrics.total_rounds += 1
            metrics.total_drafted_tokens += sum(len(b.draft_ids) for b in branches)
            metrics.total_edge_draft_time_ms += sum(b.draft_time_ms for b in branches)

            edge_reqs = []
            for branch in branches:
                edge_req = EdgeRequest(
                    request_id=request_id,
                    round_id=branch.branch_id,
                    prefix_ids=list(branch.prefix),
                    draft_ids=branch.draft_ids,
                    draft_logprobs=branch.draft_logprobs,
                    edge_draft_time_ms=branch.draft_time_ms,
                    policy_metadata={
                        "policy_name": policy_name,
                        "K": k,
                        "tree_id": tree_id,
                        "branch_offset": branch.offset,
                        "branch_width": len(branches),
                    },
                )
                metrics.uplink_bytes += len(json.dumps(edge_req.to_dict()).encode())
                edge_reqs.append(edge_req)

            bubble_t0 = time.perf_counter()
            try:
                if hasattr(self.cloud_client, "verify_batch"):
                    responses = self.cloud_client.verify_batch(edge_reqs)
                else:
                    responses = [self.cloud_client(edge_req) for edge_req in edge_reqs]
                branch_results = [
                    (branch, _VerifyResult(branch.branch_id, resp, resp.rtt_ms or 0.0))
                    for branch, resp in zip(branches, responses)
                ]
            except Exception as exc:
                logger.error("Tree verifier batch error: %s", exc)
                branch_results = []
            metrics.total_bubble_ms += (time.perf_counter() - bubble_t0) * 1000
            if len(branch_results) < len(branches):
                try:
                    while len(branch_results) < len(branches):
                        branch_results.append(result_queue.get_nowait())
                except queue.Empty:
                    pass

            if not branch_results:
                break

            for branch, vr in branch_results:
                resp = vr.cloud_response
                self._history.append(resp.accepted_len)
                metrics.total_server_verify_time_ms += resp.server_verify_time_ms
                metrics.total_network_time_ms += max(0.0, vr.rtt_ms - resp.server_verify_time_ms)
                metrics.downlink_bytes += len(json.dumps(resp.to_dict()).encode())

            base_branch, base_vr = next(
                ((branch, vr) for branch, vr in branch_results if branch.offset == 0),
                min(branch_results, key=lambda item: item[0].offset),
            )
            base_accept = base_vr.cloud_response.accepted_len
            valid_results = [
                (branch, vr)
                for branch, vr in branch_results
                if branch.offset <= base_accept
            ]
            selected_branch, selected_vr = max(
                valid_results,
                key=lambda item: item[0].offset + item[1].cloud_response.accepted_len,
            )
            selected_resp = selected_vr.cloud_response
            selected_progress = selected_branch.offset + selected_resp.accepted_len
            new_committed = list(committed_prefix)
            new_committed.extend(base_draft[: selected_branch.offset])
            new_committed.extend(selected_resp.accepted_token_ids)
            if selected_resp.correction_token_id is not None:
                new_committed.append(selected_resp.correction_token_id)

            total_drafted = sum(len(b.draft_ids) for b in branches)
            metrics.total_accepted_tokens += selected_progress
            if selected_progress < k:
                metrics.total_rollbacks += 1
            metrics.slot_details.append(
                {
                    "slot_id": selected_branch.branch_id,
                    "drafted": total_drafted,
                    "accepted": selected_progress,
                    "full_hit": selected_progress >= k,
                    "wasted_tokens": max(0, total_drafted - selected_progress),
                    "draft_ms": sum(b.draft_time_ms for b in branches),
                    "verify_ms": sum(
                        vr.cloud_response.server_verify_time_ms for _, vr in branch_results
                    ),
                    "rtt_ms": max(vr.rtt_ms for _, vr in branch_results),
                    "rollback": selected_progress < k,
                    "tree_branch_width": len(branches),
                    "selected_offset": selected_branch.offset,
                    "stale_branches": len(branches) - 1,
                    "base_accepted": base_accept,
                }
            )
            committed_prefix.clear()
            committed_prefix.extend(new_committed)

            metrics.average_rtt_ms = (
                metrics.average_rtt_ms * (metrics.total_rounds - 1)
                + max(vr.rtt_ms for _, vr in branch_results)
            ) / metrics.total_rounds
            speculative_prefix.clear()
            speculative_prefix.extend(committed_prefix)
