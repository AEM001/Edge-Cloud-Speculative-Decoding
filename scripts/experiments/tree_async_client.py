"""Tree-based speculative pipeline client used by quick_test.

This module keeps the tree-specific pipeline logic isolated from the
benchmark driver so it can be imported elsewhere without dragging in
all of quick_test's orchestration code.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections import Counter, deque
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
    """Async client that drafts tree branches as next-round prefetch only."""

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
            center = max(0, min(max_offset, int(round(0.6 * k))))
            seeds = [center, center - 1, center + 1]
        else:
            counts = Counter(self._history)
            mode = counts.most_common(1)[0][0]
            seeds = [mode, mode - 1, mode + 1]

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

    def _prefetch_from_branch(
        self,
        branch: TreeBranch,
        base_draft: List[int],
        base_accepted: int,
        correction: int | None,
    ) -> tuple[List[int], List[float]] | None:
        """Return reusable continuation tokens if a local branch matches reality."""
        if branch.offset > base_accepted:
            return None

        expected_prefix = list(base_draft[branch.offset:base_accepted])
        if correction is not None:
            expected_prefix.append(correction)

        if len(branch.draft_ids) < len(expected_prefix):
            return None
        if branch.draft_ids[:len(expected_prefix)] != expected_prefix:
            return None

        reuse_start = len(expected_prefix)
        return (
            list(branch.draft_ids[reuse_start:]),
            list(branch.draft_logprobs[reuse_start:]),
        )

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
        prefetched_ids: List[int] = []
        prefetched_logprobs: List[float] = []

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
            if prefetched_ids:
                base_draft = prefetched_ids[:k]
                base_logprobs = prefetched_logprobs[:len(base_draft)]
                base_draft_ms = 0.0
                prefetched_ids = prefetched_ids[len(base_draft):]
                prefetched_logprobs = prefetched_logprobs[len(base_draft):]
            else:
                base_t0 = time.perf_counter()
                base_resp = self.draft_generator.generate_draft_tokens(
                    DraftRequest(verified_prefix=list(committed_prefix), num_draft_tokens=k),
                    temperature=self.temperature,
                )
                base_draft_ms = (time.perf_counter() - base_t0) * 1000
                if not base_resp.draft_token_ids:
                    break
                base_draft = list(base_resp.draft_token_ids)
                base_logprobs = list(base_resp.logprobs)
            branches.append(
                TreeBranch(
                    branch_id=tree_id * 100,
                    offset=0,
                    prefix=list(committed_prefix),
                    draft_ids=base_draft,
                    draft_logprobs=base_logprobs,
                    draft_time_ms=base_draft_ms,
                )
            )

            # Fire only the baseline verify in the background. Tree branches
            # are local prefetch candidates for the next round; they are never
            # sent to the verifier in this round.
            base_edge_req = EdgeRequest(
                request_id=request_id,
                round_id=tree_id * 100,
                prefix_ids=list(committed_prefix),
                draft_ids=base_draft,
                draft_logprobs=base_logprobs,
                edge_draft_time_ms=base_draft_ms,
                policy_metadata={"policy_name": policy_name, "K": k,
                                  "tree_id": tree_id, "branch_offset": 0},
            )
            base_verify_result: List = []   # filled by thread

            def _run_base_verify(req=base_edge_req, out=base_verify_result):
                try:
                    resp = self.cloud_client(req)
                    out.append(resp)
                except Exception as exc:
                    logger.error("Base verify error: %s", exc)

            verify_thread = threading.Thread(target=_run_base_verify, daemon=True)
            verify_thread.start()

            # Draft speculative branches while baseline verify is in flight.
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
                verify_thread.join(timeout=120)
                break

            metrics.total_rounds += 1
            metrics.total_drafted_tokens += len(base_draft)
            metrics.total_edge_draft_time_ms += base_draft_ms + batch_draft_ms

            bubble_t0 = time.perf_counter()
            verify_thread.join(timeout=120)
            base_wait_ms = (time.perf_counter() - bubble_t0) * 1000
            total_wait_ms = (time.perf_counter() - bubble_t0) * 1000
            metrics.total_bubble_ms += total_wait_ms

            if not base_verify_result:
                break
            base_resp_cloud = base_verify_result[0]
            base_accepted = base_resp_cloud.accepted_len
            correction = base_resp_cloud.correction_token_id
            branch_results = [
                (branches[0], _VerifyResult(branches[0].branch_id, base_resp_cloud,
                                            base_resp_cloud.rtt_ms or 0.0))
            ]
            metrics.uplink_bytes += (len(committed_prefix) + len(base_draft)) * 4 + 64

            if not branch_results:
                break

            for branch, vr in branch_results:
                resp = vr.cloud_response
                self._history.append(resp.accepted_len)
                metrics.total_server_verify_time_ms += resp.server_verify_time_ms
                metrics.total_network_time_ms += max(0.0, vr.rtt_ms - resp.server_verify_time_ms)
                metrics.downlink_bytes += len(resp.accepted_token_ids) * 4 + 32

            new_committed = list(committed_prefix)
            new_committed.extend(base_resp_cloud.accepted_token_ids)
            if correction is not None:
                new_committed.append(correction)

            selected_branch = branches[0]
            prefetched_ids = []
            prefetched_logprobs = []
            usable_branches = sorted(
                (branch for branch in branches[1:] if branch.offset <= base_accepted),
                key=lambda branch: branch.offset,
                reverse=True,
            )
            for branch in usable_branches:
                reusable = self._prefetch_from_branch(
                    branch=branch,
                    base_draft=base_draft,
                    base_accepted=base_accepted,
                    correction=correction,
                )
                if reusable is None:
                    continue
                selected_branch = branch
                prefetched_ids, prefetched_logprobs = reusable
                break

            total_drafted = sum(len(b.draft_ids) for b in branches)
            metrics.total_accepted_tokens += base_accepted
            if base_accepted < k:
                metrics.total_rollbacks += 1
            metrics.slot_details.append(
                {
                    "slot_id": selected_branch.branch_id,
                    "drafted": total_drafted,
                    "accepted": base_accepted,
                    "full_hit": base_accepted >= k,
                    "wasted_tokens": max(0, total_drafted - base_accepted - len(prefetched_ids)),
                    "draft_ms": sum(b.draft_time_ms for b in branches),
                    "verify_ms": sum(
                        vr.cloud_response.server_verify_time_ms for _, vr in branch_results
                    ),
                    "rtt_ms": max(vr.rtt_ms for _, vr in branch_results),
                    "rollback": base_accepted < k,
                    "tree_branch_width": len(branches),
                    "selected_offset": selected_branch.offset,
                    "stale_branches": sum(1 for b in branches if b.branch_id != selected_branch.branch_id),
                    "base_accepted": base_accepted,
                    "base_draft_ms": base_draft_ms,
                    "branch_draft_ms": batch_draft_ms,
                    "base_wait_ms": base_wait_ms,
                    "total_wait_ms": total_wait_ms,
                    "spec_verify_wall_ms": 0.0,
                    "prefetched_tokens": len(prefetched_ids),
                    "verify_prefix_len": base_resp_cloud.prefix_len,
                    "verify_draft_len": base_resp_cloud.draft_len,
                    "verify_input_len": base_resp_cloud.input_len,
                    "prompt_logprobs_requested": base_resp_cloud.prompt_logprobs_requested,
                    "verify_batch_size": base_resp_cloud.verify_batch_size,
                    "enable_prefix_caching": base_resp_cloud.enable_prefix_caching,
                    "enforce_eager": base_resp_cloud.enforce_eager,
                    "attention_backend": base_resp_cloud.attention_backend,
                    "vllm_version": base_resp_cloud.vllm_version,
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
