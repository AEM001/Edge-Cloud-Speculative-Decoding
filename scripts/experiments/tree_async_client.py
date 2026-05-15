"""Tree-based speculative pipeline client used by quick_test.

This module implements a standalone tree-based speculative decoding client
without the complexity of the parent async pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.protocol import CloudResponse, DraftRequest, EdgeRequest


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _VerifyResult:
    slot_id: int
    cloud_response: CloudResponse
    rtt_ms: float


@dataclass
class AsyncRequestMetrics:
    """Metrics for tree-based speculative decoding (simplified)."""
    request_id: str
    prompt: str

    # Core throughput
    total_latency_ms: float = 0.0
    generated_tokens: int = 0
    tokens_per_second: float = 0.0

    # Round / token counts
    total_rounds: int = 0
    total_drafted_tokens: int = 0
    total_accepted_tokens: int = 0
    acceptance_ratio: float = 0.0

    # Timing breakdown
    total_edge_draft_time_ms: float = 0.0
    total_server_verify_time_ms: float = 0.0
    total_network_time_ms: float = 0.0
    average_rtt_ms: float = 0.0
    avg_bubble_ms: float = 0.0

    # Branch reuse metrics
    branch_reused: bool = False
    reused_tokens: int = 0

    # Pre-draft timing
    predraft_window_ms: float = 0.0
    reuse_prep_time_ms: float = 0.0

    # Network
    uplink_bytes: int = 0
    downlink_bytes: int = 0

    def compute_derived(self):
        if self.total_drafted_tokens > 0:
            self.acceptance_ratio = self.total_accepted_tokens / self.total_drafted_tokens
        if self.total_latency_ms > 0:
            self.tokens_per_second = 1000 * self.generated_tokens / self.total_latency_ms


@dataclass
class TreeBranch:
    branch_id: int
    offset: int
    prefix: List[int]
    draft_ids: List[int]
    draft_logprobs: List[float]
    draft_time_ms: float


class TreeAsyncEdgeClient:
    """Standalone tree-based speculative decoding client.
    
    Drafts multiple parallel branches (a tree) and reuses the branch
    that matches verification results in the next round.
    """

    BRANCH_WIDTH: int = 3
    BASE_ACCEPTANCE_RATIO: float = 0.60

    def __init__(
        self,
        model_manager,
        draft_generator,
        cloud_client: Callable[[EdgeRequest], CloudResponse],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        lookahead: int = 1,
        branch_width: int = BRANCH_WIDTH,
        eos_token_id: Optional[int] = None,
    ):
        self.model_manager = model_manager
        self.draft_generator = draft_generator
        self.cloud_client = cloud_client
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.lookahead = max(1, lookahead)
        self.tokenizer = draft_generator.tokenizer
        self.eos_token_id = eos_token_id
        if self.eos_token_id is None and self.tokenizer:
            self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        self.branch_width = max(1, branch_width)

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
        wall_start = time.perf_counter()

        try:
            self._tree_pipeline_loop(
                prompt_ids=prompt_ids,
                committed_prefix=committed_prefix,
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
            "bubble=%.1fms",
            metrics.generated_tokens,
            metrics.total_latency_ms,
            metrics.tokens_per_second,
            metrics.avg_bubble_ms,
        )
        return metrics

    def _tree_offsets(self, k: int, spec_ahead: int) -> List[int]:
        if spec_ahead <= 0:
            return [0]
        max_offset = min(k, spec_ahead)
        short_mode = 1 if max_offset >= 1 else 0
        center = max(0, min(max_offset, int(round(self.BASE_ACCEPTANCE_RATIO * k))))
        lower_mid = max(1, int(round(0.45 * k)))
        seeds = [max_offset, short_mode, center, lower_mid]

        offsets: List[int] = []
        for offset in seeds:
            clipped = max(0, min(max_offset, int(offset)))
            if clipped not in offsets:
                offsets.append(clipped)

        delta = 1
        while len(offsets) < self.branch_width and delta <= k:
            for candidate in (short_mode + delta, center - delta, lower_mid - delta, center + delta, max_offset - delta):
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

        # If offset < accepted, the branch was drafted before some tokens that
        # the target later accepted. Those leading branch tokens must bridge
        # back to reality before any remaining branch tokens can be reused.
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

                remaining_k = k - len(base_draft)
                if remaining_k > 0:
                    base_t0 = time.perf_counter()
                    topup_resp = self.draft_generator.generate_draft_tokens(
                        DraftRequest(
                            verified_prefix=list(committed_prefix) + base_draft,
                            num_draft_tokens=remaining_k,
                        ),
                        temperature=self.temperature,
                    )
                    base_draft_ms = (time.perf_counter() - base_t0) * 1000
                    base_draft.extend(topup_resp.draft_token_ids)
                    base_logprobs.extend(topup_resp.logprobs)
            else:
                base_t0 = time.perf_counter()
                base_resp = self.draft_generator.generate_draft_tokens(
                    DraftRequest(verified_prefix=list(committed_prefix), num_draft_tokens=k),
                    temperature=self.temperature,
                )
                base_draft_ms = (time.perf_counter() - base_t0) * 1000
                base_draft = list(base_resp.draft_token_ids)
                base_logprobs = list(base_resp.logprobs)
            if not base_draft:
                break
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
                prefix_ids=list(committed_prefix),
                draft_ids=base_draft,
            )
            base_verify_result: List = []   # filled by thread

            def _run_base_verify(req=base_edge_req, out=base_verify_result):
                try:
                    resp = self.cloud_client(req)
                    out.append((resp, time.perf_counter()))
                except Exception as exc:
                    logger.error("Base verify error: %s", exc)

            verify_thread = threading.Thread(target=_run_base_verify, daemon=True)
            verify_start = time.perf_counter()
            verify_thread.start()

            # Draft speculative branches in a separate background task. The
            # critical path never waits for this task; it only consumes branch
            # results that are already ready when base verification returns.
            offsets = self._tree_offsets(k, len(base_draft))
            branch_offsets = [o for o in offsets if o > 0][: max(0, self.branch_width - 1)]
            branch_prefixes = []
            for offset in branch_offsets:
                prefix = list(committed_prefix)
                prefix.extend(base_draft[:offset])
                branch_prefixes.append((offset, prefix))

            branch_base_id = tree_id * 100
            branch_lock = threading.Lock()
            branch_stop = threading.Event()
            branch_done_at: List[float] = []
            branch_batch_ms: List[float] = [0.0]
            branch_records: Dict[int, TreeBranch] = {
                branch_base_id + branch_index: TreeBranch(
                    branch_id=branch_base_id + branch_index,
                    offset=offset,
                    prefix=list(prefix),
                    draft_ids=[],
                    draft_logprobs=[],
                    draft_time_ms=0.0,
                )
                for branch_index, (offset, prefix) in enumerate(branch_prefixes, start=1)
            }

            def _run_branch_draft(
                prefixes=branch_prefixes,
                branch_base_id=branch_base_id,
                local_branch_records=branch_records,
                local_branch_lock=branch_lock,
                local_branch_stop=branch_stop,
                local_branch_done_at=branch_done_at,
                local_branch_batch_ms=branch_batch_ms,
            ):
                """Stream branch drafts token-by-token for early reuse.

                The snapshot consumed after verification may contain partial
                branches. That is intentional: a partial branch is enough when
                it bridges the accepted base suffix plus correction token.
                """
                try:
                    active = [
                        {
                            "branch_id": branch_base_id + branch_index,
                            "prefix": list(prefix),
                            "draft_ids": [],
                        }
                        for branch_index, (_, prefix) in enumerate(prefixes, start=1)
                    ]

                    for _ in range(k):
                        if local_branch_stop.is_set() or not active:
                            break

                        step_t0 = time.perf_counter()
                        if hasattr(self.draft_generator, "generate_draft_tokens_batch"):
                            draft_resps_ = self.draft_generator.generate_draft_tokens_batch(
                                [
                                    DraftRequest(
                                        verified_prefix=item["prefix"] + item["draft_ids"],
                                        num_draft_tokens=1,
                                    )
                                    for item in active
                                ],
                                temperature=self.temperature,
                            )
                        else:
                            draft_resps_ = [
                                self.draft_generator.generate_draft_tokens(
                                    DraftRequest(
                                        verified_prefix=item["prefix"] + item["draft_ids"],
                                        num_draft_tokens=1,
                                    ),
                                    temperature=self.temperature,
                                )
                                for item in active
                            ]

                        step_ms = (time.perf_counter() - step_t0) * 1000
                        if local_branch_stop.is_set():
                            break
                        per_branch_step_ms = step_ms / len(active) if active else 0.0
                        next_active = []
                        with local_branch_lock:
                            local_branch_batch_ms[0] += step_ms
                            for item, draft_resp in zip(active, draft_resps_):
                                if not draft_resp.draft_token_ids:
                                    continue

                                token_id = draft_resp.draft_token_ids[0]
                                logprob = (
                                    draft_resp.logprobs[0]
                                    if draft_resp.logprobs
                                    else -float("inf")
                                )
                                item["draft_ids"].append(token_id)
                                record = local_branch_records[item["branch_id"]]
                                record.draft_ids.append(token_id)
                                record.draft_logprobs.append(float(logprob))
                                record.draft_time_ms += per_branch_step_ms
                                next_active.append(item)
                        active = next_active

                    with local_branch_lock:
                        local_branch_done_at.append(time.perf_counter())
                except Exception as exc:
                    logger.error("Branch draft error: %s", exc)

            def _snapshot_streamed_branches():
                with branch_lock:
                    streamed = [
                        TreeBranch(
                            branch_id=record.branch_id,
                            offset=record.offset,
                            prefix=list(record.prefix),
                            draft_ids=list(record.draft_ids),
                            draft_logprobs=list(record.draft_logprobs),
                            draft_time_ms=record.draft_time_ms,
                        )
                        for record in branch_records.values()
                        if record.draft_ids
                    ]
                    done_at = branch_done_at[0] if branch_done_at else None
                    return streamed, branch_batch_ms[0], done_at

            branch_thread = threading.Thread(target=_run_branch_draft, daemon=True)
            if branch_prefixes:
                branch_thread.start()

            if not branches:
                verify_thread.join(timeout=120)
                break

            metrics.total_rounds += 1
            metrics.total_drafted_tokens += len(base_draft)
            metrics.total_edge_draft_time_ms += base_draft_ms

            # Start timing: pre-draft window (send verify → receive response)
            bubble_t0 = time.perf_counter()
            verify_thread.join(timeout=120)
            verify_done = time.perf_counter()
            predraft_window_ms = (verify_done - verify_start) * 1000
            metrics.predraft_window_ms += predraft_window_ms

            if not base_verify_result:
                break
            base_resp_cloud, _ = base_verify_result[0]
            branch_stop.set()
            if branch_thread.is_alive():
                branch_thread.join(timeout=0.001)
            
            # Track bubble time (wait for async operations)
            bubble_ms = (verify_done - bubble_t0) * 1000
            if metrics.total_rounds > 0:
                metrics.avg_bubble_ms = (metrics.avg_bubble_ms * (metrics.total_rounds - 1) + bubble_ms) / metrics.total_rounds

            drafted_branches, batch_draft_ms, _ = _snapshot_streamed_branches()
            if drafted_branches:
                branches.extend(drafted_branches)
                metrics.total_edge_draft_time_ms += batch_draft_ms
            verify_rtt_ms = base_resp_cloud.rtt_ms or ((verify_done - verify_start) * 1000)
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
                metrics.total_server_verify_time_ms += resp.server_verify_time_ms
                metrics.total_network_time_ms += max(0.0, vr.rtt_ms - resp.server_verify_time_ms)
                metrics.downlink_bytes += 48

            new_committed = list(committed_prefix)
            new_committed.extend(base_draft[:base_accepted])
            if correction is not None:
                new_committed.append(correction)

            # Start timing: reuse prep time (receive response → send next request)
            prep_start = time.perf_counter()
            
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

            # Record branch reuse metrics
            if prefetched_ids:
                metrics.branch_reused = True
                metrics.reused_tokens += len(prefetched_ids)
            
            prep_end = time.perf_counter()
            reuse_prep_ms = (prep_end - prep_start) * 1000
            metrics.reuse_prep_time_ms += reuse_prep_ms

            total_drafted = sum(len(b.draft_ids) for b in branches)
            metrics.total_accepted_tokens += base_accepted
            committed_prefix.clear()
            committed_prefix.extend(new_committed)

            metrics.average_rtt_ms = (
                metrics.average_rtt_ms * (metrics.total_rounds - 1)
                + max(vr.rtt_ms for _, vr in branch_results)
            ) / metrics.total_rounds
