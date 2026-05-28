"""Tree-based speculative pipeline client used by quick_test.

This module implements a standalone tree-based speculative decoding client
without the complexity of the parent async pipeline.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
    score: float = 0.0


class TreeAsyncEdgeClient:
    """Standalone tree-based speculative decoding client.
    
    Drafts multiple parallel branches (a tree) and reuses the branch
    that matches verification results in the next round.
    """

    BRANCH_WIDTH: int = 3
    PROACTIVE_DECAY: float = math.log(0.95)

    def __init__(
        self,
        model_manager,
        draft_generator,
        cloud_client: Callable[[EdgeRequest], CloudResponse],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        branch_width: int = BRANCH_WIDTH,
        branch_draft_length: int = 8,
        eos_token_id: Optional[int] = None,
    ):
        self.model_manager = model_manager
        self.draft_generator = draft_generator
        self.cloud_client = cloud_client
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = draft_generator.tokenizer
        self.eos_token_id = eos_token_id
        if self.eos_token_id is None and self.tokenizer:
            self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        self.branch_width = max(1, branch_width)
        self.branch_draft_length = branch_draft_length

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

    def _proactive_seed_branches(
        self,
        committed_prefix: List[int],
        base_draft: List[int],
        base_logprobs: List[float],
        branch_base_id: int,
    ) -> List[TreeBranch]:
        """Choose SpecEdge-style proactive branch roots from local logprobs.

        SpecEdge picks the best bonus token by scoring leaf logprob plus the
        next-token logprob and a decay. This local adaptation treats each
        position along the sent linear draft as a possible verified leaf and
        creates branches from the highest scoring bonus tokens.
        """
        if (
            not base_draft
            or self.branch_draft_length <= 0
            or not hasattr(self.draft_generator, "generate_next_token_candidates_batch")
        ):
            return []

        prefixes: List[List[int]] = []
        accumulated_scores: List[float] = []
        running_score = 0.0
        for idx, _ in enumerate(base_draft):
            logprob = base_logprobs[idx] if idx < len(base_logprobs) else -float("inf")
            if math.isfinite(logprob):
                running_score += float(logprob)
            prefixes.append(list(committed_prefix) + base_draft[:idx + 1])
            accumulated_scores.append(running_score)

        candidate_lists = self.draft_generator.generate_next_token_candidates_batch(
            prefixes=prefixes,
            num_candidates=max(1, self.branch_width),
            temperature=self.temperature,
        )

        ranked: List[Tuple[float, int, int, float]] = []
        sent_tokens = set(base_draft)
        for prefix_idx, candidates in enumerate(candidate_lists):
            offset = prefix_idx + 1
            for token_id, token_logprob in candidates:
                if token_id in sent_tokens and offset < len(base_draft):
                    continue
                score = accumulated_scores[prefix_idx] + self.PROACTIVE_DECAY + token_logprob
                ranked.append((score, offset, token_id, token_logprob))

        ranked.sort(key=lambda item: item[0], reverse=True)
        branches: List[TreeBranch] = []
        seen_roots = set()
        for score, offset, token_id, token_logprob in ranked:
            root_key = (offset, token_id)
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)
            branch_id = branch_base_id + len(branches) + 1
            branches.append(
                TreeBranch(
                    branch_id=branch_id,
                    offset=offset,
                    prefix=list(committed_prefix) + base_draft[:offset],
                    draft_ids=[token_id],
                    draft_logprobs=[token_logprob],
                    draft_time_ms=0.0,
                    score=score,
                )
            )
            if len(branches) >= self.branch_width:
                break

        return branches

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
            remaining_budget = max_tokens - (len(committed_prefix) - len(prompt_ids))
            branches: List[TreeBranch] = []
            if prefetched_ids:
                # Reused tokens are already available and valid for the current
                # prefix. Send them immediately instead of waiting to fill K.
                send_len = min(len(prefetched_ids), remaining_budget)
                base_draft = prefetched_ids[:send_len]
                base_logprobs = prefetched_logprobs[:len(base_draft)]
                base_draft_ms = 0.0
                prefetched_ids = prefetched_ids[len(base_draft):]
                prefetched_logprobs = prefetched_logprobs[len(base_draft):]
            else:
                draft_len = min(k, remaining_budget)
                base_t0 = time.perf_counter()
                base_resp = self.draft_generator.generate_draft_tokens(
                    DraftRequest(verified_prefix=list(committed_prefix), num_draft_tokens=draft_len),
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
            branch_base_id = tree_id * 100
            branch_seed_t0 = time.perf_counter()
            proactive_seeds = self._proactive_seed_branches(
                committed_prefix=list(committed_prefix),
                base_draft=base_draft,
                base_logprobs=base_logprobs,
                branch_base_id=branch_base_id,
            )
            branch_seed_ms = (time.perf_counter() - branch_seed_t0) * 1000
            branch_lock = threading.Lock()
            branch_stop = threading.Event()
            branch_done_at: List[float] = []
            branch_batch_ms: List[float] = [branch_seed_ms if proactive_seeds else 0.0]
            branch_records: Dict[int, TreeBranch] = {
                branch.branch_id: branch for branch in proactive_seeds
            }

            def _run_branch_draft(
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
                            "branch_id": branch.branch_id,
                            "prefix": list(branch.prefix),
                            "draft_ids": list(branch.draft_ids),
                        }
                        for branch in local_branch_records.values()
                    ]

                    for _ in range(max(0, self.branch_draft_length - 1)):
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
            if proactive_seeds:
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
