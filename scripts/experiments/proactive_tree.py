"""Local proactive tree planning for speculative pre-draft.

This module keeps branch policy and branch expansion separate from the async
client loop. The client decides when to verify and commit; the planner decides
which local branches are worth drafting and which branch can be reused.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.protocol import DraftRequest


logger = logging.getLogger(__name__)


@dataclass
class TreeBranch:
    branch_id: int
    offset: int
    prefix: List[int]
    draft_ids: List[int]
    draft_logprobs: List[float]
    draft_time_ms: float
    score: float = 0.0


@dataclass
class BranchDraftState:
    branches: Dict[int, TreeBranch]
    lock: threading.Lock
    stop: threading.Event
    batch_time_ms: float = 0.0
    done_at: float | None = None

    def snapshot(self) -> tuple[List[TreeBranch], float, float | None]:
        with self.lock:
            branches = [
                TreeBranch(
                    branch_id=branch.branch_id,
                    offset=branch.offset,
                    prefix=list(branch.prefix),
                    draft_ids=list(branch.draft_ids),
                    draft_logprobs=list(branch.draft_logprobs),
                    draft_time_ms=branch.draft_time_ms,
                    score=branch.score,
                )
                for branch in self.branches.values()
                if branch.draft_ids
            ]
            return branches, self.batch_time_ms, self.done_at


class ProactiveTreeDraftPlanner:
    """SpecEdge-inspired local pre-draft planner.

    The planner chooses branch roots using accumulated local draft logprobs plus
    a next-token candidate logprob and decay. It then expands those branches in
    the verification window and exposes only reusable continuations.
    """

    PROACTIVE_DECAY = math.log(0.95)

    def __init__(
        self,
        draft_generator,
        branch_width: int,
        branch_draft_length: int,
        temperature: float = 0.0,
    ):
        self.draft_generator = draft_generator
        self.branch_width = max(1, branch_width)
        self.branch_draft_length = max(0, branch_draft_length)
        self.temperature = temperature

    def seed_branches(
        self,
        committed_prefix: List[int],
        base_draft: List[int],
        base_logprobs: List[float],
        branch_base_id: int,
    ) -> List[TreeBranch]:
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

    def new_state(self, branches: List[TreeBranch], seed_time_ms: float) -> BranchDraftState:
        return BranchDraftState(
            branches={branch.branch_id: branch for branch in branches},
            lock=threading.Lock(),
            stop=threading.Event(),
            batch_time_ms=seed_time_ms if branches else 0.0,
        )

    def expand_until_stopped(self, state: BranchDraftState) -> None:
        """Expand seeded branches token by token until stopped or complete."""
        try:
            active = [
                {
                    "branch_id": branch.branch_id,
                    "prefix": list(branch.prefix),
                    "draft_ids": list(branch.draft_ids),
                }
                for branch in state.branches.values()
            ]

            for _ in range(max(0, self.branch_draft_length - 1)):
                if state.stop.is_set() or not active:
                    break

                step_t0 = time.perf_counter()
                draft_responses = self._generate_one_token_batch(active)
                step_ms = (time.perf_counter() - step_t0) * 1000
                if state.stop.is_set():
                    break

                per_branch_step_ms = step_ms / len(active) if active else 0.0
                next_active = []
                with state.lock:
                    state.batch_time_ms += step_ms
                    for item, draft_response in zip(active, draft_responses):
                        if not draft_response.draft_token_ids:
                            continue

                        token_id = draft_response.draft_token_ids[0]
                        logprob = (
                            draft_response.logprobs[0]
                            if draft_response.logprobs
                            else -float("inf")
                        )
                        item["draft_ids"].append(token_id)
                        branch = state.branches[item["branch_id"]]
                        branch.draft_ids.append(token_id)
                        branch.draft_logprobs.append(float(logprob))
                        branch.draft_time_ms += per_branch_step_ms
                        next_active.append(item)
                active = next_active

            with state.lock:
                state.done_at = time.perf_counter()
        except Exception as exc:
            logger.error("Branch draft error: %s", exc)

    def reusable_continuation(
        self,
        branch: TreeBranch,
        base_draft: List[int],
        base_accepted: int,
        correction: int | None,
    ) -> tuple[List[int], List[float]] | None:
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

    def _generate_one_token_batch(self, active_branches):
        requests = [
            DraftRequest(
                verified_prefix=item["prefix"] + item["draft_ids"],
                num_draft_tokens=1,
            )
            for item in active_branches
        ]
        if hasattr(self.draft_generator, "generate_draft_tokens_batch"):
            return self.draft_generator.generate_draft_tokens_batch(
                requests,
                temperature=self.temperature,
            )

        return [
            self.draft_generator.generate_draft_tokens(
                request,
                temperature=self.temperature,
            )
            for request in requests
        ]
