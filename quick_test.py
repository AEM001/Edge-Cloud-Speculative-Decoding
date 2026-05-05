#!/usr/bin/env python3
"""
Quick sanity check: does speculative decoding (K=7) beat throttled direct?

For each network condition the SAME throttle wrapper is applied to both
direct and speculative, so the comparison is apples-to-apples.

Key metric added: net_useful_toks_per_round  = accepted_tokens / rounds
  (i.e. how many tokens we actually *keep* per verify call)
Acceptance ratio alone is misleading because a 60% ratio on K=7 yields
4.2 kept tokens/round, which amortises the RTT much better than K=3 at 60%.
"""
import json
import logging
import math
import queue
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).parent))

from client.async_edge_client import AsyncEdgeClient, AsyncRequestMetrics, _VerifyResult
from client.edge_client import EdgeClient
from client.http_cloud_client import create_http_cloud_client
from config import DRAFT_GPU_MEM as GPU_MEMORY_UTILIZATION, DRAFT_MAX_LEN as MAX_MODEL_LEN, DRAFT_MODEL_NAME as MODEL_NAME, DRAFT_MODEL_PATH as MODEL_PATH
from draft_generator import VLLMDraftGenerator
from experiments.network_conditions import NetworkCondition, ThrottledCloudClient
from model_manager import VLLMModelManager
from prompt_loader import load_prompts_by_type
from protocol import DraftRequest, EdgeRequest


@dataclass
class TreeBranch:
    branch_id: int
    offset: int
    prefix: List[int]
    draft_ids: List[int]
    draft_logprobs: List[float]
    draft_time_ms: float


class TreeAsyncEdgeClient(AsyncEdgeClient):
    HISTORY_WINDOW: int = 50
    BRANCH_WIDTH: int = 3

    def __init__(self, *args, branch_width: int = BRANCH_WIDTH, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch_width = max(1, branch_width)
        self._history: deque = deque(maxlen=self.HISTORY_WINDOW)

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
            committed_end = len(committed_prefix)
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
            branches.append(TreeBranch(
                branch_id=tree_id * 100,
                offset=0,
                prefix=list(committed_prefix),
                draft_ids=base_draft,
                draft_logprobs=base_resp.logprobs,
                draft_time_ms=base_draft_ms,
            ))

            offsets = self._tree_offsets(k, len(base_draft))
            branch_offsets = [o for o in offsets if o > 0][:max(0, self.branch_width - 1)]
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
                    draft_resps.append(self.draft_generator.generate_draft_tokens(
                        DraftRequest(verified_prefix=list(prefix), num_draft_tokens=k),
                        temperature=self.temperature,
                    ))
                batch_draft_ms = (time.perf_counter() - batch_t0) * 1000

            per_branch_draft_ms = batch_draft_ms / len(draft_resps) if draft_resps else 0.0
            for branch_index, ((offset, prefix), draft_resp) in enumerate(zip(branch_prefixes, draft_resps), start=1):
                if not draft_resp.draft_token_ids:
                    continue
                branches.append(TreeBranch(
                    branch_id=tree_id * 100 + branch_index,
                    offset=offset,
                    prefix=list(prefix),
                    draft_ids=draft_resp.draft_token_ids,
                    draft_logprobs=draft_resp.logprobs,
                    draft_time_ms=per_branch_draft_ms,
                ))

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
            new_committed.extend(base_draft[:selected_branch.offset])
            new_committed.extend(selected_resp.accepted_token_ids)
            if selected_resp.correction_token_id is not None:
                new_committed.append(selected_resp.correction_token_id)

            total_drafted = sum(len(b.draft_ids) for b in branches)
            metrics.total_accepted_tokens += selected_progress
            if selected_progress < k:
                metrics.total_rollbacks += 1
            metrics.slot_details.append({
                "slot_id": selected_branch.branch_id,
                "drafted": total_drafted,
                "accepted": selected_progress,
                "full_hit": selected_progress >= k,
                "wasted_tokens": max(0, total_drafted - selected_progress),
                "draft_ms": sum(b.draft_time_ms for b in branches),
                "verify_ms": sum(vr.cloud_response.server_verify_time_ms for _, vr in branch_results),
                "rtt_ms": max(vr.rtt_ms for _, vr in branch_results),
                "rollback": selected_progress < k,
                "tree_branch_width": len(branches),
                "selected_offset": selected_branch.offset,
                "stale_branches": len(branches) - 1,
                "base_accepted": base_accept,
            })
            committed_prefix.clear()
            committed_prefix.extend(new_committed)

            metrics.average_rtt_ms = (
                metrics.average_rtt_ms * (metrics.total_rounds - 1)
                + max(vr.rtt_ms for _, vr in branch_results)
            ) / metrics.total_rounds
            speculative_prefix.clear()
            speculative_prefix.extend(committed_prefix)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:6006"
MAX_TOKENS = 128
K_VALUES  = [7]      # draft length
LOOKAHEAD = 1        # 1 verify in flight while 1 draft runs concurrently
PROMPT_COUNT = 2    # prompts per type (simple only)


@dataclass
class QuickResult:
    method: str                     # "direct" | "spec_k7"
    network: str                    # "good" | "medium" | "bursty"
    prompt_type: str
    prompt_id: int
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float
    # speculative-only
    acceptance_rate: float = 0.0
    num_rounds: int = 0
    net_useful_toks_per_round: float = 0.0   # accepted tokens / rounds
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    avg_rtt_ms: float = 0.0
    sim_overhead_ms: float = 0.0
    branch_width: float = 0.0
    avg_selected_offset: float = 0.0
    avg_stale_branches: float = 0.0


def load_prompts():
    simple_prompts, _ = load_prompts_by_type(count_per_type=PROMPT_COUNT)
    logger.info("Loaded %d simple prompts (complex skipped)", len(simple_prompts))
    return [(p, "simple") for p in simple_prompts]


def _direct_with_throttle(prompt: str, throttled: ThrottledCloudClient) -> Tuple[int, float, float]:
    """Direct /generate with same simulated network delay applied to direct too."""
    cond = throttled.condition
    now = time.perf_counter() - throttled._start_wall
    one_way_ms, dl_mbps, ul_mbps = cond.current_link_params(now)
    ul_bytes = len(prompt.encode()) + 64
    ul_delay = one_way_ms + NetworkCondition._payload_delay_ms(ul_bytes, ul_mbps)
    time.sleep(ul_delay / 1000.0)

    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{SERVER_URL}/generate",
            json={"prompt": prompt, "max_tokens": MAX_TOKENS, "temperature": 0.0},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        tokens = data.get("tokens_generated", 0)
        text = data.get("text", "")
    except Exception as exc:
        logger.error("Direct failed: %s", exc)
        return 0, 0.0, 0.0
    elapsed_ms = (time.perf_counter() - start) * 1000

    now2 = time.perf_counter() - throttled._start_wall
    one_way_ms2, dl_mbps2, _ = cond.current_link_params(now2)
    dl_bytes = len((text or "").encode()) + 64
    dl_delay = one_way_ms2 + NetworkCondition._payload_delay_ms(dl_bytes, dl_mbps2)
    time.sleep(dl_delay / 1000.0)

    total_ms = ul_delay + elapsed_ms + dl_delay
    overhead_ms = ul_delay + dl_delay
    return tokens, total_ms, overhead_ms


def _speculative(edge_client: EdgeClient, prompt: str, k: int) -> Optional[object]:
    try:
        return edge_client.generate(
            prompt=prompt,
            policy=lambda _rid, _toks: k,
            policy_name=f"StaticK{k}",
        )
    except Exception as exc:
        logger.error("Speculative K=%d failed: %s", k, exc)
        return None


def _tree_async_speculative(async_client: TreeAsyncEdgeClient, prompt: str, k: int) -> Optional[object]:
    try:
        return async_client.generate(
            prompt=prompt,
            policy=lambda _rid, _toks: k,
            policy_name=f"TreeK{k}",
        )
    except Exception as exc:
        logger.error("Tree speculative K=%d failed: %s", k, exc)
        return None


def _avg(lst): return sum(lst) / len(lst) if lst else 0.0


def run_quick_test():
    logger.info("=" * 70)
    logger.info("QUICK TEST  —  Direct (throttled) vs Speculative K=%s (throttled)", K_VALUES)
    logger.info("Draft model : %s", MODEL_NAME)
    logger.info("=" * 70)

    prompts = load_prompts()

    base_client = create_http_cloud_client(SERVER_URL, timeout=120.0)

    logger.info("Loading draft model on GPU 1 ...")
    model_manager = VLLMModelManager(MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, gpu_id=1)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    edge_client = EdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=base_client,   # swapped per condition below
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    tree_async_client = TreeAsyncEdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=base_client,   # swapped per condition below
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
        lookahead=LOOKAHEAD,
        branch_width=3,
    )
    logger.info("Draft model loaded.")

    results: List[QuickResult] = []

    for condition in NetworkCondition.all_profiles():
        throttled = ThrottledCloudClient(base_client, condition)
        edge_client.cloud_client  = throttled
        tree_async_client.cloud_client = throttled

        logger.info("")
        logger.info("── Network: %s", condition)

        for prompt_data, ptype in prompts:
            pid  = prompt_data["id"]
            text = prompt_data["text"]
            logger.info("  [%s | prompt %d] %s...", ptype, pid, text[:55])

            # ── Direct (with throttle) ──────────────────────────────────
            throttled.reset_stats()
            tokens, total_ms, overhead_ms = _direct_with_throttle(text, throttled)
            if tokens > 0:
                tps = tokens / (total_ms / 1000)
                results.append(QuickResult(
                    method="direct", network=condition.name,
                    prompt_type=ptype, prompt_id=pid,
                    tokens_generated=tokens, total_time_ms=total_ms,
                    tokens_per_second=tps, sim_overhead_ms=overhead_ms,
                ))
                logger.info("    direct  : %d tok  %5.0f ms  %5.1f tok/s  overhead=%d ms",
                            tokens, total_ms, tps, overhead_ms)
            time.sleep(0.3)

            # ── Sync Speculative (each K) ─────────────────────────────────────
            for k in K_VALUES:
                throttled.reset_stats()
                m = _speculative(edge_client, text, k)
                net_stats = throttled.get_stats_dict()
                if m and m.generated_tokens > 0:
                    total_ms_s = m.total_latency_ms
                    tps_s = m.generated_tokens / (total_ms_s / 1000)
                    accepted = round(m.acceptance_ratio * m.total_rounds * k)
                    net_useful = accepted / m.total_rounds if m.total_rounds else 0
                    method_name = f"sync_k{k}"
                    results.append(QuickResult(
                        method=method_name, network=condition.name,
                        prompt_type=ptype, prompt_id=pid,
                        tokens_generated=m.generated_tokens,
                        total_time_ms=total_ms_s,
                        tokens_per_second=tps_s,
                        acceptance_rate=m.acceptance_ratio,
                        num_rounds=m.total_rounds,
                        net_useful_toks_per_round=net_useful,
                        draft_time_ms=m.total_edge_draft_time_ms,
                        verify_time_ms=m.total_server_verify_time_ms,
                        avg_rtt_ms=m.average_rtt_ms,
                        sim_overhead_ms=net_stats["total_simulated_overhead_ms"],
                    ))
                    logger.info(
                        "    sync_k%-2d: %d tok  %5.0f ms  %5.1f tok/s  "
                        "accept=%4.1f%%  net_useful=%.2f tok/round  "
                        "rounds=%d  rtt=%d ms",
                        k, m.generated_tokens, total_ms_s, tps_s,
                        m.acceptance_ratio * 100, net_useful,
                        m.total_rounds, m.average_rtt_ms,
                    )
                time.sleep(0.3)

            # ── Tree Async Speculative (each K) ──────────────────────────────
            for k in K_VALUES:
                throttled.reset_stats()
                m = _tree_async_speculative(tree_async_client, text, k)
                net_stats = throttled.get_stats_dict()
                if m and m.generated_tokens > 0:
                    total_ms_t = m.total_latency_ms
                    tps_t = m.generated_tokens / (total_ms_t / 1000)
                    accepted = m.total_accepted_tokens
                    net_useful = accepted / m.total_rounds if m.total_rounds else 0
                    method_name = f"tree_k{k}_b{tree_async_client.branch_width}"
                    branch_width = _avg([
                        s.get("tree_branch_width", 0)
                        for s in m.slot_details
                    ])
                    selected_offset = _avg([
                        s.get("selected_offset", 0)
                        for s in m.slot_details
                    ])
                    stale_branches = _avg([
                        s.get("stale_branches", 0)
                        for s in m.slot_details
                    ])
                    results.append(QuickResult(
                        method=method_name, network=condition.name,
                        prompt_type=ptype, prompt_id=pid,
                        tokens_generated=m.generated_tokens,
                        total_time_ms=total_ms_t,
                        tokens_per_second=tps_t,
                        acceptance_rate=m.acceptance_ratio,
                        num_rounds=m.total_rounds,
                        net_useful_toks_per_round=net_useful,
                        draft_time_ms=m.total_edge_draft_time_ms,
                        verify_time_ms=m.total_server_verify_time_ms,
                        avg_rtt_ms=m.average_rtt_ms,
                        sim_overhead_ms=net_stats["total_simulated_overhead_ms"],
                        branch_width=branch_width,
                        avg_selected_offset=selected_offset,
                        avg_stale_branches=stale_branches,
                    ))
                    logger.info(
                        "    tree_k%-2d_b%d: %d tok  %5.0f ms  %5.1f tok/s  "
                        "accept=%4.1f%%  net_useful=%.2f tok/round  "
                        "rounds=%d  rtt=%d ms  bubble=%.0fms  offset=%.1f",
                        k, tree_async_client.branch_width,
                        m.generated_tokens, total_ms_t, tps_t,
                        m.acceptance_ratio * 100, net_useful,
                        m.total_rounds, m.average_rtt_ms,
                        m.avg_bubble_ms, selected_offset,
                    )
                time.sleep(0.3)

    # ── Summary table ────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY  —  tok/s and speedup per network condition")
    logger.info("=" * 70)
    methods_to_show = (
        [f"sync_k{k}" for k in K_VALUES]
        + [f"tree_k{k}_b3" for k in K_VALUES]
    )
    col_w = 28
    header = f"{'Network':<10}  {'direct':>8}" + "".join(
        f"  {m:>{col_w}}" for m in methods_to_show
    )
    logger.info(header)
    logger.info("-" * len(header))
    for net in [c.name for c in NetworkCondition.all_profiles()]:
        d_tps = _avg([r.tokens_per_second for r in results if r.method == "direct" and r.network == net])
        row = f"{net:<10}  {d_tps:>8.1f}"
        for mname in methods_to_show:
            s_tps   = _avg([r.tokens_per_second for r in results if r.method == mname and r.network == net])
            speedup = s_tps / d_tps if d_tps else 0
            flag    = " ✓" if speedup >= 1.0 else ""
            col     = f"{s_tps:.1f} ({speedup:.3f}x{flag})"
            row    += f"  {col:>{col_w}}"
        logger.info(row)

    # ── Save ─────────────────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "experiments" / "outputs_quick"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "quick_test_results.json"
    with open(out_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("\nResults saved to: %s", out_file)
    return True


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
