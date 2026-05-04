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
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).parent))

from client.async_edge_client import AsyncEdgeClient, PipelineSlot, _VerifyJob, _VerifyResult
from client.edge_client import EdgeClient
from client.http_cloud_client import create_http_cloud_client
from config import DRAFT_GPU_MEM as GPU_MEMORY_UTILIZATION, DRAFT_MAX_LEN as MAX_MODEL_LEN, DRAFT_MODEL_NAME as MODEL_NAME, DRAFT_MODEL_PATH as MODEL_PATH
from draft_generator import VLLMDraftGenerator
from experiments.network_conditions import NetworkCondition, ThrottledCloudClient
from model_manager import VLLMModelManager
from prompt_loader import load_prompts_by_type
from protocol import DraftRequest, EdgeRequest


# ---------------------------------------------------------------------------
# CQT (Conservative Quantile Tracking) Async Client — quick_test only
# ---------------------------------------------------------------------------

class CQTAsyncEdgeClient(AsyncEdgeClient):
    """
    AsyncEdgeClient with Conservative Quantile Tracking prefix selection.

    Before each round, instead of assuming the full K draft tokens were
    accepted (full speculative advance), we advance the speculative prefix
    by only `prefix_len = p10(history_accepted_lens)` tokens — the 10th
    percentile of the rolling acceptance-length window (last K_WINDOW rounds).

    This guarantees >=90% of async drafts start from a prefix the verifier
    will actually have committed, reducing wasted rollback work on high-RTT
    connections at the cost of slightly shorter overlap windows.

    When history is empty (first round), falls back to full optimistic advance.
    """

    CQT_WINDOW: int = 50       # rolling window size
    CQT_PERCENTILE: float = 0.10   # 10th percentile

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cqt_history: deque = deque(maxlen=self.CQT_WINDOW)

    def _cqt_prefix_len(self, k: int) -> int:
        """Return conservative prefix advance length based on history."""
        if not self._cqt_history:
            return k   # no history yet — full optimistic advance
        sorted_h = sorted(self._cqt_history)
        idx = int(math.floor(self.CQT_PERCENTILE * len(sorted_h)))
        idx = max(0, min(idx, len(sorted_h) - 1))
        return max(1, sorted_h[idx])

    def _pipeline_loop(
        self,
        prompt_ids,
        committed_prefix,
        speculative_prefix,
        in_flight,
        next_slot_id_ref,
        draft_queue,
        result_queue,
        policy,
        policy_name,
        request_id,
        metrics,
    ):
        max_tokens = self.max_new_tokens
        eos = self.eos_token_id

        def _done():
            if len(committed_prefix) - len(prompt_ids) >= max_tokens:
                return True
            if eos and eos in committed_prefix[len(prompt_ids):]:
                return True
            return False

        while not _done():

            # Phase A: drain non-blocking results
            while True:
                try:
                    vr: _VerifyResult = result_queue.get_nowait()
                except queue.Empty:
                    break
                slot = in_flight.pop(vr.slot_id, None)
                if slot is None:
                    continue
                rollback = self._apply_result(
                    vr, slot, committed_prefix, speculative_prefix, in_flight, metrics
                )
                # Record accepted length for CQT history
                self._cqt_history.append(vr.cloud_response.accepted_len)
                if rollback:
                    break

            if _done():
                break

            # Phase B: choose prefix for next draft using CQT
            slot_id = next_slot_id_ref[0]
            next_slot_id_ref[0] += 1
            K = policy(slot_id, [])

            # CQT: advance speculative_prefix by at most cqt_prefix_len tokens
            # ahead of committed_prefix, rather than the full speculative extent.
            cqt_len = self._cqt_prefix_len(K)
            committed_end = len(committed_prefix)
            full_spec_end = len(speculative_prefix)
            # How far ahead of committed is the current speculative prefix?
            spec_ahead = full_spec_end - committed_end
            # We only draft from committed + min(spec_ahead, cqt_len) ahead
            conservative_end = committed_end + min(spec_ahead, cqt_len)
            conservative_prefix = speculative_prefix[:conservative_end]

            draft_t0 = time.perf_counter()
            draft_req = DraftRequest(
                verified_prefix=list(conservative_prefix),
                num_draft_tokens=K,
            )
            draft_resp = self.draft_generator.generate_draft_tokens(
                draft_req, temperature=self.temperature,
            )
            draft_ms = (time.perf_counter() - draft_t0) * 1000

            if not draft_resp.draft_token_ids:
                break

            # Phase C: wait for one result if at max in-flight
            if len(in_flight) >= self.lookahead:
                bubble_t0 = time.perf_counter()
                vr = result_queue.get()
                metrics.total_bubble_ms += (time.perf_counter() - bubble_t0) * 1000

                slot = in_flight.pop(vr.slot_id, None)
                if slot is not None:
                    self._cqt_history.append(vr.cloud_response.accepted_len)
                    rollback = self._apply_result(
                        vr, slot, committed_prefix, speculative_prefix, in_flight, metrics
                    )
                    if rollback:
                        continue

            if _done():
                break

            # Phase D: submit draft
            slot = PipelineSlot(
                slot_id=slot_id,
                assumed_prefix=list(conservative_prefix),
                draft_ids=draft_resp.draft_token_ids,
                draft_logprobs=draft_resp.logprobs,
                draft_time_ms=draft_ms,
            )
            in_flight[slot_id] = slot

            metrics.total_rounds             += 1
            metrics.total_drafted_tokens     += len(draft_resp.draft_token_ids)
            metrics.total_edge_draft_time_ms += draft_ms

            edge_req = EdgeRequest(
                request_id=request_id,
                round_id=slot_id,
                prefix_ids=list(conservative_prefix),
                draft_ids=draft_resp.draft_token_ids,
                draft_logprobs=draft_resp.logprobs,
                edge_draft_time_ms=draft_ms,
                policy_metadata={"policy_name": policy_name, "K": K, "cqt_prefix_len": cqt_len},
            )
            metrics.uplink_bytes += len(json.dumps(edge_req.to_dict()).encode())
            draft_queue.put(_VerifyJob(slot_id=slot_id, request=edge_req))

            # Optimistically extend speculative prefix by the full draft
            # (CQT only constrains where we *start* drafting, not the
            # assumption about how much of this new draft is accepted)
            speculative_prefix.clear()
            speculative_prefix.extend(conservative_prefix)
            speculative_prefix.extend(draft_resp.draft_token_ids)

        # Drain remaining in-flight slots
        while in_flight:
            try:
                vr = result_queue.get(timeout=120)
            except queue.Empty:
                break
            slot = in_flight.pop(vr.slot_id, None)
            if slot is None:
                continue
            self._cqt_history.append(vr.cloud_response.accepted_len)
            rollback = self._apply_result(
                vr, slot, committed_prefix, speculative_prefix, in_flight, metrics
            )
            if rollback:
                break

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


def _async_speculative(async_client: AsyncEdgeClient, prompt: str, k: int) -> Optional[object]:
    try:
        return async_client.generate(
            prompt=prompt,
            policy=lambda _rid, _toks: k,
            policy_name=f"AsyncK{k}",
        )
    except Exception as exc:
        logger.error("Async speculative K=%d failed: %s", k, exc)
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
    async_client = CQTAsyncEdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=base_client,   # swapped per condition below
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
        lookahead=LOOKAHEAD,
    )
    logger.info("Draft model loaded.")

    results: List[QuickResult] = []

    for condition in NetworkCondition.all_profiles():
        throttled = ThrottledCloudClient(base_client, condition)
        edge_client.cloud_client  = throttled
        async_client.cloud_client = throttled

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

            # ── Async Speculative (each K) ────────────────────────────────────
            for k in K_VALUES:
                throttled.reset_stats()
                m = _async_speculative(async_client, text, k)
                net_stats = throttled.get_stats_dict()
                if m and m.generated_tokens > 0:
                    total_ms_a = m.total_latency_ms
                    tps_a = m.generated_tokens / (total_ms_a / 1000)
                    accepted = m.total_accepted_tokens
                    net_useful = accepted / m.total_rounds if m.total_rounds else 0
                    method_name = f"cqt_k{k}"
                    results.append(QuickResult(
                        method=method_name, network=condition.name,
                        prompt_type=ptype, prompt_id=pid,
                        tokens_generated=m.generated_tokens,
                        total_time_ms=total_ms_a,
                        tokens_per_second=tps_a,
                        acceptance_rate=m.acceptance_ratio,
                        num_rounds=m.total_rounds,
                        net_useful_toks_per_round=net_useful,
                        draft_time_ms=m.total_edge_draft_time_ms,
                        verify_time_ms=m.total_server_verify_time_ms,
                        avg_rtt_ms=m.average_rtt_ms,
                        sim_overhead_ms=net_stats["total_simulated_overhead_ms"],
                    ))
                    logger.info(
                        "    cqt_k%-2d : %d tok  %5.0f ms  %5.1f tok/s  "
                        "accept=%4.1f%%  net_useful=%.2f tok/round  "
                        "rounds=%d  rtt=%d ms  bubble=%.0fms",
                        k, m.generated_tokens, total_ms_a, tps_a,
                        m.acceptance_ratio * 100, net_useful,
                        m.total_rounds, m.average_rtt_ms,
                        m.avg_bubble_ms,
                    )
                time.sleep(0.3)

    # ── Summary table ────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY  —  tok/s and speedup per network condition")
    logger.info("=" * 70)
    methods_to_show = [f"sync_k{k}" for k in K_VALUES] + [f"cqt_k{k}" for k in K_VALUES]
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
