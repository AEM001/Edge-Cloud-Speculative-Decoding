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
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).parent))

from client.async_edge_client import AsyncEdgeClient
from client.edge_client import EdgeClient
from client.http_cloud_client import create_http_cloud_client
from config import DRAFT_GPU_MEM as GPU_MEMORY_UTILIZATION, DRAFT_MAX_LEN as MAX_MODEL_LEN, DRAFT_MODEL_NAME as MODEL_NAME, DRAFT_MODEL_PATH as MODEL_PATH
from draft_generator import VLLMDraftGenerator
from experiments.network_conditions import NetworkCondition, ThrottledCloudClient
from model_manager import VLLMModelManager
from prompt_loader import load_prompts_by_type

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
    async_client = AsyncEdgeClient(
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
                    method_name = f"async_k{k}"
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
                        "    async_k%-1d: %d tok  %5.0f ms  %5.1f tok/s  "
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
    methods_to_show = [f"sync_k{k}" for k in K_VALUES] + [f"async_k{k}" for k in K_VALUES]
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
