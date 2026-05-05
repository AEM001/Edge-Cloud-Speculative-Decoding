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
from typing import Dict, List, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client.edge_client import EdgeClient
from client.http_cloud_client import create_http_cloud_client
from config import DRAFT_GPU_MEM as GPU_MEMORY_UTILIZATION, DRAFT_MAX_LEN as MAX_MODEL_LEN, DRAFT_MODEL_NAME as MODEL_NAME, DRAFT_MODEL_PATH as MODEL_PATH
from core.draft_generator import VLLMDraftGenerator
from core.protocol import DraftRequest
from experiments.network_conditions import NetworkCondition, ThrottledCloudClient
from core.model_manager import VLLMModelManager
from experiments.prompt_loader import load_speed_bench_prompts
from experiments.tree_async_client import TreeAsyncEdgeClient

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
    simple_prompts = load_speed_bench_prompts(
        count=PROMPT_COUNT,
        category="coding",
        multiturn=False
    )
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


def run_direct_case(prompt_meta, text: str, throttled: ThrottledCloudClient) -> Optional[QuickResult]:
    throttled.reset_stats()
    tokens, total_ms, overhead_ms = _direct_with_throttle(text, throttled)
    if tokens <= 0:
        return None
    tps = tokens / (total_ms / 1000)
    result = QuickResult(
        method="direct",
        network=throttled.condition.name,
        prompt_type=prompt_meta["type"],
        prompt_id=prompt_meta["id"],
        tokens_generated=tokens,
        total_time_ms=total_ms,
        tokens_per_second=tps,
        sim_overhead_ms=overhead_ms,
    )
    logger.info(
        "    direct  : %d tok  %5.0f ms  %5.1f tok/s  overhead=%d ms",
        tokens,
        total_ms,
        tps,
        overhead_ms,
    )
    return result


def run_sync_spec_case(
    edge_client: EdgeClient,
    throttled: ThrottledCloudClient,
    prompt_meta,
    text: str,
    k: int,
) -> Optional[QuickResult]:
    throttled.reset_stats()
    metrics = _speculative(edge_client, text, k)
    if not metrics or metrics.generated_tokens <= 0:
        return None
    net_stats = throttled.get_stats_dict()
    total_ms = metrics.total_latency_ms
    tps = metrics.generated_tokens / (total_ms / 1000)
    accepted = round(metrics.acceptance_ratio * metrics.total_rounds * k)
    net_useful = accepted / metrics.total_rounds if metrics.total_rounds else 0
    method_name = f"sync_k{k}"
    logger.info(
        "    %s: %d tok  %5.0f ms  %5.1f tok/s  accept=%4.1f%%  net_useful=%.2f tok/round  "
        "rounds=%d  rtt=%d ms",
        method_name,
        metrics.generated_tokens,
        total_ms,
        tps,
        metrics.acceptance_ratio * 100,
        net_useful,
        metrics.total_rounds,
        metrics.average_rtt_ms,
    )
    return QuickResult(
        method=method_name,
        network=throttled.condition.name,
        prompt_type=prompt_meta["type"],
        prompt_id=prompt_meta["id"],
        tokens_generated=metrics.generated_tokens,
        total_time_ms=total_ms,
        tokens_per_second=tps,
        acceptance_rate=metrics.acceptance_ratio,
        num_rounds=metrics.total_rounds,
        net_useful_toks_per_round=net_useful,
        draft_time_ms=metrics.total_edge_draft_time_ms,
        verify_time_ms=metrics.total_server_verify_time_ms,
        avg_rtt_ms=metrics.average_rtt_ms,
        sim_overhead_ms=net_stats["total_simulated_overhead_ms"],
    )


def run_tree_spec_case(
    tree_async_client: TreeAsyncEdgeClient,
    throttled: ThrottledCloudClient,
    prompt_meta,
    text: str,
    k: int,
) -> Optional[QuickResult]:
    throttled.reset_stats()
    metrics = _tree_async_speculative(tree_async_client, text, k)
    if not metrics or metrics.generated_tokens <= 0:
        return None
    net_stats = throttled.get_stats_dict()
    total_ms = metrics.total_latency_ms
    tps = metrics.generated_tokens / (total_ms / 1000)
    accepted = metrics.total_accepted_tokens
    net_useful = accepted / metrics.total_rounds if metrics.total_rounds else 0
    method_name = f"tree_k{k}_b{tree_async_client.branch_width}"
    branch_width = _avg([s.get("tree_branch_width", 0) for s in metrics.slot_details])
    selected_offset = _avg([s.get("selected_offset", 0) for s in metrics.slot_details])
    stale_branches = _avg([s.get("stale_branches", 0) for s in metrics.slot_details])
    logger.info(
        "    %s: %d tok  %5.0f ms  %5.1f tok/s  accept=%4.1f%%  net_useful=%.2f tok/round  "
        "rounds=%d  rtt=%d ms  bubble=%.0fms  offset=%.1f",
        method_name,
        metrics.generated_tokens,
        total_ms,
        tps,
        metrics.acceptance_ratio * 100,
        net_useful,
        metrics.total_rounds,
        metrics.average_rtt_ms,
        metrics.avg_bubble_ms,
        selected_offset,
    )
    return QuickResult(
        method=method_name,
        network=throttled.condition.name,
        prompt_type=prompt_meta["type"],
        prompt_id=prompt_meta["id"],
        tokens_generated=metrics.generated_tokens,
        total_time_ms=total_ms,
        tokens_per_second=tps,
        acceptance_rate=metrics.acceptance_ratio,
        num_rounds=metrics.total_rounds,
        net_useful_toks_per_round=net_useful,
        draft_time_ms=metrics.total_edge_draft_time_ms,
        verify_time_ms=metrics.total_server_verify_time_ms,
        avg_rtt_ms=metrics.average_rtt_ms,
        sim_overhead_ms=net_stats["total_simulated_overhead_ms"],
        branch_width=branch_width,
        avg_selected_offset=selected_offset,
        avg_stale_branches=stale_branches,
    )


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

    # ── Warmup: fire a few calls so torch.compile / GPU kernels are hot ──
    logger.info("Warming up draft model and verify server ...")
    warmup_prompt = prompts[0][0]["text"]
    warmup_prefix = list(draft_generator.tokenizer.encode(warmup_prompt))
    for _ in range(3):
        draft_generator.generate_draft_tokens(
            DraftRequest(verified_prefix=warmup_prefix, num_draft_tokens=7),
            temperature=0.0,
        )
    for _ in range(3):
        try:
            requests.post(
                f"{SERVER_URL}/generate",
                json={"prompt": warmup_prompt, "max_tokens": 16, "temperature": 0.0},
                timeout=60.0,
            )
        except Exception:
            pass
    logger.info("Warmup complete.")

    results: List[QuickResult] = []
    tree_method_suffix = f"b{tree_async_client.branch_width}"

    for condition in NetworkCondition.all_profiles():
        throttled = ThrottledCloudClient(base_client, condition)
        edge_client.cloud_client  = throttled
        tree_async_client.cloud_client = throttled

        logger.info("")
        logger.info("── Network: %s", condition)

        for prompt_data, ptype in prompts:
            pid  = prompt_data["id"]
            text = prompt_data["text"]
            prompt_meta = {"id": pid, "type": ptype}
            logger.info("  [%s | prompt %d] %s...", ptype, pid, text[:55])

            result = run_direct_case(prompt_meta, text, throttled)
            if result:
                results.append(result)
            time.sleep(0.3)

            for k in K_VALUES:
                spec_result = run_sync_spec_case(edge_client, throttled, prompt_meta, text, k)
                if spec_result:
                    results.append(spec_result)
                time.sleep(0.3)

            for k in K_VALUES:
                tree_result = run_tree_spec_case(tree_async_client, throttled, prompt_meta, text, k)
                if tree_result:
                    results.append(tree_result)
                time.sleep(0.3)

    # ── Summary table ────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY  —  tok/s and speedup per network condition")
    logger.info("=" * 70)
    methods_to_show = (
        [f"sync_k{k}" for k in K_VALUES]
        + [f"tree_k{k}_{tree_method_suffix}" for k in K_VALUES]
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
    output_dir = Path(__file__).parent / "outputs_quick"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "quick_test_results.json"
    with open(out_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("\nResults saved to: %s", out_file)
    return True


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
