#!/usr/bin/env python3
"""
Quick sanity check: does speculative decoding (K=7) beat throttled direct?

For each network condition the SAME throttle wrapper is applied to both
direct and speculative, so the comparison is apples-to-apples.

Results use a normalized metric schema so direct, sync speculative, and
tree async runs can be compared through the same output/timing/speculative
fields while keeping method-specific raw round details separately.
"""
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client.edge_client import EdgeClient
from client.http_cloud_client import create_http_cloud_client
from config import DRAFT_GPU_MEM as GPU_MEMORY_UTILIZATION, DRAFT_MAX_LEN as MAX_MODEL_LEN, DRAFT_MODEL_NAME as MODEL_NAME, DRAFT_MODEL_PATH as MODEL_PATH
from core.draft_generator import VLLMDraftGenerator
from core.protocol import DraftRequest, EdgeRequest
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
DIRECT_SESSION = requests.Session()
DIRECT_SESSION.trust_env = False


@dataclass
class OutputMetrics:
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float


@dataclass
class TimingMetrics:
    client_wall_ms: float = 0.0
    local_draft_ms: float = 0.0
    server_model_ms: float = 0.0
    server_total_ms: float = 0.0
    http_rpc_ms: float = 0.0
    simulated_ul_ms: float = 0.0
    simulated_dl_ms: float = 0.0
    simulated_network_ms: float = 0.0
    avg_rtt_ms: float = 0.0
    critical_path_wait_ms: float = 0.0


@dataclass
class SpeculativeMetrics:
    rounds: int = 0
    k: int = 0
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    correction_tokens: int = 0
    generated_per_round: float = 0.0
    accepted_draft_per_round: float = 0.0
    acceptance_rate: float = 0.0
    wasted_draft_tokens: int = 0


@dataclass
class AsyncDetailMetrics:
    launched_branch_count: float = 0.0
    ready_branch_count: float = 0.0
    reused_branch_count: float = 0.0
    prefetched_tokens: float = 0.0
    exposed_branch_ms: float = 0.0
    selected_offset: float = 0.0
    base_accepted: float = 0.0
    stale_branch_count: float = 0.0


@dataclass
class VerifyRuntimeMetrics:
    avg_prefix_len: float = 0.0
    avg_draft_len: float = 0.0
    avg_input_len: float = 0.0
    prefix_caching: Optional[bool] = None
    enforce_eager: Optional[bool] = None
    attention_backend: Optional[str] = None
    vllm_version: Optional[str] = None


@dataclass
class ExperimentResult:
    method: str
    method_family: str              # direct | sync_spec | tree_async
    network: str
    prompt_type: str
    prompt_id: int
    output: OutputMetrics
    timing: TimingMetrics
    speculative: SpeculativeMetrics
    async_detail: AsyncDetailMetrics
    verify_runtime: VerifyRuntimeMetrics
    raw: Dict[str, Any]


@dataclass
class DirectTiming:
    tokens: int
    total_ms: float
    overhead_ms: float
    server_ms: float
    http_ms: float
    ul_ms: float
    dl_ms: float


def load_prompts():
    simple_prompts = load_speed_bench_prompts(
        count=PROMPT_COUNT,
        category="coding",
        multiturn=False
    )
    logger.info("Loaded %d simple prompts (complex skipped)", len(simple_prompts))
    return [(p, "simple") for p in simple_prompts]


def _direct_with_throttle(prompt: str, throttled: ThrottledCloudClient) -> DirectTiming:
    """Direct /generate with same simulated network delay applied to direct too."""
    cond = throttled.condition
    now = time.perf_counter() - throttled._start_wall
    one_way_ms, dl_mbps, ul_mbps = cond.current_link_params(now)
    ul_bytes = len(prompt.encode()) + 64
    ul_delay = one_way_ms + NetworkCondition._payload_delay_ms(ul_bytes, ul_mbps)
    time.sleep(ul_delay / 1000.0)

    start = time.perf_counter()
    try:
        resp = DIRECT_SESSION.post(
            f"{SERVER_URL}/generate",
            json={"prompt": prompt, "max_tokens": MAX_TOKENS, "temperature": 0.0},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        tokens = data.get("tokens_generated", 0)
        text = data.get("text", "")
        server_ms = float(data.get("generation_time_ms", 0.0))
    except Exception as exc:
        logger.error("Direct failed: %s", exc)
        return DirectTiming(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    now2 = time.perf_counter() - throttled._start_wall
    one_way_ms2, dl_mbps2, _ = cond.current_link_params(now2)
    dl_bytes = len((text or "").encode()) + 64
    dl_delay = one_way_ms2 + NetworkCondition._payload_delay_ms(dl_bytes, dl_mbps2)
    time.sleep(dl_delay / 1000.0)

    total_ms = ul_delay + elapsed_ms + dl_delay
    overhead_ms = ul_delay + dl_delay
    return DirectTiming(tokens, total_ms, overhead_ms, server_ms, elapsed_ms, ul_delay, dl_delay)


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


def run_direct_case(prompt_meta, text: str, throttled: ThrottledCloudClient) -> Optional[ExperimentResult]:
    throttled.reset_stats()
    timing = _direct_with_throttle(text, throttled)
    if timing.tokens <= 0:
        return None
    tps = timing.tokens / (timing.total_ms / 1000)
    result = ExperimentResult(
        method="direct",
        method_family="direct",
        network=throttled.condition.name,
        prompt_type=prompt_meta["type"],
        prompt_id=prompt_meta["id"],
        output=OutputMetrics(
            tokens_generated=timing.tokens,
            total_time_ms=timing.total_ms,
            tokens_per_second=tps,
        ),
        timing=TimingMetrics(
            client_wall_ms=timing.total_ms,
            server_model_ms=timing.server_ms,
            server_total_ms=timing.server_ms,
            http_rpc_ms=timing.http_ms,
            simulated_ul_ms=timing.ul_ms,
            simulated_dl_ms=timing.dl_ms,
            simulated_network_ms=timing.overhead_ms,
        ),
        speculative=SpeculativeMetrics(),
        async_detail=AsyncDetailMetrics(),
        verify_runtime=VerifyRuntimeMetrics(),
        raw={"direct_timing": asdict(timing)},
    )
    logger.info(
        "    direct  : %d tok  %5.0f ms  %5.1f tok/s  "
        "server=%d ms  http=%d ms  sim_ul=%d ms  sim_dl=%d ms",
        timing.tokens,
        timing.total_ms,
        tps,
        timing.server_ms,
        timing.http_ms,
        timing.ul_ms,
        timing.dl_ms,
    )
    return result


def run_sync_spec_case(
    edge_client: EdgeClient,
    throttled: ThrottledCloudClient,
    prompt_meta,
    text: str,
    k: int,
) -> Optional[ExperimentResult]:
    throttled.reset_stats()
    metrics = _speculative(edge_client, text, k)
    if not metrics or metrics.generated_tokens <= 0:
        return None
    net_stats = throttled.get_stats_dict()
    total_ms = metrics.total_latency_ms
    tps = metrics.generated_tokens / (total_ms / 1000)
    accepted = metrics.total_accepted_drafted_tokens
    net_useful = accepted / metrics.total_rounds if metrics.total_rounds else 0
    correction_tokens = max(0, metrics.generated_tokens - accepted)
    method_name = f"sync_k{k}"
    avg_verify_prefix_len = _avg([s.get("verify_prefix_len", 0) or 0 for s in metrics.round_details])
    avg_verify_draft_len = _avg([s.get("verify_draft_len", 0) or 0 for s in metrics.round_details])
    avg_verify_input_len = _avg([s.get("verify_input_len", 0) or 0 for s in metrics.round_details])
    logger.info(
        "    %s: %d tok  %5.0f ms  %5.1f tok/s  accept=%4.1f%%  net_useful=%.2f tok/round  "
        "rounds=%d  rtt=%d ms  verify_prefix=%.0f  verify_input=%.0f",
        method_name,
        metrics.generated_tokens,
        total_ms,
        tps,
        metrics.acceptance_ratio * 100,
        net_useful,
        metrics.total_rounds,
        metrics.average_rtt_ms,
        avg_verify_prefix_len,
        avg_verify_input_len,
    )
    return ExperimentResult(
        method=method_name,
        method_family="sync_spec",
        network=throttled.condition.name,
        prompt_type=prompt_meta["type"],
        prompt_id=prompt_meta["id"],
        output=OutputMetrics(
            tokens_generated=metrics.generated_tokens,
            total_time_ms=total_ms,
            tokens_per_second=tps,
        ),
        timing=TimingMetrics(
            client_wall_ms=total_ms,
            local_draft_ms=metrics.total_edge_draft_time_ms,
            server_model_ms=metrics.total_server_verify_time_ms,
            http_rpc_ms=max(
                0.0,
                metrics.total_network_time_ms
                + metrics.total_server_verify_time_ms
                - net_stats["total_simulated_overhead_ms"],
            ),
            simulated_ul_ms=net_stats["total_simulated_uplink_delay_ms"],
            simulated_dl_ms=net_stats["total_simulated_downlink_delay_ms"],
            simulated_network_ms=net_stats["total_simulated_overhead_ms"],
            avg_rtt_ms=metrics.average_rtt_ms,
            critical_path_wait_ms=metrics.average_rtt_ms,
        ),
        speculative=SpeculativeMetrics(
            rounds=metrics.total_rounds,
            k=k,
            drafted_tokens=metrics.total_drafted_tokens,
            accepted_draft_tokens=accepted,
            correction_tokens=correction_tokens,
            generated_per_round=metrics.generated_tokens / metrics.total_rounds if metrics.total_rounds else 0.0,
            accepted_draft_per_round=net_useful,
            acceptance_rate=metrics.acceptance_ratio,
            wasted_draft_tokens=metrics.wasted_drafted_tokens,
        ),
        async_detail=AsyncDetailMetrics(),
        verify_runtime=VerifyRuntimeMetrics(
            avg_prefix_len=avg_verify_prefix_len,
            avg_draft_len=avg_verify_draft_len,
            avg_input_len=avg_verify_input_len,
            prefix_caching=_first([s.get("enable_prefix_caching") for s in metrics.round_details]),
            enforce_eager=_first([s.get("enforce_eager") for s in metrics.round_details]),
            attention_backend=_first([s.get("attention_backend") for s in metrics.round_details]),
            vllm_version=_first([s.get("vllm_version") for s in metrics.round_details]),
        ),
        raw={"round_details": metrics.round_details, "network": net_stats},
    )


def run_tree_spec_case(
    tree_async_client: TreeAsyncEdgeClient,
    throttled: ThrottledCloudClient,
    prompt_meta,
    text: str,
    k: int,
) -> Optional[ExperimentResult]:
    throttled.reset_stats()
    metrics = _tree_async_speculative(tree_async_client, text, k)
    if not metrics or metrics.generated_tokens <= 0:
        return None
    net_stats = throttled.get_stats_dict()
    total_ms = metrics.total_latency_ms
    tps = metrics.generated_tokens / (total_ms / 1000)
    accepted = metrics.total_accepted_tokens
    net_useful = accepted / metrics.total_rounds if metrics.total_rounds else 0
    correction_tokens = max(0, metrics.generated_tokens - accepted)
    method_name = f"tree_k{k}_b{tree_async_client.branch_width}"
    branch_width = _avg([s.get("tree_branch_width", 0) for s in metrics.slot_details])
    selected_offset = _avg([s.get("selected_offset", 0) for s in metrics.slot_details])
    stale_branches = _avg([s.get("stale_branches", 0) for s in metrics.slot_details])
    base_accept = _avg([s.get("base_accepted", 0) for s in metrics.slot_details])
    base_draft_ms = _avg([s.get("base_draft_ms", 0) for s in metrics.slot_details])
    branch_draft_ms = _avg([s.get("branch_draft_ms", 0) for s in metrics.slot_details])
    active_spec_branches = _avg([s.get("active_spec_branches", 0) for s in metrics.slot_details])
    exposed_branch_ms = _avg([s.get("exposed_branch_ms", 0) for s in metrics.slot_details])
    ready_branch_count = _avg([max(0, s.get("tree_branch_width", 0) - 1) for s in metrics.slot_details])
    reused_branch_count = _avg([1 if s.get("selected_offset", 0) > 0 else 0 for s in metrics.slot_details])
    prefetched_tokens = _avg([s.get("prefetched_tokens", 0) for s in metrics.slot_details])
    base_wait_ms = _avg([s.get("base_wait_ms", 0) for s in metrics.slot_details])
    total_wait_ms = _avg([s.get("total_wait_ms", 0) for s in metrics.slot_details])
    spec_verify_wall_ms = _avg([s.get("spec_verify_wall_ms", 0) for s in metrics.slot_details])
    avg_verify_prefix_len = _avg([s.get("verify_prefix_len", 0) or 0 for s in metrics.slot_details])
    avg_verify_draft_len = _avg([s.get("verify_draft_len", 0) or 0 for s in metrics.slot_details])
    avg_verify_input_len = _avg([s.get("verify_input_len", 0) or 0 for s in metrics.slot_details])
    logger.info(
        "    %s: %d tok  %5.0f ms  %5.1f tok/s  accept=%4.1f%%  net_useful=%.2f tok/round  "
        "rounds=%d  rtt=%d ms  bubble=%.0fms  offset=%.1f  verify_prefix=%.0f  verify_input=%.0f",
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
        avg_verify_prefix_len,
        avg_verify_input_len,
    )
    logger.info(
        "      tree_diag: base_accept=%.2f  base_draft=%.0fms  branch_draft=%.0fms  "
        "exposed_branch=%.0fms  active_spec=%.1f  base_wait=%.0fms  total_wait=%.0fms  "
        "spec_verify_wall=%.0fms  stale=%.1f",
        base_accept,
        base_draft_ms,
        branch_draft_ms,
        exposed_branch_ms,
        active_spec_branches,
        base_wait_ms,
        total_wait_ms,
        spec_verify_wall_ms,
        stale_branches,
    )
    return ExperimentResult(
        method=method_name,
        method_family="tree_async",
        network=throttled.condition.name,
        prompt_type=prompt_meta["type"],
        prompt_id=prompt_meta["id"],
        output=OutputMetrics(
            tokens_generated=metrics.generated_tokens,
            total_time_ms=total_ms,
            tokens_per_second=tps,
        ),
        timing=TimingMetrics(
            client_wall_ms=total_ms,
            local_draft_ms=metrics.total_edge_draft_time_ms,
            server_model_ms=metrics.total_server_verify_time_ms,
            http_rpc_ms=max(
                0.0,
                metrics.total_network_time_ms
                + metrics.total_server_verify_time_ms
                - net_stats["total_simulated_overhead_ms"],
            ),
            simulated_ul_ms=net_stats["total_simulated_uplink_delay_ms"],
            simulated_dl_ms=net_stats["total_simulated_downlink_delay_ms"],
            simulated_network_ms=net_stats["total_simulated_overhead_ms"],
            avg_rtt_ms=metrics.average_rtt_ms,
            critical_path_wait_ms=metrics.avg_bubble_ms,
        ),
        speculative=SpeculativeMetrics(
            rounds=metrics.total_rounds,
            k=k,
            drafted_tokens=metrics.total_drafted_tokens,
            accepted_draft_tokens=accepted,
            correction_tokens=correction_tokens,
            generated_per_round=metrics.generated_tokens / metrics.total_rounds if metrics.total_rounds else 0.0,
            accepted_draft_per_round=net_useful,
            acceptance_rate=metrics.acceptance_ratio,
            wasted_draft_tokens=max(0, metrics.total_drafted_tokens - accepted),
        ),
        async_detail=AsyncDetailMetrics(
            launched_branch_count=active_spec_branches,
            ready_branch_count=ready_branch_count,
            reused_branch_count=reused_branch_count,
            prefetched_tokens=prefetched_tokens,
            exposed_branch_ms=exposed_branch_ms,
            selected_offset=selected_offset,
            base_accepted=base_accept,
            stale_branch_count=stale_branches,
        ),
        verify_runtime=VerifyRuntimeMetrics(
            avg_prefix_len=avg_verify_prefix_len,
            avg_draft_len=avg_verify_draft_len,
            avg_input_len=avg_verify_input_len,
            prefix_caching=_first([s.get("enable_prefix_caching") for s in metrics.slot_details]),
            enforce_eager=_first([s.get("enforce_eager") for s in metrics.slot_details]),
            attention_backend=_first([s.get("attention_backend") for s in metrics.slot_details]),
            vllm_version=_first([s.get("vllm_version") for s in metrics.slot_details]),
        ),
        raw={
            "slot_details": metrics.slot_details,
            "network": net_stats,
            "avg_branch_width": branch_width,
            "avg_base_draft_ms": base_draft_ms,
            "avg_branch_draft_ms": branch_draft_ms,
            "avg_base_wait_ms": base_wait_ms,
            "avg_total_wait_ms": total_wait_ms,
            "avg_spec_verify_wall_ms": spec_verify_wall_ms,
        },
    )


def _avg(lst): return sum(lst) / len(lst) if lst else 0.0


def _first(lst):
    for item in lst:
        if item is not None:
            return item
    return None


def _build_summary(results: List[ExperimentResult], methods_to_show: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for net in [c.name for c in NetworkCondition.all_profiles()]:
        direct_rows = [r for r in results if r.method == "direct" and r.network == net]
        direct_tps = _avg([r.output.tokens_per_second for r in direct_rows])
        methods: Dict[str, Any] = {
            "direct": {
                "tokens_per_second": direct_tps,
                "total_time_ms": _avg([r.output.total_time_ms for r in direct_rows]),
                "server_model_ms": _avg([r.timing.server_model_ms for r in direct_rows]),
                "http_rpc_ms": _avg([r.timing.http_rpc_ms for r in direct_rows]),
                "simulated_network_ms": _avg([r.timing.simulated_network_ms for r in direct_rows]),
            }
        }
        for method in methods_to_show:
            rows = [r for r in results if r.method == method and r.network == net]
            tps = _avg([r.output.tokens_per_second for r in rows])
            methods[method] = {
                "tokens_per_second": tps,
                "speedup_vs_direct": tps / direct_tps if direct_tps else 0.0,
                "total_time_ms": _avg([r.output.total_time_ms for r in rows]),
                "rounds": _avg([r.speculative.rounds for r in rows]),
                "acceptance_rate": _avg([r.speculative.acceptance_rate for r in rows]),
                "accepted_draft_per_round": _avg([
                    r.speculative.accepted_draft_per_round for r in rows
                ]),
                "generated_per_round": _avg([r.speculative.generated_per_round for r in rows]),
                "local_draft_ms": _avg([r.timing.local_draft_ms for r in rows]),
                "server_model_ms": _avg([r.timing.server_model_ms for r in rows]),
                "avg_rtt_ms": _avg([r.timing.avg_rtt_ms for r in rows]),
                "simulated_network_ms": _avg([r.timing.simulated_network_ms for r in rows]),
                "launched_branch_count": _avg([
                    r.async_detail.launched_branch_count for r in rows
                ]),
                "ready_branch_count": _avg([r.async_detail.ready_branch_count for r in rows]),
                "reused_branch_count": _avg([r.async_detail.reused_branch_count for r in rows]),
                "prefetched_tokens": _avg([r.async_detail.prefetched_tokens for r in rows]),
                "exposed_branch_ms": _avg([r.async_detail.exposed_branch_ms for r in rows]),
            }
        summary[net] = methods
    return summary


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

    # ── Warmup: heat draft GPU and verify GPU via the actual verify path ──
    logger.info("Warming up draft model and verify server ...")
    warmup_prompt = prompts[0][0]["text"]
    warmup_prefix = list(draft_generator.tokenizer.encode(warmup_prompt))
    for _ in range(3):
        wr = draft_generator.generate_draft_tokens(
            DraftRequest(verified_prefix=warmup_prefix, num_draft_tokens=7),
            temperature=0.0,
        )
        try:
            base_client.verify(EdgeRequest(
                request_id="warmup",
                round_id=0,
                prefix_ids=warmup_prefix,
                draft_ids=wr.draft_token_ids,
                draft_logprobs=wr.logprobs,
                edge_draft_time_ms=0.0,
            ))
        except Exception:
            pass
    try:
        DIRECT_SESSION.post(
            f"{SERVER_URL}/generate",
            json={"prompt": warmup_prompt, "max_tokens": 8, "temperature": 0.0},
            timeout=60.0,
        )
    except Exception:
        pass
    logger.info("Warmup complete.")

    results: List[ExperimentResult] = []
    tree_method_suffix = f"b{tree_async_client.branch_width}"

    for condition in [NetworkCondition.good(), NetworkCondition.medium()]:
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
        d_tps = _avg([r.output.tokens_per_second for r in results if r.method == "direct" and r.network == net])
        row = f"{net:<10}  {d_tps:>8.1f}"
        for mname in methods_to_show:
            s_tps = _avg([
                r.output.tokens_per_second
                for r in results
                if r.method == mname and r.network == net
            ])
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

    raw_file = output_dir / "quick_test_rounds.jsonl"
    with open(raw_file, "w") as f:
        for result in results:
            base = {
                "method": result.method,
                "method_family": result.method_family,
                "network": result.network,
                "prompt_type": result.prompt_type,
                "prompt_id": result.prompt_id,
            }
            details = (
                result.raw.get("round_details")
                or result.raw.get("slot_details")
                or [result.raw.get("direct_timing", {})]
            )
            for index, detail in enumerate(details):
                f.write(json.dumps({**base, "detail_index": index, "detail": detail}) + "\n")

    summary_file = output_dir / "quick_test_summary.json"
    summary = _build_summary(results, methods_to_show)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\nResults saved to: %s", out_file)
    logger.info("Round details saved to: %s", raw_file)
    logger.info("Summary saved to: %s", summary_file)
    return True


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
