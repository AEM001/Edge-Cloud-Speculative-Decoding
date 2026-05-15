import argparse
import json
import logging
import os
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
from core.draft_generator import VLLMDraftGenerator
from core.protocol import DraftRequest, EdgeRequest
from experiments.network_conditions import NetworkCondition, ThrottledCloudClient
from core.model_manager import VLLMModelManager
from experiments.prompt_loader import load_prompts
from experiments.tree_async_client import TreeAsyncEdgeClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Read settings from environment variables (set by quick.sh)
SERVER_URL = os.getenv("VERIFY_SERVER_URL", "http://localhost:6006")
DRAFT_MODEL_PATH = Path(os.getenv("DRAFT_MODEL_PATH", "/root/code/draft/models/Qwen2.5-1.5B-Instruct-AWQ"))
DRAFT_MODEL_NAME = os.getenv("DRAFT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
DRAFT_GPU_MEM = float(os.getenv("DRAFT_GPU_MEM", "0.4"))
DRAFT_MAX_LEN = int(os.getenv("DRAFT_MAX_LEN", "4096"))
DRAFT_GPU_ID = int(os.getenv("DRAFT_GPU_ID", "1"))

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
    branch_reused: bool = False
    reused_tokens: float = 0.0
    predraft_window_ms: float = 0.0
    reuse_prep_time_ms: float = 0.0


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


def _load_prompt_set(prompt_types: List[str], prompt_count: int):
    """Load prompts from specified types with given count per type."""
    prompts = []
    for src in prompt_types:
        loaded = load_prompts(source=src, count=prompt_count, min_length=200, max_length=500)
        prompts.extend((p, src) for p in loaded)
    type_counts = {t: sum(1 for _, pt in prompts if pt == t) for t in prompt_types}
    logger.info("Loaded %d prompts: %s", len(prompts), ", ".join(f"{count} {t}" for t, count in type_counts.items()))
    return prompts


def _direct_with_throttle(prompt: str, throttled: ThrottledCloudClient, max_tokens: int) -> DirectTiming:
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
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
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


def run_direct_case(prompt_meta, text: str, throttled: ThrottledCloudClient, max_tokens: int) -> Optional[ExperimentResult]:
    throttled.reset_stats()
    timing = _direct_with_throttle(text, throttled, max_tokens)
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
    avg_server_verify_ms = _avg([s.get("server_time_ms", 0) or 0 for s in metrics.round_details])
    avg_network_ms = (
        net_stats["total_simulated_overhead_ms"] / net_stats["num_calls"]
        if net_stats["num_calls"]
        else 0.0
    )
    logger.info(
        "    %s: %d tok  %5.0f ms  %5.1f tok/s  accept=%4.1f%%  net_useful=%.2f tok/round  "
        "rounds=%d  rtt=%d ms  verify_ms=%.0f  sim_net=%.0f  verify_prefix=%.0f  verify_input=%.0f",
        method_name,
        metrics.generated_tokens,
        total_ms,
        tps,
        metrics.acceptance_ratio * 100,
        net_useful,
        metrics.total_rounds,
        metrics.average_rtt_ms,
        avg_server_verify_ms,
        avg_network_ms,
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
            wasted_draft_tokens=max(0, metrics.total_drafted_tokens - accepted),
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
    
    # Simplified metrics from AsyncRequestMetrics
    branch_reused = metrics.branch_reused
    reused_tokens = metrics.reused_tokens / metrics.total_rounds if metrics.total_rounds else 0.0
    predraft_window_ms = metrics.predraft_window_ms / metrics.total_rounds if metrics.total_rounds else 0.0
    reuse_prep_time_ms = metrics.reuse_prep_time_ms / metrics.total_rounds if metrics.total_rounds else 0.0
    
    avg_network_ms = (
        net_stats["total_simulated_overhead_ms"] / net_stats["num_calls"]
        if net_stats["num_calls"]
        else 0.0
    )
    avg_server_verify_ms = metrics.total_server_verify_time_ms / metrics.total_rounds if metrics.total_rounds else 0.0
    
    logger.info(
        "    %s: %d tok  %5.0f ms  %5.1f tok/s  accept=%4.1f%%  net_useful=%.2f tok/round  "
        "rounds=%d  rtt=%d ms  verify_ms=%.0f  sim_net=%.0f  bubble=%.0fms",
        method_name,
        metrics.generated_tokens,
        total_ms,
        tps,
        metrics.acceptance_ratio * 100,
        net_useful,
        metrics.total_rounds,
        metrics.average_rtt_ms,
        avg_server_verify_ms,
        avg_network_ms,
        metrics.avg_bubble_ms,
    )
    logger.info(
        "      tree_diag: branch_reused=%s  reused_tokens=%.1f  predraft_window=%.0fms  reuse_prep=%.0fms",
        branch_reused,
        reused_tokens,
        predraft_window_ms,
        reuse_prep_time_ms,
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
            branch_reused=branch_reused,
            reused_tokens=reused_tokens,
            predraft_window_ms=predraft_window_ms,
            reuse_prep_time_ms=reuse_prep_time_ms,
        ),
        verify_runtime=VerifyRuntimeMetrics(),
        raw={
            "network": net_stats,
        },
    )


def _avg(lst): return sum(lst) / len(lst) if lst else 0.0


def _first(lst):
    for item in lst:
        if item is not None:
            return item
    return None


def _build_summary(results: List[ExperimentResult], methods_to_show: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"config": config}
    
    # Overall summary by network condition
    for net in config["network_conditions"]:
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
            }
        summary[net] = methods
    
    # Per-prompt-type breakdown
    for ptype in config["prompt_types"]:
        ptype_summary: Dict[str, Any] = {}
        for net in config["network_conditions"]:
            direct_rows = [r for r in results if r.method == "direct" and r.network == net and r.prompt_type == ptype]
            direct_tps = _avg([r.output.tokens_per_second for r in direct_rows])
            methods: Dict[str, Any] = {
                "direct": {
                    "tokens_per_second": direct_tps,
                    "total_time_ms": _avg([r.output.total_time_ms for r in direct_rows]),
                }
            }
            for method in methods_to_show:
                rows = [r for r in results if r.method == method and r.network == net and r.prompt_type == ptype]
                tps = _avg([r.output.tokens_per_second for r in rows])
                methods[method] = {
                    "tokens_per_second": tps,
                    "speedup_vs_direct": tps / direct_tps if direct_tps else 0.0,
                    "total_time_ms": _avg([r.output.total_time_ms for r in rows]),
                    "rounds": _avg([r.speculative.rounds for r in rows]),
                    "acceptance_rate": _avg([r.speculative.acceptance_rate for r in rows]),
                }
            ptype_summary[net] = methods
        summary[f"by_type_{ptype}"] = ptype_summary
    
    return summary


def run_quick_test(config: Dict[str, Any]):
    logger.info("=" * 70)
    logger.info("QUICK TEST  —  Direct (throttled) vs Speculative K=%s (throttled)", config["k_values"])
    logger.info("Draft model : %s", DRAFT_MODEL_NAME)
    logger.info("Config: max_tokens=%d, prompt_count=%d, prompt_types=%s, networks=%s",
                config["max_tokens"], config["prompt_count"], ",".join(config["prompt_types"]), ",".join(config["network_conditions"]))
    logger.info("Speculative: k_values=%s, tree_branch_width=%d, tree_branch_draft_length=%d",
                config["k_values"], config["tree_branch_width"], config["tree_branch_draft_length"])
    logger.info("=" * 70)

    prompts = _load_prompt_set(config["prompt_types"], config["prompt_count"])

    base_client = create_http_cloud_client(SERVER_URL, timeout=120.0)

    logger.info("Loading draft model on GPU %d ...", DRAFT_GPU_ID)
    model_manager = VLLMModelManager(DRAFT_MODEL_PATH, DRAFT_GPU_MEM, DRAFT_MAX_LEN, gpu_id=DRAFT_GPU_ID)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    edge_client = EdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=base_client,   # swapped per condition below
        max_new_tokens=config["max_tokens"],
        temperature=0.0,
    )
    tree_async_client = TreeAsyncEdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=base_client,   # swapped per condition below
        max_new_tokens=config["max_tokens"],
        temperature=0.0,
        branch_width=config["tree_branch_width"],
        branch_draft_length=config["tree_branch_draft_length"],
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
                prefix_ids=warmup_prefix,
                draft_ids=wr.draft_token_ids,
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

    # Build network conditions based on config
    conditions = []
    for net_name in config["network_conditions"]:
        if net_name == "good":
            conditions.append(NetworkCondition.good())
        elif net_name == "medium":
            conditions.append(NetworkCondition.medium())
        elif net_name == "bursty":
            conditions.append(NetworkCondition.bursty())
        else:
            logger.warning("Unknown network condition: %s", net_name)
    
    for condition in conditions:
        throttled = ThrottledCloudClient(base_client, condition)
        edge_client.cloud_client  = throttled
        tree_async_client.cloud_client = throttled

        logger.info("")
        logger.info("── Network: %s", condition)

        for prompt_data, ptype in prompts:
            pid  = prompt_data["id"]
            text = prompt_data["text"]
            prompt_meta = {"id": pid, "type": ptype}
            logger.info("  [%s | prompt %d]", ptype, pid)

            result = run_direct_case(prompt_meta, text, throttled, config["max_tokens"])
            if result:
                results.append(result)
            time.sleep(0.3)

            for k in config["k_values"]:
                spec_result = run_sync_spec_case(edge_client, throttled, prompt_meta, text, k)
                if spec_result:
                    results.append(spec_result)
                time.sleep(0.3)

            for k in config["k_values"]:
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
        [f"sync_k{k}" for k in config["k_values"]]
        + [f"tree_k{k}_{tree_method_suffix}" for k in config["k_values"]]
    )
    col_w = 28
    header = f"{'Network':<10}  {'direct':>8}" + "".join(
        f"  {m:>{col_w}}" for m in methods_to_show
    )
    logger.info(header)
    logger.info("-" * len(header))
    for net in config["network_conditions"]:
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

    # ── Per-prompt-type summary ──────────────────────────────────────────
    for ptype in config["prompt_types"]:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"SUMMARY  —  tok/s and speedup for {ptype}")
        logger.info("=" * 70)
        logger.info(header)
        logger.info("-" * len(header))
        for net in config["network_conditions"]:
            d_tps = _avg([r.output.tokens_per_second for r in results if r.method == "direct" and r.network == net and r.prompt_type == ptype])
            row = f"{net:<10}  {d_tps:>8.1f}"
            for mname in methods_to_show:
                s_tps = _avg([
                    r.output.tokens_per_second
                    for r in results
                    if r.method == mname and r.network == net and r.prompt_type == ptype
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
    summary = _build_summary(results, methods_to_show, config)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\nResults saved to: %s", out_file)
    logger.info("Round details saved to: %s", raw_file)
    logger.info("Summary saved to: %s", summary_file)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Run quick test with configurable settings")
    parser.add_argument("--network", type=str, nargs='+', default=["good", "medium"],
                        choices=["good", "medium", "bursty"],
                        help="Network condition(s) to test (default: good medium)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum tokens to generate (default: 128)")
    parser.add_argument("--prompt-count", type=int, default=10,
                        help="Number of prompts per type (default: 10)")
    parser.add_argument("--prompt-types", type=str, nargs='+', default=["gsm8k", "humaneval"],
                        choices=["gsm8k", "humaneval"],
                        help="Prompt type(s) to use (default: gsm8k humaneval)")
    parser.add_argument("--k-values", type=int, nargs='+', default=[8],
                        help="Draft length K values (default: 8)")
    parser.add_argument("--tree-branch-width", type=int, default=3,
                        help="Number of tree branches (default: 3)")
    parser.add_argument("--tree-branch-draft-length", type=int, default=8,
                        help="Pre-draft length for each tree branch (default: 8)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = {
        "max_tokens": args.max_tokens,
        "prompt_count": args.prompt_count,
        "prompt_types": args.prompt_types,
        "network_conditions": args.network,
        "k_values": args.k_values,
        "tree_branch_width": args.tree_branch_width,
        "tree_branch_draft_length": args.tree_branch_draft_length,
    }
    success = run_quick_test(config)
    sys.exit(0 if success else 1)
