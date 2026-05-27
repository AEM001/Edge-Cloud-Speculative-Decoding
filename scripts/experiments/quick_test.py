import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client.http_cloud_client import create_http_cloud_client
from core.draft_generator import VLLMDraftGenerator
from core.model_manager import VLLMModelManager
from core.protocol import DraftRequest, EdgeRequest
from experiments.network_conditions import NetworkCondition, ThrottledCloudClient
from experiments.prompt_loader import load_prompts
from experiments.tree_async_client import TreeAsyncEdgeClient


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVER_URL = os.getenv("VERIFY_SERVER_URL", "http://localhost:6006")
DRAFT_MODEL_PATH = Path(os.getenv("DRAFT_MODEL_PATH", "/root/code/draft/models/Qwen2.5-1.5B-Instruct-AWQ"))
DRAFT_MODEL_NAME = os.getenv("DRAFT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
DRAFT_GPU_MEM = float(os.getenv("DRAFT_GPU_MEM", "0.4"))
DRAFT_MAX_LEN = int(os.getenv("DRAFT_MAX_LEN", "32768"))
DRAFT_GPU_ID = int(os.getenv("DRAFT_GPU_ID", "1"))
REQUEST_TIMEOUT_SEC = float(os.getenv("QUICK_TEST_TIMEOUT_SEC", "600"))

OUTPUT_DIR = Path(__file__).parent / "outputs_quick"
RESULTS_PATH = OUTPUT_DIR / "quick_test_results.json"

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
    method_family: str
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


def load_prompt_set(prompt_types: List[str], prompt_count: int):
    prompts = []
    for source in prompt_types:
        if source.startswith(("longwriter", "longbench_v2")):
            loaded = load_prompts(source=source, count=prompt_count)
        else:
            loaded = load_prompts(source=source, count=prompt_count, min_length=200, max_length=500)
        prompts.extend((prompt, source) for prompt in loaded)

    counts = {source: sum(1 for _, ptype in prompts if ptype == source) for source in prompt_types}
    logger.info("Loaded %d prompts: %s", len(prompts), ", ".join(f"{n} {k}" for k, n in counts.items()))
    return prompts


def run_direct_case(prompt_meta: Dict[str, Any], prompt: str, throttled: ThrottledCloudClient, max_tokens: int):
    throttled.reset_stats()
    timing = direct_generate_with_throttle(prompt, throttled, max_tokens)
    if timing.tokens <= 0:
        return None

    tps = timing.tokens / (timing.total_ms / 1000)
    logger.info(
        "    direct: %d tok  %.0f ms  %.1f tok/s  server=%.0f ms  http=%.0f ms  sim_net=%.0f ms",
        timing.tokens,
        timing.total_ms,
        tps,
        timing.server_ms,
        timing.http_ms,
        timing.overhead_ms,
    )

    return ExperimentResult(
        method="direct",
        method_family="direct",
        network=throttled.condition.name,
        prompt_type=prompt_meta["type"],
        prompt_id=prompt_meta["id"],
        output=OutputMetrics(timing.tokens, timing.total_ms, tps),
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


def direct_generate_with_throttle(prompt: str, throttled: ThrottledCloudClient, max_tokens: int) -> DirectTiming:
    condition = throttled.condition
    one_way_ms, dl_mbps, ul_mbps = condition.current_link_params(time.perf_counter() - throttled._start_wall)
    ul_bytes = len(prompt.encode()) + 64
    ul_delay = one_way_ms + NetworkCondition._payload_delay_ms(ul_bytes, ul_mbps)
    time.sleep(ul_delay / 1000.0)

    start = time.perf_counter()
    try:
        response = DIRECT_SESSION.post(
            f"{SERVER_URL}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.error("Direct generation failed: %s", exc)
        return DirectTiming(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    http_ms = (time.perf_counter() - start) * 1000
    text = data.get("text", "")
    tokens = int(data.get("tokens_generated", 0))
    server_ms = float(data.get("generation_time_ms", 0.0))

    one_way_ms, dl_mbps, _ = condition.current_link_params(time.perf_counter() - throttled._start_wall)
    dl_bytes = len(text.encode()) + 64
    dl_delay = one_way_ms + NetworkCondition._payload_delay_ms(dl_bytes, dl_mbps)
    time.sleep(dl_delay / 1000.0)

    return DirectTiming(
        tokens=tokens,
        total_ms=ul_delay + http_ms + dl_delay,
        overhead_ms=ul_delay + dl_delay,
        server_ms=server_ms,
        http_ms=http_ms,
        ul_ms=ul_delay,
        dl_ms=dl_delay,
    )


def run_tree_case(
    tree_client: TreeAsyncEdgeClient,
    throttled: ThrottledCloudClient,
    prompt_meta: Dict[str, Any],
    prompt: str,
    k: int,
):
    throttled.reset_stats()
    try:
        metrics = tree_client.generate(prompt=prompt, policy=lambda _rid, _toks: k, policy_name=f"TreeK{k}")
    except Exception as exc:
        logger.error("Tree async K=%d failed: %s", k, exc)
        return None

    if not metrics or metrics.generated_tokens <= 0:
        return None

    net_stats = throttled.get_stats_dict()
    total_ms = metrics.total_latency_ms
    tps = metrics.generated_tokens / (total_ms / 1000)
    accepted = metrics.total_accepted_tokens
    correction_tokens = max(0, metrics.generated_tokens - accepted)
    accepted_per_round = accepted / metrics.total_rounds if metrics.total_rounds else 0.0
    avg_network_ms = net_stats["total_simulated_overhead_ms"] / net_stats["num_calls"] if net_stats["num_calls"] else 0.0
    avg_verify_ms = metrics.total_server_verify_time_ms / metrics.total_rounds if metrics.total_rounds else 0.0
    reused_tokens = metrics.reused_tokens / metrics.total_rounds if metrics.total_rounds else 0.0
    predraft_window_ms = metrics.predraft_window_ms / metrics.total_rounds if metrics.total_rounds else 0.0
    reuse_prep_time_ms = metrics.reuse_prep_time_ms / metrics.total_rounds if metrics.total_rounds else 0.0
    method = f"tree_k{k}_b{tree_client.branch_width}"

    logger.info(
        "    %s: %d tok  %.0f ms  %.1f tok/s  accept=%.1f%%  rounds=%d  rtt=%.0f ms  verify=%.0f ms  sim_net=%.0f ms",
        method,
        metrics.generated_tokens,
        total_ms,
        tps,
        metrics.acceptance_ratio * 100,
        metrics.total_rounds,
        metrics.average_rtt_ms,
        avg_verify_ms,
        avg_network_ms,
    )
    logger.info(
        "      tree_diag: branch_reused=%s  reused_tokens=%.1f  predraft_window=%.0f ms  reuse_prep=%.0f ms",
        metrics.branch_reused,
        reused_tokens,
        predraft_window_ms,
        reuse_prep_time_ms,
    )

    return ExperimentResult(
        method=method,
        method_family="tree_async",
        network=throttled.condition.name,
        prompt_type=prompt_meta["type"],
        prompt_id=prompt_meta["id"],
        output=OutputMetrics(metrics.generated_tokens, total_ms, tps),
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
            accepted_draft_per_round=accepted_per_round,
            acceptance_rate=metrics.acceptance_ratio,
            wasted_draft_tokens=max(0, metrics.total_drafted_tokens - accepted),
        ),
        async_detail=AsyncDetailMetrics(
            branch_reused=metrics.branch_reused,
            reused_tokens=reused_tokens,
            predraft_window_ms=predraft_window_ms,
            reuse_prep_time_ms=reuse_prep_time_ms,
        ),
        verify_runtime=VerifyRuntimeMetrics(),
        raw={"network": net_stats},
    )


def warmup(draft_generator: VLLMDraftGenerator, base_client, prompt: str) -> None:
    logger.info("Warming up draft model and verify server ...")
    warmup_prefix = list(draft_generator.tokenizer.encode(prompt))
    for _ in range(3):
        draft = draft_generator.generate_draft_tokens(
            DraftRequest(verified_prefix=warmup_prefix, num_draft_tokens=7),
            temperature=0.0,
        )
        try:
            base_client.verify(
                EdgeRequest(
                    request_id="warmup",
                    prefix_ids=warmup_prefix,
                    draft_ids=draft.draft_token_ids,
                )
            )
        except Exception:
            pass

    try:
        DIRECT_SESSION.post(
            f"{SERVER_URL}/generate",
            json={"prompt": prompt, "max_tokens": 8, "temperature": 0.0},
            timeout=60.0,
        )
    except Exception:
        pass
    logger.info("Warmup complete.")


def save_results(results: List[ExperimentResult], config: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "results": [asdict(result) for result in results],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Results saved to: %s", RESULTS_PATH)
    logger.info("Run `python3 scripts/experiments/analyze_quick_run.py` to build summaries.")


def run_quick_test(config: Dict[str, Any]) -> bool:
    logger.info("=" * 70)
    logger.info("QUICK TEST - direct vs tree async on good network")
    logger.info("Draft model: %s", DRAFT_MODEL_NAME)
    logger.info(
        "Config: max_tokens=%d, prompt_count=%d, prompt_types=%s, k=%d, branch_width=%d, branch_draft_length=%d",
        config["max_tokens"],
        config["prompt_count"],
        ",".join(config["prompt_types"]),
        config["k"],
        config["tree_branch_width"],
        config["tree_branch_draft_length"],
    )
    logger.info("=" * 70)

    prompts = load_prompt_set(config["prompt_types"], config["prompt_count"])
    base_client = create_http_cloud_client(SERVER_URL, timeout=REQUEST_TIMEOUT_SEC)

    logger.info("Loading draft model on GPU %d ...", DRAFT_GPU_ID)
    model_manager = VLLMModelManager(DRAFT_MODEL_PATH, DRAFT_GPU_MEM, DRAFT_MAX_LEN, gpu_id=DRAFT_GPU_ID)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    tree_client = TreeAsyncEdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=base_client,
        max_new_tokens=config["max_tokens"],
        temperature=0.0,
        branch_width=config["tree_branch_width"],
        branch_draft_length=config["tree_branch_draft_length"],
    )
    logger.info("Draft model loaded.")

    warmup(draft_generator, base_client, prompts[0][0]["text"])

    condition = NetworkCondition.good()
    throttled = ThrottledCloudClient(base_client, condition)
    tree_client.cloud_client = throttled

    logger.info("")
    logger.info("Network: %s", condition)

    results: List[ExperimentResult] = []
    for prompt_data, prompt_type in prompts:
        prompt_meta = {"id": prompt_data["id"], "type": prompt_type}
        prompt = prompt_data["text"]
        logger.info("  [%s | prompt %d]", prompt_type, prompt_data["id"])

        direct_result = run_direct_case(prompt_meta, prompt, throttled, config["max_tokens"])
        if direct_result:
            results.append(direct_result)
        time.sleep(0.3)

        tree_result = run_tree_case(tree_client, throttled, prompt_meta, prompt, config["k"])
        if tree_result:
            results.append(tree_result)
        time.sleep(0.3)

    save_results(results, config)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Run direct and tree-async quick tests on the good network profile.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens to generate.")
    parser.add_argument("--prompt-count", type=int, default=10, help="Number of prompts per type.")
    parser.add_argument(
        "--prompt-types",
        type=str,
        nargs="+",
        default=["longbench_v2:short"],
        help=(
            "Prompt type(s): gsm8k, humaneval, longwriter, "
            "longwriter_single_turn:input_4k/input_6k/input_8k/input_10k, "
            "or longbench_v2:short/medium/long/train."
        ),
    )
    parser.add_argument("--k", type=int, default=17, help="Tree base draft length.")
    parser.add_argument("--tree-branch-width", type=int, default=3, help="Number of tree branches.")
    parser.add_argument("--tree-branch-draft-length", type=int, default=15, help="Pre-draft length per branch.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_quick_test(
        {
            "network": "good",
            "methods": ["direct", "tree_async"],
            "max_tokens": args.max_tokens,
            "prompt_count": args.prompt_count,
            "prompt_types": args.prompt_types,
            "k": args.k,
            "tree_branch_width": args.tree_branch_width,
            "tree_branch_draft_length": args.tree_branch_draft_length,
        }
    )
    sys.exit(0 if success else 1)
