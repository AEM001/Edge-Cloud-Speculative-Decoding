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
from client.specextend_edge_client import SpecExtendEdgeClient
from core.protocol import SpecExtendRequest
from core.qwen_specextend_backend import (
    QwenBackendConfig,
    QwenSpecExtendDraftBackend,
    dtype_from_env,
)
from experiments.network_conditions import NetworkCondition, ThrottledCloudClient
from experiments.prompt_loader import load_prompts


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVER_URL = os.getenv("VERIFY_SERVER_URL", "http://localhost:3471")
DRAFT_MODEL_PATH = Path(os.getenv("DRAFT_MODEL_PATH", "/root/autodl-tmp/Qwen3-1.7B"))
DRAFT_GPU_ID = os.getenv("DRAFT_GPU_ID", "1")
DRAFT_DEVICE = os.getenv("DRAFT_DEVICE", f"cuda:{DRAFT_GPU_ID}")
DRAFT_DTYPE = dtype_from_env(os.getenv("DRAFT_DTYPE", "fp16"))
DRAFT_MAX_LEN = int(os.getenv("DRAFT_MAX_LEN", "32768"))
DRAFT_ATTN_IMPLEMENTATION = os.getenv("DRAFT_ATTN_IMPLEMENTATION", "sdpa")
REQUEST_TIMEOUT_SEC = float(os.getenv("QUICK_TEST_TIMEOUT_SEC", "600"))

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "quick_benchmark_config.json"
DIRECT_SESSION = requests.Session()
DIRECT_SESSION.trust_env = False


DEFAULT_CONFIG: Dict[str, Any] = {
    "network": "good",
    "methods": ["direct", "specextend"],
    "max_tokens": 256,
    "prompt_count": 1,
    "prompt_types": ["pg19"],
    "dataset_split": "pg19_512",
    "prompt_input_tokens": 2048,
    "async_pipeline": os.getenv("SPECEXTEND_ASYNC_PIPELINE", "1"),
    "pipeline_offsets": os.getenv("SPECEXTEND_PIPELINE_OFFSETS", "full,half"),
    "draft_mode": "branching",
    "draft_length": 8,
    "draft_tree_nodes": 32,
    "draft_tree_max_depth": 6,
    "retrieval_chunk_size": 64,
    "retrieve_top_k": 16,
    "retrieve_every_n_steps": 16,
    "spec_profiles": [
        {
            "name": "specextend",
            "label": "SpecExtend sparse KV",
        },
    ],
}


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


@dataclass
class SpeculativeMetrics:
    rounds: int = 0
    draft_length: int = 0
    accepted_draft_tokens: int = 0
    correction_tokens: int = 0
    generated_per_round: float = 0.0
    acceptance_length: float = 0.0


@dataclass
class VerifyRuntimeMetrics:
    backend: Optional[str] = None
    specextend_tree_verify: Optional[bool] = None
    attention_scores: Optional[bool] = None


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
    verify_runtime: VerifyRuntimeMetrics
    raw: Dict[str, Any]


@dataclass
class DirectTiming:
    tokens: int
    total_ms: float
    server_ms: float
    http_ms: float
    ul_ms: float
    dl_ms: float


def get_results_path() -> Path:
    return OUTPUT_DIR / f"quick_test_results_{time.strftime('%Y%m%d_%H%M%S')}.json"


def load_config_file(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return dict(DEFAULT_CONFIG)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    loaded = json.loads(path.read_text())
    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    if "spec_profiles" not in loaded:
        config["spec_profiles"] = DEFAULT_CONFIG["spec_profiles"]
    normalize_spec_config(config)
    return config


def normalize_spec_config(config: Dict[str, Any]) -> None:
    if "draft_mode" not in config:
        config["draft_mode"] = DEFAULT_CONFIG["draft_mode"]
    if "draft_length" not in config:
        config["draft_length"] = DEFAULT_CONFIG["draft_length"]
    if "draft_tree_nodes" not in config:
        config["draft_tree_nodes"] = DEFAULT_CONFIG["draft_tree_nodes"]
    if "draft_tree_max_depth" not in config:
        config["draft_tree_max_depth"] = DEFAULT_CONFIG["draft_tree_max_depth"]


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    overrides = {
        "max_tokens": args.max_tokens,
        "prompt_count": args.prompt_count,
        "prompt_types": args.prompt_types,
        "dataset_split": args.dataset_split,
        "prompt_input_tokens": args.prompt_input_tokens,
        "draft_mode": args.draft_mode,
        "draft_length": args.draft_length,
        "draft_tree_nodes": args.draft_tree_nodes,
        "draft_tree_max_depth": args.draft_tree_max_depth,
        "retrieval_chunk_size": args.retrieval_chunk_size,
        "retrieve_top_k": args.retrieve_top_k,
        "retrieve_every_n_steps": args.retrieve_every_n_steps,
        "methods": args.methods,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    normalize_spec_config(config)
    return config


def apply_draft_mode_config(config: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> None:
    source = config if profile is None else {**config, **profile}
    draft_mode = str(source.get("draft_mode", DEFAULT_CONFIG["draft_mode"])).strip().lower()
    if draft_mode not in {"linear", "branching"}:
        raise ValueError(f"Unsupported draft_mode: {draft_mode}")

    os.environ["DRAFT_TREE_MODE"] = draft_mode
    os.environ["DRAFT_TREE_NODES"] = str(int(source.get("draft_tree_nodes", DEFAULT_CONFIG["draft_tree_nodes"])))
    os.environ["DRAFT_TREE_MAX_DEPTH"] = str(
        int(source.get("draft_tree_max_depth", DEFAULT_CONFIG["draft_tree_max_depth"]))
    )


def load_prompt_set(prompt_types: List[str], prompt_count: int, split: str = "test"):
    prompts = []
    for source in prompt_types:
        loaded = load_prompts(source=source, count=prompt_count, split=split)
        prompts.extend((prompt, source) for prompt in loaded)
    logger.info("Loaded %d prompts", len(prompts))
    return prompts


def truncate_prompts_to_tokens(prompts, tokenizer, max_input_tokens: int):
    if max_input_tokens <= 0:
        return prompts

    truncated = []
    for prompt_data, prompt_type in prompts:
        input_ids = tokenizer.encode(prompt_data["text"])
        token_count = len(input_ids)
        if token_count > max_input_tokens:
            prompt_data = dict(prompt_data)
            prompt_data["text"] = tokenizer.decode(input_ids[:max_input_tokens])
            prompt_data["prompt_input_tokens"] = max_input_tokens
            prompt_data["original_prompt_input_tokens"] = token_count
        else:
            prompt_data = dict(prompt_data)
            prompt_data["prompt_input_tokens"] = token_count
            prompt_data["original_prompt_input_tokens"] = token_count
        truncated.append((prompt_data, prompt_type))

    logger.info("Using fixed input length up to %d tokens", max_input_tokens)
    return truncated


def direct_generate_with_throttle(prompt: str, throttled: ThrottledCloudClient, max_tokens: int) -> DirectTiming:
    condition = throttled.condition
    one_way_ms, dl_mbps, ul_mbps = condition.current_link_params(time.perf_counter() - throttled._start_wall)
    ul_bytes = len(prompt.encode()) + 64
    ul_delay = one_way_ms + NetworkCondition._payload_delay_ms(ul_bytes, ul_mbps)
    time.sleep(ul_delay / 1000.0)

    start = time.perf_counter()
    response = DIRECT_SESSION.post(
        f"{SERVER_URL}/generate",
        json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=REQUEST_TIMEOUT_SEC,
    )
    response.raise_for_status()
    data = response.json()
    http_ms = (time.perf_counter() - start) * 1000

    text = data.get("text", "")
    one_way_ms, dl_mbps, _ = condition.current_link_params(time.perf_counter() - throttled._start_wall)
    dl_bytes = len(text.encode()) + 64
    dl_delay = one_way_ms + NetworkCondition._payload_delay_ms(dl_bytes, dl_mbps)
    time.sleep(dl_delay / 1000.0)

    return DirectTiming(
        tokens=int(data.get("tokens_generated", 0)),
        total_ms=ul_delay + http_ms + dl_delay,
        server_ms=float(data.get("generation_time_ms", 0.0)),
        http_ms=http_ms,
        ul_ms=ul_delay,
        dl_ms=dl_delay,
    )


def run_direct_case(prompt_meta: Dict[str, Any], prompt: str, throttled: ThrottledCloudClient, max_tokens: int):
    throttled.reset_stats()
    try:
        timing = direct_generate_with_throttle(prompt, throttled, max_tokens)
    except Exception as exc:
        logger.error("Direct generation failed: %s", exc)
        return None
    if timing.tokens <= 0:
        return None

    tps = timing.tokens / (timing.total_ms / 1000)
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
            simulated_network_ms=timing.ul_ms + timing.dl_ms,
        ),
        speculative=SpeculativeMetrics(),
        verify_runtime=VerifyRuntimeMetrics(),
        raw={"direct_timing": asdict(timing)},
    )


def run_specextend_case(
    client: SpecExtendEdgeClient,
    throttled: ThrottledCloudClient,
    prompt_meta: Dict[str, Any],
    prompt: str,
    draft_length: int,
    profile: Dict[str, Any],
):
    throttled.reset_stats()
    try:
        metrics = client.generate(prompt)
    except Exception as exc:
        logger.error("SpecExtend failed: %s", exc)
        return None
    if metrics.generated_tokens <= 0:
        return None

    net_stats = throttled.get_stats_dict()
    total_ms = metrics.total_latency_ms
    tps = metrics.generated_tokens / (total_ms / 1000)
    correction_tokens = max(0, metrics.generated_tokens - metrics.total_accepted_tokens)
    acceptance_length = metrics.total_accepted_tokens / metrics.total_rounds if metrics.total_rounds else 0.0

    return ExperimentResult(
        method=profile["name"],
        method_family="specextend",
        network=throttled.condition.name,
        prompt_type=prompt_meta["type"],
        prompt_id=prompt_meta["id"],
        output=OutputMetrics(metrics.generated_tokens, total_ms, tps),
        timing=TimingMetrics(
            client_wall_ms=total_ms,
            local_draft_ms=metrics.total_edge_draft_time_ms,
            server_model_ms=metrics.total_server_verify_time_ms,
            server_total_ms=metrics.total_server_verify_time_ms,
            simulated_ul_ms=net_stats["total_simulated_uplink_delay_ms"],
            simulated_dl_ms=net_stats["total_simulated_downlink_delay_ms"],
            simulated_network_ms=net_stats["total_simulated_overhead_ms"],
            avg_rtt_ms=metrics.total_network_time_ms / metrics.total_rounds if metrics.total_rounds else 0.0,
        ),
        speculative=SpeculativeMetrics(
            rounds=metrics.total_rounds,
            draft_length=draft_length,
            accepted_draft_tokens=metrics.total_accepted_tokens,
            correction_tokens=correction_tokens,
            generated_per_round=metrics.generated_tokens / metrics.total_rounds if metrics.total_rounds else 0.0,
            acceptance_length=acceptance_length,
        ),
        verify_runtime=VerifyRuntimeMetrics(
            backend="custom_qwen3",
            specextend_tree_verify=str(profile.get("draft_mode", "")).lower() == "branching",
            attention_scores=metrics.retrieval_updates > 0,
        ),
        raw={
            "profile": profile,
            "network": net_stats,
            "selected_chunk_ids": metrics.selected_chunk_ids,
            "retrieval_updates": metrics.retrieval_updates,
            "round_details": metrics.round_details,
        },
    )


def build_specextend_client(
    draft_backend: QwenSpecExtendDraftBackend,
    cloud_verify,
    config: Dict[str, Any],
    profile: Dict[str, Any],
) -> SpecExtendEdgeClient:
    draft_length = int(profile.get("draft_length", config["draft_length"]))
    return SpecExtendEdgeClient(
        draft_backend=draft_backend,
        cloud_verify=cloud_verify,
        max_new_tokens=int(profile.get("max_tokens", config["max_tokens"])),
        draft_length=draft_length,
        retrieval_chunk_size=int(profile.get("retrieval_chunk_size", config["retrieval_chunk_size"])),
        retrieve_top_k=int(profile.get("retrieve_top_k", config["retrieve_top_k"])),
        retrieve_every_n_steps=int(profile.get("retrieve_every_n_steps", config["retrieve_every_n_steps"])),
        retrieve_on_first_round=bool(profile.get("retrieve_on_first_round", config.get("retrieve_on_first_round", False))),
    )


def selected_spec_profiles(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    requested = set(config.get("methods") or [])
    profiles = []
    for profile in config.get("spec_profiles", []):
        if not requested or profile["name"] in requested:
            profiles.append(profile)
    return profiles


def warmup(draft_backend: QwenSpecExtendDraftBackend, cloud_client, prompt: str) -> None:
    logger.info("Warming up custom draft and target backends ...")
    prompt_ids = list(draft_backend.tokenizer.encode(prompt))
    draft = draft_backend.build_draft(
        verified_prefix=prompt_ids,
        correction_token_id=None,
        draft_length=2,
        retrieval_chunk_ids=None,
    )
    try:
        cloud_client.verify_specextend(
            SpecExtendRequest(
                request_id="warmup",
                prefix_ids=prompt_ids,
                draft_ids=draft.draft.input_ids,
                draft_position_ids=draft.draft.position_ids,
                parent_indices=draft.draft.parent_indices,
                draft_attention_mask=draft.draft.attention_mask,
            )
        )
    except Exception:
        logger.debug("Warmup verify failed", exc_info=True)
    logger.info("Warmup complete.")


def save_results(results: List[ExperimentResult], config: Dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = get_results_path()
    results_path.write_text(
        json.dumps({"config": config, "results": [asdict(result) for result in results]}, indent=2)
    )
    logger.info("Results saved to: %s", results_path)
    return results_path


def run_quick_test(config: Dict[str, Any]) -> Optional[Path]:
    logger.info("=" * 70)
    logger.info("QUICK TEST - direct vs SpecExtend benchmark profiles")
    logger.info("Draft model: %s on %s", DRAFT_MODEL_PATH, DRAFT_DEVICE)
    logger.info(
        "Draft mode: %s nodes=%s depth=%s",
        config["draft_mode"],
        config["draft_tree_nodes"],
        config["draft_tree_max_depth"],
    )
    logger.info("=" * 70)
    apply_draft_mode_config(config)

    prompts = load_prompt_set(config["prompt_types"], config["prompt_count"], config.get("dataset_split", "test"))
    base_client = create_http_cloud_client(SERVER_URL, timeout=REQUEST_TIMEOUT_SEC)

    draft_backend = QwenSpecExtendDraftBackend(
        QwenBackendConfig(
            model_path=DRAFT_MODEL_PATH,
            device=DRAFT_DEVICE,
            dtype=DRAFT_DTYPE,
            max_model_len=DRAFT_MAX_LEN,
            attn_implementation=DRAFT_ATTN_IMPLEMENTATION,
        )
    )
    prompts = truncate_prompts_to_tokens(prompts, draft_backend.tokenizer, config["prompt_input_tokens"])

    warmup(draft_backend, base_client, prompts[0][0]["text"])

    condition = NetworkCondition.good()
    throttled = ThrottledCloudClient(base_client, condition)
    profiles = selected_spec_profiles(config)

    results: List[ExperimentResult] = []
    for prompt_data, prompt_type in prompts:
        prompt_meta = {"id": prompt_data["id"], "type": prompt_type}
        prompt = prompt_data["text"]
        logger.info("  [%s | prompt %d]", prompt_type, prompt_data["id"])

        if "direct" in set(config.get("methods") or ["direct"]):
            direct_result = run_direct_case(prompt_meta, prompt, throttled, config["max_tokens"])
            if direct_result:
                results.append(direct_result)

        for profile in profiles:
            effective_profile = {**config, **profile}
            apply_draft_mode_config(config, profile)
            logger.info(
                "    profile=%s draft_mode=%s nodes=%s depth=%s",
                profile["name"],
                effective_profile["draft_mode"],
                effective_profile["draft_tree_nodes"],
                effective_profile["draft_tree_max_depth"],
            )
            draft_backend.cache.reset()
            specextend_client = build_specextend_client(
                draft_backend,
                throttled.verify_specextend,
                config,
                profile,
            )
            spec_result = run_specextend_case(
                specextend_client,
                throttled,
                prompt_meta,
                prompt,
                int(profile.get("draft_length", config["draft_length"])),
                effective_profile,
            )
            if spec_result:
                results.append(spec_result)

    if not results:
        logger.error("No successful benchmark results were produced.")
        return None

    return save_results(results, config)


def parse_args():
    parser = argparse.ArgumentParser(description="Run direct and real SpecExtend quick tests.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--prompt-count", type=int, default=None)
    parser.add_argument("--prompt-types", type=str, nargs="+", default=None)
    parser.add_argument("--dataset-split", type=str, default=None, help="Dataset file to use for pg19 (e.g. pg19_512, pg19_1K, pg19_2K, pg19_4K, pg19_8K, pg19_16K).")
    parser.add_argument("--prompt-input-tokens", type=int, default=None, help="Truncate prompts to this many input tokens.")
    parser.add_argument("--draft-mode", choices=["linear", "branching"], default=None)
    parser.add_argument("--draft-length", type=int, default=None, help="Draft tokens proposed per linear round.")
    parser.add_argument("--draft-tree-nodes", type=int, default=None)
    parser.add_argument("--draft-tree-max-depth", type=int, default=None)
    parser.add_argument("--retrieval-chunk-size", type=int, default=None)
    parser.add_argument("--retrieve-top-k", type=int, default=None)
    parser.add_argument("--retrieve-every-n-steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = apply_cli_overrides(load_config_file(args.config), args)
    results_path = run_quick_test(config)
    if results_path is None:
        sys.exit(1)
    print(f"RESULTS_PATH={results_path}")
    sys.exit(0)
