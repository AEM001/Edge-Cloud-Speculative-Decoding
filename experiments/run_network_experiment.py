#!/usr/bin/env python3
"""
Network-constrained speculative decoding experiment.

Compares the following generation methods under multiple simulated network
conditions (latency + bandwidth throttle):

  • direct          — single call to /generate on the verify server
  • spec_k3/5/7     — synchronous speculative decoding with K ∈ {3, 5, 7}
  • async_k3/5/7    — async (pipelined) speculative decoding with K ∈ {3, 5, 7}

Re-usable modules used:
  experiments/network_conditions.py  — NetworkCondition, ThrottledCloudClient
  experiments/metrics_collector.py   — MetricsCollector, ExperimentResult

Quick start (verify server must be running on localhost:6006):
    python -m experiments.run_network_experiment --prompts 2 --max-tokens 128
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

# Ensure project root is on sys.path when run as __main__
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from client.edge_client import EdgeClient
from client.http_cloud_client import create_http_cloud_client
from config import (
    DRAFT_GPU_MEM as GPU_MEM,
    DRAFT_MAX_LEN as MAX_LEN,
    DRAFT_MODEL_PATH as MODEL_PATH,
    VERIFY_SERVER_URL,
)
from draft_generator import VLLMDraftGenerator
from experiments.metrics_collector import (
    MetricsCollector,
    from_direct,
    from_sync_metrics,
)
from experiments.network_conditions import NetworkCondition, ThrottledCloudClient
from model_manager import VLLMModelManager
from prompt_loader import load_prompts_by_type

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SERVER_URL = VERIFY_SERVER_URL
DEFAULT_MAX_TOKENS = 128
DEFAULT_PROMPTS_PER_TYPE = 2
DEFAULT_K_VALUES = [3, 5, 7]
DEFAULT_TEMPERATURE = 0.0
DEFAULT_GPU_ID = 1                  # edge/draft GPU

# Network profiles to sweep (ordered from fastest to slowest)
DEFAULT_NETWORK_PROFILES = [
    NetworkCondition.good(),
    NetworkCondition.medium(),
    NetworkCondition.bursty(),
]


# ---------------------------------------------------------------------------
# Direct generation helper
# ---------------------------------------------------------------------------

def run_direct(
    prompt: str,
    server_url: str,
    max_tokens: int,
    temperature: float,
    throttled_client: Optional[ThrottledCloudClient] = None,
    timeout: float = 120.0,
) -> Tuple[int, float]:
    """
    Call ``/generate`` on the verify server directly.

    When a `throttled_client` is provided its network stats are read for the
    simulated overhead.  The HTTP call itself still goes straight to the server
    (the throttle wrapper is for EdgeRequest/CloudResponse flows only); for
    direct calls we manually apply the equivalent sleep to simulate the link.
    """
    start = time.perf_counter()

    # Simulate uplink (prompt → server)
    if throttled_client is not None:
        cond = throttled_client.condition
        now = time.perf_counter() - throttled_client._start_wall
        one_way_ms, dl_mbps, ul_mbps = cond.current_link_params(now)
        uplink_bytes = len(prompt.encode("utf-8")) + 64
        uplink_delay = one_way_ms + NetworkCondition._payload_delay_ms(uplink_bytes, ul_mbps)
        time.sleep(uplink_delay / 1000.0)

    try:
        resp = requests.post(
            f"{server_url}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        tokens = data.get("tokens_generated", 0)
        text = data.get("text", "")
    except Exception as exc:
        logger.error("Direct generation failed: %s", exc)
        elapsed = (time.perf_counter() - start) * 1000
        return 0, elapsed

    # Simulate downlink (response → edge)
    if throttled_client is not None:
        now2 = time.perf_counter() - throttled_client._start_wall
        one_way_ms2, dl_mbps2, _ = cond.current_link_params(now2)
        downlink_bytes = len((text or "").encode("utf-8")) + 64
        downlink_delay = one_way_ms2 + NetworkCondition._payload_delay_ms(downlink_bytes, dl_mbps2)
        time.sleep(downlink_delay / 1000.0)

    elapsed = (time.perf_counter() - start) * 1000
    return tokens, elapsed


# ---------------------------------------------------------------------------
# Per-condition experiment runner
# ---------------------------------------------------------------------------

def run_condition(
    condition: NetworkCondition,
    base_client,
    edge_client: EdgeClient,
    prompts: List[dict],
    k_values: List[int],
    max_tokens: int,
    temperature: float,
    server_url: str,
    collector: MetricsCollector,
) -> None:
    """
    Run all methods (direct, spec_k*) for every prompt under one network
    condition.  Results are appended directly to `collector`.
    """
    logger.info("")
    logger.info("════════════════════════════════════════════")
    logger.info("  Network condition: %s", condition)
    logger.info("════════════════════════════════════════════")

    throttled = ThrottledCloudClient(base_client, condition)
    edge_client.cloud_client = throttled

    for prompt_data in prompts:
        pid = prompt_data["id"]
        ptype = prompt_data["type"]
        text = prompt_data["text"]

        logger.info("")
        logger.info("── Prompt %d (%s): %s…", pid, ptype, text[:60])

        # ── 1. Direct ──────────────────────────────────────────────────
        logger.info("   [direct]")
        throttled.reset_stats()
        tokens, latency_ms = run_direct(
            prompt=text,
            server_url=server_url,
            max_tokens=max_tokens,
            temperature=temperature,
            throttled_client=throttled,
        )
        net_stats = throttled.get_stats_dict()
        collector.add(
            from_direct(
                tokens=tokens,
                latency_ms=latency_ms,
                network_condition=condition.name,
                prompt_id=pid,
                prompt_type=ptype,
                simulated_overhead_ms=net_stats["total_simulated_overhead_ms"],
                simulated_uplink_delay_ms=net_stats["total_simulated_uplink_delay_ms"],
                simulated_downlink_delay_ms=net_stats["total_simulated_downlink_delay_ms"],
                error=(tokens == 0),
            )
        )
        time.sleep(0.3)

        # ── 2. Synchronous speculative (K = 3, 5, 7) ───────────────────
        for k in k_values:
            logger.info("   [spec_k%d]", k)
            throttled.reset_stats()
            try:
                metrics = edge_client.generate(
                    prompt=text,
                    policy=lambda _rid, _toks, _k=k: _k,
                    policy_name=f"StaticK{k}",
                )
                net_stats = throttled.get_stats_dict()
                collector.add(
                    from_sync_metrics(
                        raw=metrics,
                        k=k,
                        network_condition=condition.name,
                        prompt_id=pid,
                        prompt_type=ptype,
                        simulated_overhead_ms=net_stats["total_simulated_overhead_ms"],
                        simulated_uplink_delay_ms=net_stats["total_simulated_uplink_delay_ms"],
                        simulated_downlink_delay_ms=net_stats["total_simulated_downlink_delay_ms"],
                    )
                )
            except Exception as exc:
                logger.error("   spec_k%d failed: %s", k, exc)
                net_stats = throttled.get_stats_dict()
                collector.add(
                    from_sync_metrics(
                        raw=_zero_sync_metrics(text),
                        k=k,
                        network_condition=condition.name,
                        prompt_id=pid,
                        prompt_type=ptype,
                        simulated_overhead_ms=net_stats["total_simulated_overhead_ms"],
                        simulated_uplink_delay_ms=net_stats["total_simulated_uplink_delay_ms"],
                        simulated_downlink_delay_ms=net_stats["total_simulated_downlink_delay_ms"],
                        error=True,
                        error_message=str(exc),
                    )
                )
            time.sleep(0.3)


# ---------------------------------------------------------------------------
# Zero-value stand-ins for error paths
# ---------------------------------------------------------------------------

def _zero_sync_metrics(prompt: str):
    """Return a zeroed-out RequestMetrics-compatible object."""
    from client.edge_client import RequestMetrics
    m = RequestMetrics(request_id="err", prompt=prompt)
    return m


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    logger.info("PicoSpec Network Experiment")
    logger.info("  server         : %s", args.server)
    logger.info("  max_tokens     : %d", args.max_tokens)
    logger.info("  prompts/type   : %d", args.prompts)
    logger.info("  K values       : %s", args.k_values)
    logger.info("  profiles       : %s", [p.name for p in args.profiles])

    # ── Verify server health ───────────────────────────────────────────
    try:
        r = requests.get(f"{args.server}/health", timeout=10)
        r.raise_for_status()
        logger.info("Verify server healthy: %s", r.json())
    except Exception as exc:
        logger.error("Cannot reach verify server at %s: %s", args.server, exc)
        sys.exit(1)

    # ── Load draft model ───────────────────────────────────────────────
    logger.info("Loading draft model from %s …", MODEL_PATH)
    model_manager = VLLMModelManager(
        model_path=MODEL_PATH,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_LEN,
        gpu_id=DEFAULT_GPU_ID,
    )
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    logger.info("Draft model loaded.")

    # ── Build base (un-throttled) cloud client ────────────────────────
    base_client = create_http_cloud_client(args.server, timeout=120.0)

    # ── Build EdgeClient ──────────────────────────────────────────────
    edge_client = EdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=base_client,      # will be overridden per condition
        max_new_tokens=args.max_tokens,
        temperature=DEFAULT_TEMPERATURE,
    )

    # ── Load prompts ──────────────────────────────────────────────────
    simple_prompts, complex_prompts = load_prompts_by_type(
        count_per_type=args.prompts
    )
    prompts = simple_prompts + complex_prompts
    logger.info("Loaded %d prompts (%d simple + %d complex)",
                len(prompts), len(simple_prompts), len(complex_prompts))

    # ── Run experiment ─────────────────────────────────────────────────
    collector = MetricsCollector()
    experiment_start = time.time()

    for condition in args.profiles:
        run_condition(
            condition=condition,
            base_client=base_client,
            edge_client=edge_client,
            prompts=prompts,
            k_values=args.k_values,
            max_tokens=args.max_tokens,
            temperature=DEFAULT_TEMPERATURE,
            server_url=args.server,
            collector=collector,
        )

    elapsed = time.time() - experiment_start
    logger.info("\nExperiment complete in %.1f s", elapsed)

    # ── Save results ──────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "outputs_network"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    collector.save_json(out_dir / f"results_{ts}.json")
    collector.save_csv(out_dir / f"results_{ts}.csv")
    collector.save_summary_json(out_dir / f"summary_{ts}.json")

    # Symlink / overwrite the "latest" files for convenience
    for suffix in ("json", "csv"):
        latest = out_dir / f"results_latest.{suffix}"
        src = out_dir / f"results_{ts}.{suffix}"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(src.name)

    # ── Print report ──────────────────────────────────────────────────
    print("\n")
    collector.print_table()
    print("\n")
    collector.print_metric_spotlight()

    logger.info("\nOutput directory: %s", out_dir.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Network-constrained speculative decoding experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--server", default=DEFAULT_SERVER_URL,
        help="Base URL of the verify server",
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Maximum new tokens per generation",
    )
    p.add_argument(
        "--prompts", type=int, default=DEFAULT_PROMPTS_PER_TYPE,
        help="Number of prompts per type (simple + complex)",
    )
    p.add_argument(
        "--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
        help="Draft lengths K to test",
    )
    p.add_argument(
        "--profiles", nargs="+", default=None,
        choices=["good", "medium", "bursty"],
        help="Network profiles to include (default: all)",
    )
    p.add_argument(
        "--no-direct", action="store_true",
        help="Skip direct generation runs",
    )
    args = p.parse_args()

    # Resolve profiles
    all_profiles = {p.name: p for p in DEFAULT_NETWORK_PROFILES}
    if args.profiles:
        args.profiles = [all_profiles[n] for n in args.profiles if n in all_profiles]
    else:
        args.profiles = DEFAULT_NETWORK_PROFILES

    return args


if __name__ == "__main__":
    main(_parse_args())
