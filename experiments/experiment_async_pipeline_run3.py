"""
Experiment Run 3 (revised): Synchronous vs Asynchronous Speculative Decoding Pipeline
=======================================================================================

Key fixes vs initial run:
  - enforce_eager removed from draft model (CUDA Graph enabled)
  - TokensPrompt used to avoid decode→re-encode every round
  - prompt_logprobs removed (was recomputing entire prefix each step)
  - Network latency simulation added: 20 / 50 / 100 ms one-way

Methods benchmarked (per latency tier):
  1. Direct            — autoregressive baseline (verify server only)
  2. Sync K=2/4/8      — stop-and-wait speculative decoding
  3. Async K=2/4/8     — PicoSpec Parallel Drafting (lookahead=2)

Network tiers:
  - LAN:  simulated 20 ms one-way (40 ms RTT)
  - WAN:  simulated 50 ms one-way (100 ms RTT)
  - Edge: simulated 100 ms one-way (200 ms RTT)"""

import json
import logging
import re
import sys
import time
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from client.edge_client import EdgeClient, RequestMetrics
from client.async_edge_client import AsyncEdgeClient, AsyncRequestMetrics
from client.http_cloud_client import create_http_cloud_client
from config_local import GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, MODEL_NAME, MODEL_PATH
from draft_generator import VLLMDraftGenerator
from model_manager import VLLMModelManager

VERIFY_MODEL = "Qwen2.5-7B-Instruct-AWQ"

# ─── Output dir (created before logging setup) ────────────────────────────────
_RUN_DATE = time.strftime("%Y-%m-%d")
OUT_DIR = Path(__file__).parent / f"outputs_run3_{_RUN_DATE}"
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUT_DIR / "async_log.txt"),
    ],
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
BENCHMARK_TEMPERATURE = 0.0
PROMPT_COUNT = 5           # 5 easy + 5 hard  (faster iteration)
MAX_TOKENS = 128
SERVER_URL = "http://localhost:6006"
K_VALUES = [4, 8, 12]
LOOKAHEAD = 2
ASYNC_TIMEOUT_S = 60   # skip async test if it takes longer than this

# Simulated one-way network latency tiers (ms)
# Each tier is tested independently; Direct always uses 0 ms
NET_LATENCY_TIERS = {
    "LAN":  20,    # ~40 ms RTT — intra-datacenter or fast local network
    "WAN":  50,    # ~100 ms RTT — inter-region WAN
    "Edge": 100,   # ~200 ms RTT — true edge (4G/5G uplink)
}


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class TestResult:
    method: str
    prompt_type: str
    prompt_id: int
    prompt_length: int
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float

    num_rounds: int = 0
    acceptance_rate: float = 0.0
    mean_k: float = 0.0

    avg_draft_ms: float = 0.0
    avg_verify_ms: float = 0.0
    avg_rtt_ms: float = 0.0
    avg_network_ms: float = 0.0

    pipeline_efficiency: float = 0.0
    avg_bubble_ms: float = 0.0
    prefetch_waste_ratio: float = 0.0
    async_speedup_vs_sync: float = 0.0
    rollback_rate: float = 0.0
    effective_tokens_per_round: float = 0.0
    overlap_utilisation: float = 0.0
    net_tier: str = "LAN"


# ─── Prompt loading ───────────────────────────────────────────────────────────

def load_prompts(filepath: Path, count: int = PROMPT_COUNT) -> Tuple[List[Dict], List[Dict]]:
    easy, hard = [], []
    with open(filepath) as f:
        content = f.read()

    easy_sec = re.search(r"EASY BENCHMARKS.*?HARD BENCHMARKS", content, re.DOTALL)
    if easy_sec:
        for idx, (_, length, text) in enumerate(
            re.findall(
                r"\[(\d+)\]\s+benchmark_[\d\-]+.*?Actual length:\s*(\d+)\s*chars.*?User:\s*(.*?)(?=\n\n\[|$)",
                easy_sec.group(),
                re.DOTALL,
            )[: count * 2],
            1,
        ):
            easy.append({"id": idx, "length": int(length), "text": text.strip()[:400]})

    hard_sec = re.search(r"HARD BENCHMARKS.*?(?=$)", content, re.DOTALL)
    if hard_sec:
        for idx, (_, length, text) in enumerate(
            re.findall(
                r"\[(\d+)\]\s+benchmark_[\d\-]+.*?Actual length:\s*(\d+)\s*chars.*?User:\s*(.*?)(?=\n\n\[|$)",
                hard_sec.group(),
                re.DOTALL,
            )[: count * 2],
            1,
        ):
            hard.append({"id": idx, "length": int(length), "text": text.strip()[:400]})

    def pick(lst, n, target):
        lst = sorted(lst, key=lambda x: abs(x["length"] - target))[:n]
        for i, p in enumerate(sorted(lst, key=lambda x: x["id"]), 1):
            p["id"] = i
        return lst

    easy_sel = pick(easy, count, 300)
    hard_sel = pick(hard, count, 500)
    logger.info("Prompts loaded: %d easy, %d hard", len(easy_sel), len(hard_sel))
    return easy_sel, hard_sel


PROMPTS_FILE = Path(__file__).parent.parent / "benchmarks" / "prompts.txt"


# ─── Network latency simulation ───────────────────────────────────────────────

class LatencyCloudClient:
    """Wraps a cloud client and injects simulated one-way network latency."""
    def __init__(self, inner_client, one_way_ms: float):
        self._inner = inner_client
        self._delay_s = one_way_ms / 1000.0

    def __call__(self, request):
        time.sleep(self._delay_s)       # simulated uplink
        response = self._inner(request)
        time.sleep(self._delay_s)       # simulated downlink
        return response


# ─── Test helpers ─────────────────────────────────────────────────────────────

def test_direct(server_url: str, prompt: str) -> Tuple[int, float]:
    start = time.time()
    session = requests.Session()
    session.trust_env = False
    try:
        r = session.post(
            f"{server_url}/generate",
            json={"prompt": prompt, "max_tokens": MAX_TOKENS, "temperature": BENCHMARK_TEMPERATURE},
            timeout=120.0,
        )
        r.raise_for_status()
        result = r.json()
        return result["tokens_generated"], (time.time() - start) * 1000
    except Exception as e:
        logger.error("Direct failed: %s", e)
        return 0, 0.0


def result_from_sync_metrics(
    m: RequestMetrics, method: str, prompt_type: str, pid: int, plen: int,
    net_tier: str = "LAN",
) -> TestResult:
    rounds = m.total_rounds or 1
    return TestResult(
        method=method,
        prompt_type=prompt_type,
        prompt_id=pid,
        prompt_length=plen,
        tokens_generated=m.generated_tokens,
        total_time_ms=m.total_latency_ms,
        tokens_per_second=1000 * m.generated_tokens / m.total_latency_ms if m.total_latency_ms > 0 else 0,
        num_rounds=m.total_rounds,
        acceptance_rate=m.acceptance_ratio,
        mean_k=m.mean_K_chosen,
        avg_draft_ms=m.total_edge_draft_time_ms / rounds,
        avg_verify_ms=m.total_server_verify_time_ms / rounds,
        avg_rtt_ms=m.average_rtt_ms,
        avg_network_ms=m.total_network_time_ms / rounds,
        net_tier=net_tier,
    )


def result_from_async_metrics(
    m: AsyncRequestMetrics, method: str, prompt_type: str, pid: int, plen: int,
    net_tier: str = "LAN",
) -> TestResult:
    rounds = m.total_rounds or 1
    total_draft = m.total_edge_draft_time_ms
    total_verify = m.total_server_verify_time_ms
    overlap_util = total_verify / (total_draft + total_verify) if (total_draft + total_verify) > 0 else 0
    return TestResult(
        method=method,
        prompt_type=prompt_type,
        prompt_id=pid,
        prompt_length=plen,
        tokens_generated=m.generated_tokens,
        total_time_ms=m.total_latency_ms,
        tokens_per_second=m.tokens_per_second,
        num_rounds=m.total_rounds,
        acceptance_rate=m.acceptance_ratio,
        mean_k=m.mean_k_chosen,
        avg_draft_ms=total_draft / rounds,
        avg_verify_ms=total_verify / rounds,
        avg_rtt_ms=m.average_rtt_ms,
        avg_network_ms=m.total_network_time_ms / rounds,
        pipeline_efficiency=m.pipeline_efficiency,
        avg_bubble_ms=m.avg_bubble_ms,
        prefetch_waste_ratio=m.prefetch_waste_ratio,
        async_speedup_vs_sync=m.async_speedup_vs_sync,
        rollback_rate=m.total_rollbacks / rounds,
        effective_tokens_per_round=m.total_accepted_tokens / rounds,
        overlap_utilisation=overlap_util,
        net_tier=net_tier,
    )


# ─── Plotting ─────────────────────────────────────────────────────────────────

def save_charts(results: List[TestResult], out_dir: Path) -> None:
    tiers      = list(NET_LATENCY_TIERS.keys())
    sync_meth  = [f"sync_k{k}"  for k in K_VALUES]
    async_meth = [f"async_k{k}" for k in K_VALUES]
    sync_lab   = [f"Sync K={k}"  for k in K_VALUES]
    async_lab  = [f"Async K={k}" for k in K_VALUES]
    all_meth   = ["direct"] + sync_meth + async_meth
    all_labels = ["Direct"] + sync_lab + async_lab

    tier_colors   = {"LAN": "#2196F3", "WAN": "#FF9800", "Edge": "#F44336"}
    palette_sync  = ["#4CAF50", "#9C27B0", "#795548"]
    palette_async = ["#00BCD4", "#E91E63", "#FF5722"]
    direct_color  = "#607D8B"

    def mf(field, method, ptype, tier):
        vals = [getattr(r, field) for r in results
                if r.method == method and r.prompt_type == ptype and r.net_tier == tier]
        return float(np.mean(vals)) if vals else 0.0

    # ── 1. Throughput by tier (3 rows × 2 cols) ──────────────────────────────
    fig, axes = plt.subplots(len(tiers), 2, figsize=(16, 5 * len(tiers)))
    for row, tier in enumerate(tiers):
        for col, ptype in enumerate(["easy", "hard"]):
            ax = axes[row][col]
            toks   = [mf("tokens_per_second", m, ptype, tier) for m in all_meth]
            colors = [direct_color] + palette_sync + palette_async
            bars   = ax.bar(all_labels, toks, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_title(f"Throughput [{tier}] — {ptype.capitalize()}")
            ax.set_ylabel("Tokens / second")
            ax.set_ylim(0, max(toks) * 1.3 if max(toks) > 0 else 1)
            for bar, val in zip(bars, toks):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=7)
            ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_by_tier.png", dpi=150)
    plt.close()

    # ── 2. Speedup vs Direct per tier (line chart) ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for col, ptype in enumerate(["easy", "hard"]):
        ax = axes[col]
        for tier, tc in tier_colors.items():
            d_tps = mf("tokens_per_second", "direct", ptype, tier) or 1.0
            sync_spd  = [mf("tokens_per_second", f"sync_k{k}",  ptype, tier) / d_tps for k in K_VALUES]
            async_spd = [mf("tokens_per_second", f"async_k{k}", ptype, tier) / d_tps for k in K_VALUES]
            ax.plot(K_VALUES, sync_spd,  "o--", color=tc, alpha=0.7, label=f"Sync {tier}")
            ax.plot(K_VALUES, async_spd, "^-",  color=tc, alpha=1.0, label=f"Async {tier}")
        ax.axhline(1.0, linestyle=":", color="black", linewidth=0.8, label="Direct")
        ax.set_title(f"Speedup vs Direct — {ptype.capitalize()}")
        ax.set_xlabel("K (draft tokens)")
        ax.set_ylabel("Speedup (tok/s / Direct tok/s)")
        ax.set_xticks(K_VALUES)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_vs_tier.png", dpi=150)
    plt.close()

    # ── 3. Latency breakdown per tier (easy prompts, sync+async) ─────────────
    spec_meth = sync_meth + async_meth
    spec_lab  = sync_lab  + async_lab
    fig, axes = plt.subplots(1, len(tiers), figsize=(6 * len(tiers), 5))
    for ax, tier in zip(axes, tiers):
        d_ms = [mf("avg_draft_ms",   m, "easy", tier) for m in spec_meth]
        n_ms = [mf("avg_network_ms", m, "easy", tier) for m in spec_meth]
        v_ms = [mf("avg_verify_ms",  m, "easy", tier) for m in spec_meth]
        x = np.arange(len(spec_lab))
        ax.bar(x, d_ms, label="Draft",          color="#4CAF50")
        ax.bar(x, n_ms, bottom=d_ms,            label="Network (sim)", color="#FF9800")
        ax.bar(x, v_ms, bottom=[d+n for d,n in zip(d_ms,n_ms)], label="Verify", color="#F44336")
        ax.set_title(f"Latency/Round — Easy [{tier}]")
        ax.set_ylabel("ms per round")
        ax.set_xticks(x)
        ax.set_xticklabels(spec_lab, rotation=25, fontsize=8)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "latency_breakdown.png", dpi=150)
    plt.close()

    # ── 4. Pipeline efficiency by tier (async only) ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.25
    x = np.arange(len(K_VALUES))
    for col, ptype in enumerate(["easy", "hard"]):
        ax = axes[col]
        for i, (tier, tc) in enumerate(tier_colors.items()):
            effs = [mf("pipeline_efficiency", f"async_k{k}", ptype, tier) * 100 for k in K_VALUES]
            ax.bar(x + (i - 1) * width, effs, width, label=tier, color=tc, alpha=0.85)
        ax.set_title(f"Async Pipeline Efficiency — {ptype.capitalize()}")
        ax.set_ylabel("Full-Hit Rate (%)")
        ax.set_ylim(0, 110)
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in K_VALUES])
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pipeline_efficiency.png", dpi=150)
    plt.close()

    # ── 5. Bubble time by tier (async only) ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for col, ptype in enumerate(["easy", "hard"]):
        ax = axes[col]
        for i, (tier, tc) in enumerate(tier_colors.items()):
            bubbles = [mf("avg_bubble_ms", f"async_k{k}", ptype, tier) for k in K_VALUES]
            ax.bar(x + (i - 1) * width, bubbles, width, label=f"Bubble {tier}",
                   color=tc, alpha=0.85)
        drafts = [mf("avg_draft_ms", f"async_k{k}", "easy", "LAN") for k in K_VALUES]
        ax.plot(x, drafts, "k--o", linewidth=1.2, markersize=4, label="Draft ms (LAN)")
        ax.set_title(f"Pipeline Bubble vs Draft — {ptype.capitalize()}")
        ax.set_ylabel("ms per round")
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in K_VALUES])
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "bubble_time.png", dpi=150)
    plt.close()

    logger.info("Charts saved to %s", out_dir)


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_experiment():
    easy_prompts, hard_prompts = load_prompts(PROMPTS_FILE, count=PROMPT_COUNT)

    logger.info("=" * 80)
    logger.info("ASYNC PIPELINE EXPERIMENT — RUN 3 (revised)")
    logger.info("Draft model  : %s (%s)", MODEL_NAME, MODEL_PATH)
    logger.info("Verify model : %s @ %s", VERIFY_MODEL, SERVER_URL)
    logger.info("K values: %s  Lookahead: %d  Tiers: %s", K_VALUES, LOOKAHEAD, list(NET_LATENCY_TIERS))
    logger.info("Prompts: %d easy + %d hard", len(easy_prompts), len(hard_prompts))
    logger.info("Output: %s", OUT_DIR)
    logger.info("=" * 80)

    logger.info("[Setup] Loading draft model (CUDA Graph enabled)...")
    model_manager = VLLMModelManager(MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    base_cloud_client = create_http_cloud_client(SERVER_URL, timeout=300.0)
    logger.info("[Setup] Ready")

    results: List[TestResult] = []
    prompt_sets = [("easy", easy_prompts), ("hard", hard_prompts)]

    # ── Step 1: Direct baseline (no latency sim) ───────────────────────────────
    logger.info("=" * 60)
    logger.info("DIRECT BASELINE")
    logger.info("=" * 60)
    for ptype, plist in prompt_sets:
        for pd in plist:
            pid, ptext, plen = pd["id"], pd["text"], pd["length"]
            logger.info("  Direct %s Prompt %d (len=%d)...", ptype.upper(), pid, plen)
            toks, ms = test_direct(SERVER_URL, ptext)
            if toks > 0:
                tps = toks / (ms / 1000)
                for tier in NET_LATENCY_TIERS:
                    results.append(TestResult(
                        method="direct", prompt_type=ptype, prompt_id=pid,
                        prompt_length=plen, tokens_generated=toks,
                        total_time_ms=ms, tokens_per_second=tps,
                        net_tier=tier,
                    ))
                logger.info("    → %d tok  %.0fms  %.2f tok/s", toks, ms, tps)
            time.sleep(0.3)

    # ── Step 2: Speculative decoding per latency tier ──────────────────────────
    for tier_name, one_way_ms in NET_LATENCY_TIERS.items():
        rtt_ms = one_way_ms * 2
        logger.info("=" * 80)
        logger.info("TIER: %s  (one-way: %d ms → RTT: %d ms)", tier_name, one_way_ms, rtt_ms)
        logger.info("=" * 80)

        latency_client = LatencyCloudClient(base_cloud_client, one_way_ms)

        sync_client = EdgeClient(
            model_manager=model_manager,
            draft_generator=draft_generator,
            cloud_client=latency_client,
            max_new_tokens=MAX_TOKENS,
            temperature=BENCHMARK_TEMPERATURE,
        )
        async_client = AsyncEdgeClient(
            model_manager=model_manager,
            draft_generator=draft_generator,
            cloud_client=latency_client,
            max_new_tokens=MAX_TOKENS,
            temperature=BENCHMARK_TEMPERATURE,
            lookahead=LOOKAHEAD,
        )

        for ptype, plist in prompt_sets:
            for pd in plist:
                pid, ptext, plen = pd["id"], pd["text"], pd["length"]
                logger.info("  [%s] %s Prompt %d (len=%d)", tier_name, ptype.upper(), pid, plen)

                for k in K_VALUES:
                    policy = lambda _rid, _dt, _k=k: _k

                    # Sync K=k
                    logger.info("    Sync K=%d [%s]...", k, tier_name)
                    try:
                        sm: RequestMetrics = sync_client.generate(
                            prompt=ptext, policy=policy, policy_name=f"SyncK{k}")
                        if sm.generated_tokens > 0:
                            r = result_from_sync_metrics(sm, f"sync_k{k}", ptype, pid, plen,
                                                         net_tier=tier_name)
                            results.append(r)
                            logger.info("      → %d tok  %.0fms  %.2f tok/s | acc=%.1f%%  rtt=%.0fms",
                                        r.tokens_generated, r.total_time_ms, r.tokens_per_second,
                                        r.acceptance_rate * 100, r.avg_rtt_ms)
                    except Exception as e:
                        logger.error("Sync K=%d [%s] failed: %s", k, tier_name, e)
                    time.sleep(0.2)

                    # Async K=k
                    logger.info("    Async K=%d [%s] (lookahead=%d)...", k, tier_name, LOOKAHEAD)
                    _async_result: list = []
                    _async_exc:    list = []
                    def _run_async(res=_async_result, exc=_async_exc):
                        try:
                            res.append(async_client.generate(
                                prompt=ptext, policy=policy, policy_name=f"AsyncK{k}"))
                        except Exception as _e:
                            exc.append(_e)
                    t = threading.Thread(target=_run_async, daemon=True)
                    t.start()
                    t.join(timeout=ASYNC_TIMEOUT_S)
                    if t.is_alive():
                        logger.warning("    Async K=%d [%s] TIMED OUT after %ds — skipping",
                                       k, tier_name, ASYNC_TIMEOUT_S)
                        # Rebuild AsyncEdgeClient to discard poisoned internal state
                        async_client = AsyncEdgeClient(
                            model_manager=model_manager,
                            draft_generator=draft_generator,
                            cloud_client=latency_client,
                            max_new_tokens=MAX_TOKENS,
                            temperature=BENCHMARK_TEMPERATURE,
                            lookahead=LOOKAHEAD,
                        )
                    elif _async_exc:
                        logger.error("Async K=%d [%s] failed: %s", k, tier_name, _async_exc[0])
                    elif _async_result:
                        am: AsyncRequestMetrics = _async_result[0]
                        if am.generated_tokens > 0:
                            r = result_from_async_metrics(am, f"async_k{k}", ptype, pid, plen,
                                                          net_tier=tier_name)
                            results.append(r)
                            logger.info(
                                "      → %d tok  %.0fms  %.2f tok/s | acc=%.1f%%  "
                                "pipe_eff=%.1f%%  bubble=%.1fms  waste=%.1f%%",
                                r.tokens_generated, r.total_time_ms, r.tokens_per_second,
                                r.acceptance_rate * 100, r.pipeline_efficiency * 100,
                                r.avg_bubble_ms, r.prefetch_waste_ratio * 100)
                    time.sleep(0.2)

    # ── Summary table ──────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("SUMMARY BY TIER")
    logger.info("=" * 80)
    all_methods = ["direct"] + [f"sync_k{k}" for k in K_VALUES] + [f"async_k{k}" for k in K_VALUES]
    summary = {}

    for tier in list(NET_LATENCY_TIERS.keys()):
        for ptype in ["easy", "hard"]:
            direct_tps = float(np.mean([r.tokens_per_second for r in results
                                         if r.method == "direct" and r.prompt_type == ptype
                                         and r.net_tier == tier]) or 1.0)
            for method in all_methods:
                entries = [r for r in results
                           if r.method == method and r.prompt_type == ptype and r.net_tier == tier]
                if not entries:
                    continue
                key = f"{tier}_{ptype}_{method}"
                tps_vals = [r.tokens_per_second for r in entries]
                summary[key] = {
                    "tier": tier,
                    "prompt_type": ptype,
                    "method": method,
                    "mean_tps": float(np.mean(tps_vals)),
                    "median_tps": float(np.median(tps_vals)),
                    "std_tps": float(np.std(tps_vals)),
                    "speedup_vs_direct": float(np.mean(tps_vals)) / direct_tps,
                    "n": len(entries),
                }
                if method != "direct":
                    summary[key].update({
                        "mean_acceptance": float(np.mean([r.acceptance_rate for r in entries])),
                        "mean_rtt_ms": float(np.mean([r.avg_rtt_ms for r in entries])),
                        "mean_rounds": float(np.mean([r.num_rounds for r in entries])),
                    })
                if method.startswith("async"):
                    summary[key].update({
                        "mean_pipeline_efficiency": float(np.mean([r.pipeline_efficiency for r in entries])),
                        "mean_bubble_ms": float(np.mean([r.avg_bubble_ms for r in entries])),
                        "mean_prefetch_waste": float(np.mean([r.prefetch_waste_ratio for r in entries])),
                        "mean_async_speedup_vs_sync": float(np.mean([r.async_speedup_vs_sync for r in entries])),
                        "mean_rollback_rate": float(np.mean([r.rollback_rate for r in entries])),
                        "mean_overlap_utilisation": float(np.mean([r.overlap_utilisation for r in entries])),
                    })

        logger.info("--- %s ---", tier)
        d_e = float(np.mean([r.tokens_per_second for r in results
                              if r.method == "direct" and r.prompt_type == "easy" and r.net_tier == tier]) or 1)
        d_h = float(np.mean([r.tokens_per_second for r in results
                              if r.method == "direct" and r.prompt_type == "hard" and r.net_tier == tier]) or 1)
        for method in all_methods:
            e_tps = summary.get(f"{tier}_easy_{method}", {}).get("mean_tps", 0)
            h_tps = summary.get(f"{tier}_hard_{method}", {}).get("mean_tps", 0)
            logger.info("  %-14s  Easy: %6.1f tok/s (%4.2fx)  Hard: %6.1f tok/s (%4.2fx)",
                        method, e_tps, e_tps / d_e, h_tps, h_tps / d_h)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_file = OUT_DIR / "async_results.json"
    with open(out_file, "w") as f:
        json.dump({"results": [asdict(r) for r in results], "summary": summary}, f, indent=2)
    logger.info("Results saved: %s", out_file)

    # ── Save charts ────────────────────────────────────────────────────────────
    try:
        save_charts(results, OUT_DIR)
    except Exception as e:
        logger.error("Chart generation failed: %s", e)

    return results, summary


if __name__ == "__main__":
    run_experiment()
