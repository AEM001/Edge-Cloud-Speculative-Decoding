"""
Experiment: Synchronous vs Asynchronous Speculative Decoding Pipeline
======================================================================

Mirrors the structure of comprehensive_benchmark_run4.py but compares:
  1. Direct          — autoregressive baseline (cloud LLM only)
  2. Sync K=2/4      — standard stop-and-wait speculative decoding
  3. Async K=2/4     — PicoSpec Parallel Drafting (AsyncEdgeClient)

New pipeline-specific metrics (not in run4):
  - pipeline_efficiency      : fraction of rounds with full acceptance
  - avg_bubble_ms            : avg time drafter blocked waiting for verifier
  - prefetch_waste_ratio     : fraction of pre-drafted tokens discarded
  - async_speedup_vs_sync    : theoretical speedup max(T_d,T_v)/(T_d+T_v)
  - overlap_utilisation      : verify_time / (draft_time + verify_time)
  - rollback_rate            : rollbacks per round
  - effective_tokens_per_round: accepted tokens / total rounds

Output: experiments/outputs_async/
  - async_results.json
  - async_log.txt
  - pipeline_efficiency.png
  - bubble_time.png
  - throughput_comparison.png
  - latency_breakdown.png
  - acceptance_vs_pipeline.png
"""

import json
import logging
import re
import sys
import time
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "outputs_async" / "async_log.txt"),
    ],
)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
BENCHMARK_TEMPERATURE = 0.0
PROMPT_COUNT = 10          # easy + hard
MAX_TOKENS = 128
SERVER_URL = "http://localhost:6006"
K_VALUES = [2, 4]
LOOKAHEAD = 1              # PicoSpec baseline: 1 pre-draft batch in flight


# ─── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class TestResult:
    method: str              # "direct" | "sync_k2" | "sync_k4" | "async_k2" | "async_k4"
    prompt_type: str         # "easy" | "hard"
    prompt_id: int
    prompt_length: int
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float

    # Round / acceptance
    num_rounds: int = 0
    acceptance_rate: float = 0.0
    mean_k: float = 0.0

    # Timing breakdown
    avg_draft_ms: float = 0.0
    avg_verify_ms: float = 0.0
    avg_rtt_ms: float = 0.0
    avg_network_ms: float = 0.0

    # Async-specific (0 for sync/direct)
    pipeline_efficiency: float = 0.0
    avg_bubble_ms: float = 0.0
    prefetch_waste_ratio: float = 0.0
    async_speedup_vs_sync: float = 0.0
    rollback_rate: float = 0.0
    effective_tokens_per_round: float = 0.0
    overlap_utilisation: float = 0.0


# ─── Prompt loading (same logic as run4) ─────────────────────────────────────

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
    m: RequestMetrics, method: str, prompt_type: str, pid: int, plen: int
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
        # Async fields left at 0
    )


def result_from_async_metrics(
    m: AsyncRequestMetrics, method: str, prompt_type: str, pid: int, plen: int
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
    )


# ─── Plotting ─────────────────────────────────────────────────────────────────

def save_charts(results: List[TestResult], out_dir: Path) -> None:
    methods = ["direct", "sync_k2", "sync_k4", "async_k2", "async_k4"]
    colors  = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"]
    labels  = ["Direct", "Sync K=2", "Sync K=4", "Async K=2", "Async K=4"]

    def mean_field(field: str, method: str, ptype: str) -> Optional[float]:
        vals = [getattr(r, field) for r in results if r.method == method and r.prompt_type == ptype]
        return float(np.mean(vals)) if vals else None

    # ── 1. Throughput comparison ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ptype in zip(axes, ["easy", "hard"]):
        toks = [mean_field("tokens_per_second", m, ptype) or 0 for m in methods]
        bars = ax.bar(labels, toks, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Throughput — {ptype.capitalize()} Prompts")
        ax.set_ylabel("Tokens / second")
        ax.set_ylim(0, max(toks) * 1.2 if max(toks) > 0 else 1)
        for bar, val in zip(bars, toks):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_comparison.png", dpi=150)
    plt.close()

    # ── 2. Latency breakdown (draft / network / verify per round) ───────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    spec_methods = ["sync_k2", "sync_k4", "async_k2", "async_k4"]
    spec_labels  = ["Sync K=2", "Sync K=4", "Async K=2", "Async K=4"]
    for ax, ptype in zip(axes, ["easy", "hard"]):
        d_ms  = [mean_field("avg_draft_ms",   m, ptype) or 0 for m in spec_methods]
        n_ms  = [mean_field("avg_network_ms", m, ptype) or 0 for m in spec_methods]
        v_ms  = [mean_field("avg_verify_ms",  m, ptype) or 0 for m in spec_methods]
        x = np.arange(len(spec_labels))
        ax.bar(x, d_ms, label="Draft", color="#4CAF50")
        ax.bar(x, n_ms, bottom=d_ms, label="Network", color="#FF9800")
        ax.bar(x, v_ms, bottom=[d + n for d, n in zip(d_ms, n_ms)], label="Verify", color="#F44336")
        ax.set_title(f"Latency Breakdown / Round — {ptype.capitalize()}")
        ax.set_ylabel("ms per round")
        ax.set_xticks(x)
        ax.set_xticklabels(spec_labels, rotation=15)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "latency_breakdown.png", dpi=150)
    plt.close()

    # ── 3. Pipeline efficiency (async only) ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    async_methods = ["async_k2", "async_k4"]
    async_labels  = ["Async K=2", "Async K=4"]
    for ax, ptype in zip(axes, ["easy", "hard"]):
        effs = [mean_field("pipeline_efficiency", m, ptype) or 0 for m in async_methods]
        bars = ax.bar(async_labels, [e * 100 for e in effs], color=["#4CAF50", "#9C27B0"],
                      edgecolor="black")
        ax.set_title(f"Pipeline Efficiency — {ptype.capitalize()}")
        ax.set_ylabel("Full-Hit Rate (%)")
        ax.set_ylim(0, 110)
        for bar, val in zip(bars, effs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "pipeline_efficiency.png", dpi=150)
    plt.close()

    # ── 4. Bubble time (async only) ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, ptype in zip(axes, ["easy", "hard"]):
        bubbles = [mean_field("avg_bubble_ms", m, ptype) or 0 for m in async_methods]
        drafts  = [mean_field("avg_draft_ms",  m, ptype) or 0 for m in async_methods]
        x = np.arange(len(async_labels))
        ax.bar(x - 0.2, drafts,  0.4, label="Avg Draft ms",  color="#4CAF50")
        ax.bar(x + 0.2, bubbles, 0.4, label="Avg Bubble ms", color="#FF5722")
        ax.set_title(f"Draft Time vs Pipeline Bubble — {ptype.capitalize()}")
        ax.set_ylabel("ms per round")
        ax.set_xticks(x)
        ax.set_xticklabels(async_labels)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bubble_time.png", dpi=150)
    plt.close()

    # ── 5. Acceptance rate vs throughput speedup (scatter) ──────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    direct_tps = {
        ptype: float(np.mean([r.tokens_per_second for r in results
                               if r.method == "direct" and r.prompt_type == ptype]) or 1)
        for ptype in ["easy", "hard"]
    }
    marker_map = {"sync_k2": "o", "sync_k4": "s", "async_k2": "^", "async_k4": "D"}
    color_map  = {"sync_k2": "#FF9800", "sync_k4": "#F44336",
                  "async_k2": "#4CAF50", "async_k4": "#9C27B0"}
    for m in ["sync_k2", "sync_k4", "async_k2", "async_k4"]:
        for ptype in ["easy", "hard"]:
            pts = [r for r in results if r.method == m and r.prompt_type == ptype]
            if not pts:
                continue
            accs  = [r.acceptance_rate for r in pts]
            spdup = [r.tokens_per_second / direct_tps[ptype] for r in pts]
            ax.scatter(accs, spdup, marker=marker_map[m], color=color_map[m],
                       label=f"{m}/{ptype}", alpha=0.75, s=60)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8, label="Direct baseline")
    ax.set_xlabel("Acceptance Rate")
    ax.set_ylabel("Speedup vs Direct")
    ax.set_title("Acceptance Rate vs Speedup (relative to Direct)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "acceptance_vs_pipeline.png", dpi=150)
    plt.close()

    logger.info("Charts saved to %s", out_dir)


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_experiment():
    # Ensure output dir exists before log handler opens the file
    out_dir = Path(__file__).parent / "outputs_async"
    out_dir.mkdir(exist_ok=True)

    easy_prompts, hard_prompts = load_prompts(PROMPTS_FILE, count=PROMPT_COUNT)

    # ── Setup models ────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("ASYNC PIPELINE EXPERIMENT")
    logger.info("Draft model : %s (%s)", MODEL_NAME, MODEL_PATH)
    logger.info("Verify server: %s", SERVER_URL)
    logger.info("Methods: Direct | Sync K=2,4 | Async K=2,4 (lookahead=%d)", LOOKAHEAD)
    logger.info("Prompts: %d easy + %d hard", len(easy_prompts), len(hard_prompts))
    logger.info("=" * 80)

    logger.info("[Setup] Loading draft model...")
    model_manager = VLLMModelManager(MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    cloud_client = create_http_cloud_client(SERVER_URL, timeout=120.0)

    sync_client = EdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=cloud_client,
        max_new_tokens=MAX_TOKENS,
        temperature=BENCHMARK_TEMPERATURE,
    )
    async_client = AsyncEdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=cloud_client,
        max_new_tokens=MAX_TOKENS,
        temperature=BENCHMARK_TEMPERATURE,
        lookahead=LOOKAHEAD,
    )
    logger.info("[Setup] Ready")

    results: List[TestResult] = []
    prompt_sets = [("easy", easy_prompts), ("hard", hard_prompts)]
    total_tests = len(easy_prompts + hard_prompts) * (1 + 2 * len(K_VALUES))
    test_count = 0

    for ptype, plist in prompt_sets:
        logger.info("%s", "=" * 60)
        logger.info("%s prompts (%d)", ptype.upper(), len(plist))
        logger.info("%s", "=" * 60)

        for pd in plist:
            pid, ptext, plen = pd["id"], pd["text"], pd["length"]
            logger.info("[%s Prompt %d] len=%d chars", ptype.upper(), pid, plen)

            # ── Direct ──────────────────────────────────────────────────
            test_count += 1
            logger.info("  [%d/%d] Direct...", test_count, total_tests)
            toks, ms = test_direct(SERVER_URL, ptext)
            if toks > 0:
                tps = toks / (ms / 1000)
                results.append(TestResult(
                    method="direct", prompt_type=ptype, prompt_id=pid,
                    prompt_length=plen, tokens_generated=toks,
                    total_time_ms=ms, tokens_per_second=tps,
                ))
                logger.info("      %d tok  %.0fms  %.2f tok/s", toks, ms, tps)
            time.sleep(0.3)

            for k in K_VALUES:
                policy = lambda _rid, _dt, _k=k: _k

                # ── Sync K=k ──────────────────────────────────────────
                test_count += 1
                logger.info("  [%d/%d] Sync K=%d...", test_count, total_tests, k)
                try:
                    sm: RequestMetrics = sync_client.generate(
                        prompt=ptext,
                        policy=policy,
                        policy_name=f"SyncK{k}",
                    )
                    if sm.generated_tokens > 0:
                        r = result_from_sync_metrics(sm, f"sync_k{k}", ptype, pid, plen)
                        results.append(r)
                        logger.info(
                            "      %d tok  %.0fms  %.2f tok/s | rounds=%d  "
                            "accept=%.1f%%  rtt=%.0fms",
                            r.tokens_generated, r.total_time_ms, r.tokens_per_second,
                            r.num_rounds, r.acceptance_rate * 100, r.avg_rtt_ms,
                        )
                except Exception as e:
                    logger.error("Sync K=%d failed: %s", k, e)
                time.sleep(0.3)

                # ── Async K=k ─────────────────────────────────────────
                test_count += 1
                logger.info("  [%d/%d] Async K=%d (lookahead=%d)...",
                            test_count, total_tests, k, LOOKAHEAD)
                try:
                    am: AsyncRequestMetrics = async_client.generate(
                        prompt=ptext,
                        policy=policy,
                        policy_name=f"AsyncK{k}",
                    )
                    if am.generated_tokens > 0:
                        r = result_from_async_metrics(am, f"async_k{k}", ptype, pid, plen)
                        results.append(r)
                        logger.info(
                            "      %d tok  %.0fms  %.2f tok/s | rounds=%d  "
                            "accept=%.1f%%  pipe_eff=%.1f%%  bubble=%.1fms  "
                            "rollbacks=%d  waste=%.1f%%",
                            r.tokens_generated, r.total_time_ms, r.tokens_per_second,
                            r.num_rounds, r.acceptance_rate * 100,
                            r.pipeline_efficiency * 100, r.avg_bubble_ms,
                            int(r.rollback_rate * r.num_rounds),
                            r.prefetch_waste_ratio * 100,
                        )
                except Exception as e:
                    logger.error("Async K=%d failed: %s", k, e)
                time.sleep(0.3)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    all_methods = ["direct", "sync_k2", "sync_k4", "async_k2", "async_k4"]
    summary = {}

    for ptype in ["easy", "hard"]:
        for method in all_methods:
            entries = [r for r in results if r.method == method and r.prompt_type == ptype]
            if not entries:
                continue
            key = f"{ptype}_{method}"
            tps_vals = [r.tokens_per_second for r in entries]
            summary[key] = {
                "mean_tps": float(np.mean(tps_vals)),
                "median_tps": float(np.median(tps_vals)),
                "std_tps": float(np.std(tps_vals)),
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
                    "mean_effective_tpr": float(np.mean([r.effective_tokens_per_round for r in entries])),
                    "mean_overlap_utilisation": float(np.mean([r.overlap_utilisation for r in entries])),
                })

    # Print summary table
    header = f"{'Method':<14} {'Easy tok/s':>12} {'Hard tok/s':>12} {'Accept%':>8} {'PipeEff%':>10} {'Bubble ms':>10} {'Waste%':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for method in all_methods:
        e = summary.get(f"easy_{method}", {})
        h = summary.get(f"hard_{method}", {})
        acc   = f"{e.get('mean_acceptance', 0)*100:6.1f}" if method != "direct" else "    -"
        peff  = f"{e.get('mean_pipeline_efficiency', 0)*100:8.1f}" if method.startswith("async") else "       -"
        bub   = f"{e.get('mean_bubble_ms', 0):8.1f}" if method.startswith("async") else "       -"
        waste = f"{e.get('mean_prefetch_waste', 0)*100:6.1f}" if method.startswith("async") else "     -"
        logger.info(
            "%-14s %12.2f %12.2f %s %s %s %s",
            method,
            e.get("mean_tps", 0),
            h.get("mean_tps", 0),
            acc, peff, bub, waste,
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_file = out_dir / "async_results.json"
    with open(out_file, "w") as f:
        json.dump(
            {"results": [asdict(r) for r in results], "summary": summary},
            f,
            indent=2,
        )
    logger.info("Results saved: %s", out_file)

    # ── Save charts ────────────────────────────────────────────────────────
    try:
        save_charts(results, out_dir)
    except Exception as e:
        logger.error("Chart generation failed: %s", e)

    return results, summary


if __name__ == "__main__":
    run_experiment()
