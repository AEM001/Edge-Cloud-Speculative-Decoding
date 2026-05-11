#!/usr/bin/env python3
"""
Analyze quick_test results and generate charts + a markdown report.
Focus: mathematical/statistical metrics from acceptance rates, latency, speedup.
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Matplotlib setup — headless + publication-quality defaults
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
RESULTS_JSON = BASE_DIR / "quick_test_results.json"
ROUNDS_JSONL = BASE_DIR / "quick_test_rounds.jsonl"
SUMMARY_JSON = BASE_DIR / "quick_test_summary.json"
OUT_DIR = BASE_DIR / "analysis_output"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def stream_jsonl(path: Path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

print("Loading results ...")
results: List[Dict] = load_json(RESULTS_JSON)
summary: Dict = load_json(SUMMARY_JSON)

# --- aggregate per-prompt stats ---
prompt_records: List[Dict] = []
for r in results:
    rec = {
        "method": r["method"],
        "method_family": r["method_family"],
        "network": r["network"],
        "prompt_id": r["prompt_id"],
        "tokens_generated": r["output"]["tokens_generated"],
        "total_time_ms": r["output"]["total_time_ms"],
        "tokens_per_second": r["output"]["tokens_per_second"],
        "rounds": r["speculative"]["rounds"],
        "acceptance_rate": r["speculative"]["acceptance_rate"],
        "accepted_draft_per_round": r["speculative"]["accepted_draft_per_round"],
        "generated_per_round": r["speculative"]["generated_per_round"],
        "local_draft_ms": r["timing"]["local_draft_ms"],
        "server_model_ms": r["timing"]["server_model_ms"],
        "simulated_network_ms": r["timing"]["simulated_network_ms"],
        "avg_rtt_ms": r["timing"]["avg_rtt_ms"],
        "prefetched_tokens": r["async_detail"]["prefetched_tokens"],
        "selected_offset": r["async_detail"]["selected_offset"],
        "exposed_branch_ms": r["async_detail"]["exposed_branch_ms"],
    }
    prompt_records.append(rec)

# --- per-round stats (streamed from JSONL) ---
sync_rounds: List[Dict] = []
tree_rounds: List[Dict] = []
tree_slots_by_network: Dict[str, List[Dict]] = defaultdict(list)
sync_rounds_by_network: Dict[str, List[Dict]] = defaultdict(list)
tree_rounds_by_network: Dict[str, List[Dict]] = defaultdict(list)
round_count = 0
for row in stream_jsonl(ROUNDS_JSONL):
    round_count += 1
    mf = row.get("method_family", "")
    if mf == "sync_spec":
        sync_rounds.append(row["detail"])
        sync_rounds_by_network[row.get("network", "")].append(row["detail"])
    elif mf == "tree_async":
        tree_rounds.append(row["detail"])
        tree_rounds_by_network[row.get("network", "")].append(row["detail"])

for r in results:
    if r.get("method_family") != "tree_async":
        continue
    for slot in r.get("raw", {}).get("slot_details", []):
        tree_slots_by_network[r["network"]].append(slot)

print(f"  -> {len(prompt_records)} prompt records, {round_count} round records")
print(f"  -> {len(sync_rounds)} sync rounds, {len(tree_rounds)} tree rounds")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def stdev(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def percentile(vals: List[float], p: float) -> float:
    s = sorted(vals)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def ci95(vals: List[float]) -> Tuple[float, float]:
    """Approximate 95% CI using t≈2 for n≥20."""
    m = mean(vals)
    sd = stdev(vals)
    n = len(vals)
    if n < 2:
        return (m, m)
    err = 1.96 * sd / math.sqrt(n)
    return (m - err, m + err)


def grouped_by(records: List[Dict], key: str) -> Dict[str, List[Dict]]:
    d = defaultdict(list)
    for r in records:
        d[r[key]].append(r)
    return dict(d)


def filter_records(records: List[Dict], **kwargs) -> List[Dict]:
    out = []
    for r in records:
        if all(r.get(k) == v for k, v in kwargs.items()):
            out.append(r)
    return out


def raw_mean(records: List[Dict], key: str) -> float:
    return mean([float(r.get(key, 0.0) or 0.0) for r in records])


def raw_sum(records: List[Dict], key: str) -> float:
    return sum(float(r.get(key, 0.0) or 0.0) for r in records)


def tree_pipeline_stats(nw: str) -> Dict[str, float]:
    slots = tree_slots_by_network.get(nw, [])
    branch_ms = [float(s.get("branch_draft_ms", 0.0) or 0.0) for s in slots]
    base_ms = [float(s.get("base_draft_ms", 0.0) or 0.0) for s in slots]
    wait_ms = [float(s.get("total_wait_ms", 0.0) or 0.0) for s in slots]
    exposed_ms = [float(s.get("exposed_branch_ms", 0.0) or 0.0) for s in slots]
    prefetched = [float(s.get("prefetched_tokens", 0.0) or 0.0) for s in slots]
    selected_offsets = [int(s.get("selected_offset", 0) or 0) for s in slots]
    branch_tokens = [float(s.get("streamed_branch_tokens", 0.0) or 0.0) for s in slots]
    full_offset = sum(1 for x in selected_offsets if x == 8)
    base_offset = sum(1 for x in selected_offsets if x == 0)
    reused = sum(1 for x in prefetched if x > 0)
    useful_prefetch_tokens = sum(prefetched)
    streamed_tokens = sum(branch_tokens)
    return {
        "n": len(slots),
        "base_draft_ms": mean(base_ms),
        "branch_draft_ms": mean(branch_ms),
        "wait_ms": mean(wait_ms),
        "exposed_branch_ms": mean(exposed_ms),
        "hidden_branch_pct": 100.0 * (1.0 - sum(exposed_ms) / sum(branch_ms)) if sum(branch_ms) > 0 else 0.0,
        "reuse_round_pct": 100.0 * reused / len(slots) if slots else 0.0,
        "full_offset_pct": 100.0 * full_offset / len(slots) if slots else 0.0,
        "base_offset_pct": 100.0 * base_offset / len(slots) if slots else 0.0,
        "prefetched_mean": mean(prefetched),
        "prefetched_p50": percentile(prefetched, 50),
        "prefetched_p90": percentile(prefetched, 90),
        "useful_prefetch_tokens": useful_prefetch_tokens,
        "streamed_branch_tokens": streamed_tokens,
        "useful_branch_token_pct": 100.0 * useful_prefetch_tokens / streamed_tokens if streamed_tokens > 0 else 0.0,
    }


def paired_tree_vs_sync(nw: str) -> Dict[str, float]:
    sync_recs = filter_records(prompt_records, method="sync_k8", network=nw)
    tree_recs = filter_records(prompt_records, method="tree_k8_b3", network=nw)
    sync_by_pid = {r["prompt_id"]: r for r in sync_recs}
    tree_by_pid = {r["prompt_id"]: r for r in tree_recs}
    ratios = []
    wins = 0
    round_delta = []
    for pid in sorted(set(sync_by_pid) & set(tree_by_pid)):
        s = sync_by_pid[pid]
        t = tree_by_pid[pid]
        if s["tokens_per_second"] > 0:
            ratio = t["tokens_per_second"] / s["tokens_per_second"]
            ratios.append(ratio)
            wins += int(ratio > 1.0)
        round_delta.append(t["rounds"] - s["rounds"])
    return {
        "n": len(ratios),
        "mean": mean(ratios),
        "p50": percentile(ratios, 50),
        "min": min(ratios) if ratios else 0.0,
        "max": max(ratios) if ratios else 0.0,
        "wins": wins,
        "round_delta_mean": mean(round_delta),
    }


# ---------------------------------------------------------------------------
# 1. Acceptance length distribution (sync rounds)
# ---------------------------------------------------------------------------

def build_acceptance_distribution(rounds: List[Dict], max_accept: int = 8) -> List[int]:
    counts = Counter()
    for rd in rounds:
        acc = rd.get("accepted", 0)
        counts[acc] += 1
    return [counts.get(i, 0) for i in range(max_accept + 1)]


sync_accept_dist = build_acceptance_distribution(sync_rounds)
tree_accept_dist = build_acceptance_distribution(tree_rounds)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(9)
width = 0.35
bars1 = ax.bar(x - width / 2, sync_accept_dist, width, label="Sync K=8", color="#1f77b4")
bars2 = ax.bar(x + width / 2, tree_accept_dist, width, label="Tree K=8 B=3", color="#ff7f0e")
ax.set_xlabel("Accepted Tokens per Round")
ax.set_ylabel("Count (number of rounds)")
ax.set_title("Acceptance Length Distribution (Per Round)")
ax.set_xticks(x)
ax.legend()
for bar in bars1 + bars2:
    h = bar.get_height()
    if h > 0:
        ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig01_acceptance_distribution.png", dpi=200)
plt.close(fig)
print("  -> fig01_acceptance_distribution.png")

# ---------------------------------------------------------------------------
# 2. Throughput comparison (grouped bar)
# ---------------------------------------------------------------------------

networks = ["good", "medium"]
methods = ["direct", "sync_k8", "tree_k8_b3"]
throughput = {nw: {m: summary[nw][m]["tokens_per_second"] for m in methods} for nw in networks}

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(networks))
width = 0.25
colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
for idx, m in enumerate(methods):
    vals = [throughput[nw][m] for nw in networks]
    ax.bar(x + idx * width - width, vals, width, label=m, color=colors[idx])
ax.set_ylabel("Tokens / Second")
ax.set_title("Throughput by Network Condition & Method")
ax.set_xticks(x)
ax.set_xticklabels(networks)
ax.legend()
for i, nw in enumerate(networks):
    for idx, m in enumerate(methods):
        v = throughput[nw][m]
        ax.annotate(f"{v:.1f}", xy=(i + idx * width - width, v),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig02_throughput_comparison.png", dpi=200)
plt.close(fig)
print("  -> fig02_throughput_comparison.png")

# ---------------------------------------------------------------------------
# 3. Time breakdown (critical path only - showing how pre-draft is hidden)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, nw in zip(axes, networks):
    labels = ["Direct", "Sync", "Tree"]
    verification_time = [summary[nw][m]["server_model_ms"] for m in methods]
    network_time = [summary[nw][m]["simulated_network_ms"] for m in methods]
    # For tree, use avg_branch_draft_ms (actual branch draft time, excludes pre-draft)
    # For sync and direct, use local_draft_ms
    draft_time = []
    for m in methods:
        if m == "tree_k8_b3":
            draft_time.append(tree_pipeline_stats(nw)["branch_draft_ms"])
        else:
            draft_time.append(summary[nw][m].get("local_draft_ms", 0.0))

    x = np.arange(len(labels))
    width = 0.5
    ax.bar(x, verification_time, width, label="Verification time", color="#1f77b4")
    ax.bar(x, network_time, width, bottom=verification_time, label="Network RTT", color="#ff7f0e")
    ax.bar(x, draft_time, width, bottom=[v + n for v, n in zip(verification_time, network_time)], label="Draft time", color="#2ca02c")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Critical Path Latency — {nw} network")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "fig03_time_breakdown.png", dpi=200)
plt.close(fig)
print("  -> fig03_time_breakdown.png")

# ---------------------------------------------------------------------------
# 4. Prefetch token distribution (histogram + CDF)
# ---------------------------------------------------------------------------

tree_prefetch = [r["prefetched_tokens"] for r in prompt_records if r["method_family"] == "tree_async"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# histogram
ax1.hist(tree_prefetch, bins=range(0, int(max(tree_prefetch)) + 2), edgecolor="black", color="#9467bd")
ax1.set_xlabel("Prefetched Tokens")
ax1.set_ylabel("Frequency")
ax1.set_title("Prefetch Token Distribution (Tree)")
# CDF
sorted_pf = sorted(tree_prefetch)
yvals = np.arange(1, len(sorted_pf) + 1) / len(sorted_pf) * 100
ax2.plot(sorted_pf, yvals, marker=".", linestyle="-", color="#9467bd")
ax2.axhline(50, color="gray", linestyle="--", linewidth=0.8)
ax2.axhline(90, color="gray", linestyle="--", linewidth=0.8)
ax2.set_xlabel("Prefetched Tokens")
ax2.set_ylabel("CDF (%)")
ax2.set_title("Prefetch Token CDF")
plt.tight_layout()
fig.savefig(OUT_DIR / "fig04_prefetch_distribution.png", dpi=200)
plt.close(fig)
print("  -> fig04_prefetch_distribution.png")

# ---------------------------------------------------------------------------
# 5. Tree selected offset distribution
# ---------------------------------------------------------------------------

tree_offsets = [int(r["selected_offset"]) for r in prompt_records if r["method_family"] == "tree_async"]
offset_counts = Counter(tree_offsets)
max_off = max(offset_counts.keys()) if offset_counts else 0
labels_off = [str(i) for i in range(max_off + 1)]
counts_off = [offset_counts.get(i, 0) for i in range(max_off + 1)]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(labels_off, counts_off, color="#8c564b", edgecolor="black")
ax.set_xlabel("Selected Offset")
ax.set_ylabel("Count (tree rounds)")
ax.set_title("Tree Branch Selected Offset Distribution")
for i, v in enumerate(counts_off):
    if v > 0:
        ax.text(i, v + 1, str(v), ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig05_tree_offset_distribution.png", dpi=200)
plt.close(fig)
print("  -> fig05_tree_offset_distribution.png")

# REMOVED: Figure 06 (RTT vs acceptance scatter)

# Calculate speedups for statistics (used in report tables)
speedups = []
for nw in networks:
    sync_recs = filter_records(prompt_records, method="sync_k8", network=nw)
    tree_recs = filter_records(prompt_records, method="tree_k8_b3", network=nw)
    sync_by_pid = {r["prompt_id"]: r for r in sync_recs}
    tree_by_pid = {r["prompt_id"]: r for r in tree_recs}
    for pid in sorted(set(sync_by_pid) & set(tree_by_pid)):
        s = sync_by_pid[pid]["tokens_per_second"]
        t = tree_by_pid[pid]["tokens_per_second"]
        if s > 0:
            speedups.append(t / s)

# REMOVED: Figure 07 (speedup per prompt histogram)

# REMOVED: Figure 08 (draft vs server time scatter)

# REMOVED: Figure 09 (acceptance rate distribution)

# REMOVED: Figure 10 (RTT timeline)

# ---------------------------------------------------------------------------
# Statistical tables
# ---------------------------------------------------------------------------

def fmt_stat(vals: List[float]) -> Dict[str, float]:
    return {
        "n": len(vals),
        "mean": mean(vals),
        "std": stdev(vals),
        "min": min(vals) if vals else 0.0,
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p99": percentile(vals, 99),
        "max": max(vals) if vals else 0.0,
    }


stats_tables = {}

# Table 1: throughput by method/network
for name, key in [("Throughput (tok/s)", "tokens_per_second"),
                   ("Total Time (ms)", "total_time_ms"),
                   ("Acceptance Rate", "acceptance_rate")]:
    rows = {}
    for nw in networks:
        for m in ["sync_k8", "tree_k8_b3"]:
            vals = [r[key] for r in prompt_records if r["network"] == nw and r["method"] == m]
            rows[f"{nw}_{m}"] = fmt_stat(vals)
    stats_tables[name] = rows

# Table 2: speedup
ci_low, ci_high = ci95(speedups)
speedup_stats = {
    "n": len(speedups),
    "mean": mean(speedups),
    "std": stdev(speedups),
    "min": min(speedups),
    "p50": percentile(speedups, 50),
    "p90": percentile(speedups, 90),
    "max": max(speedups),
    "ci95_low": ci_low,
    "ci95_high": ci_high,
}

# Table 3: per-round acceptance distribution with percentages
total_sync = sum(sync_accept_dist)
total_tree = sum(tree_accept_dist)
sync_pct = [c / total_sync * 100 if total_sync else 0 for c in sync_accept_dist]
tree_pct = [c / total_tree * 100 if total_tree else 0 for c in tree_accept_dist]

# Table 5: tree prefetch stats
tree_pf_stats = fmt_stat(tree_prefetch)
tree_pipeline_by_network = {nw: tree_pipeline_stats(nw) for nw in networks}
paired_by_network = {nw: paired_tree_vs_sync(nw) for nw in networks}

# ---------------------------------------------------------------------------
# Markdown Report
# ---------------------------------------------------------------------------

report_path = OUT_DIR / "analysis_report.md"
with open(report_path, "w") as f:
    f.write("# 快速测试结果 — 统计分析报告\n\n")
    f.write(f"**生成时间:** {os.popen('date').read().strip()}\n")
    f.write(f"**数据文件:** `quick_test_results.json`, `quick_test_rounds.jsonl`, `quick_test_summary.json`\n\n")

    # Executive summary
    f.write("## 执行摘要\n\n")
    f.write(f"- **提示词记录数:** {len(prompt_records)}\n")
    f.write(f"- **同步轮数:** {len(sync_rounds)} | **树形轮数:** {len(tree_rounds)}\n")
    f.write(f"- **加速比 (Tree vs Sync):** {speedup_stats['mean']:.4f}x ± {speedup_stats['std']:.4f} (n={speedup_stats['n']})\n")
    f.write(f"- **95% 置信区间:** [{speedup_stats['ci95_low']:.4f}, {speedup_stats['ci95_high']:.4f}]\n")
    f.write(f"- **接受率分布:** 完全接受 (8/8) ≈ {sync_pct[8]:.1f}%, 短接受 (0–2) ≈ {sum(sync_pct[:3]):.1f}%\n\n")

    f.write("## 核心结论: Tree Async 是否真正有效\n\n")
    f.write("这个项目的目标不是单纯比较模型速度，而是验证 **edge draft model + cloud verification model** 在真实网络 RTT 下，能否让本地预草稿窗口吸收远端 verification + RTT 的等待时间。\n\n")
    f.write("| 网络 | Tree/Sync 平均加速 | Tree 赢的样本 | 平均轮数差(Tree-Sync) | verification+RTT 被草稿窗口吸收程度 | 复用轮次比例 | P90 预取 tokens | 有用预取/流式分支 tokens |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for nw in networks:
        p = paired_by_network[nw]
        t = tree_pipeline_by_network[nw]
        f.write(
            f"| {nw} | {p['mean']:.3f}x | {p['wins']}/{p['n']} | {p['round_delta_mean']:.2f} | "
            f"{t['hidden_branch_pct']:.2f}% | {t['reuse_round_pct']:.1f}% | {t['prefetched_p90']:.1f} | {t['useful_branch_token_pct']:.1f}% |\n"
        )
    f.write("\n")
    f.write("解释: `exposed_branch_ms` 接近 0 不是草稿时间不存在；在这个场景下，更准确的说法是: edge 的预草稿窗口把 cloud verification + RTT 的等待吸收进了本地 draft pipeline。真正应该从 raw slot details 读 `branch_draft_ms`、`base_wait_ms/total_wait_ms`、`prefetched_tokens` 和 `selected_offset`。\n\n")

    # Throughput table
    f.write("## 表1: 各网络条件与方法下的吞吐量\n\n")
    f.write("| 网络 | 方法 | n | 平均 tok/s | 标准差 | 最小值 | P50 | P90 | 最大值 |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for nw in networks:
        for m in ["sync_k8", "tree_k8_b3"]:
            s = stats_tables["Throughput (tok/s)"][f"{nw}_{m}"]
            f.write(f"| {nw} | {m} | {s['n']} | {s['mean']:.2f} | {s['std']:.2f} | {s['min']:.2f} | {s['p50']:.2f} | {s['p90']:.2f} | {s['max']:.2f} |\n")
    f.write("\n")

    # Acceptance distribution table
    f.write("## 表2: 接受长度分布 (每轮)\n\n")
    f.write("| 接受数 | 同步计数 | 同步 % | 树形计数 | 树形 % |\n")
    f.write("|---:|---:|---:|---:|---:|\n")
    for i in range(9):
        f.write(f"| {i} | {sync_accept_dist[i]} | {sync_pct[i]:.2f}% | {tree_accept_dist[i]} | {tree_pct[i]:.2f}% |\n")
    f.write("\n")

    # Latency breakdown table
    f.write("## 表3: 延迟分解 (端到端摘要 + raw tree draft)\n\n")
    f.write("| 网络 | 方法 | 总耗时 ms | 验证时间 ms | 网络 RTT ms | 草稿时间 ms | 轮数 |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|\n")
    for nw in networks:
        for m in methods:
            d = summary[nw][m]
            if m == "tree_k8_b3":
                ld = tree_pipeline_by_network[nw]["branch_draft_ms"]
            else:
                ld = d.get('local_draft_ms', 0.0)
            rd = d.get('rounds', 'n/a')
            f.write(f"| {nw} | {m} | {d['total_time_ms']:.1f} | {d['server_model_ms']:.1f} | {d['simulated_network_ms']:.1f} | {ld:.1f} | {rd} |\n")
    f.write("\n")

    f.write("## 表4: Raw Tree Pipeline Diagnostics\n\n")
    f.write("| 网络 | slots | base draft ms/slot | branch draft ms/slot | verification+RTT wait ms/slot | exposed tail ms | wait absorbed by draft % | offset=8 selected | offset=0 selected | prefetch mean/P50/P90 |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for nw in networks:
        t = tree_pipeline_by_network[nw]
        f.write(
            f"| {nw} | {t['n']} | {t['base_draft_ms']:.2f} | {t['branch_draft_ms']:.2f} | "
            f"{t['wait_ms']:.2f} | {t['exposed_branch_ms']:.4f} | {t['hidden_branch_pct']:.2f}% | "
            f"{t['full_offset_pct']:.1f}% | {t['base_offset_pct']:.1f}% | "
            f"{t['prefetched_mean']:.2f}/{t['prefetched_p50']:.1f}/{t['prefetched_p90']:.1f} |\n"
        )
    f.write("\n")
    f.write("这张表来自 `quick_test_results.json -> raw.slot_details`，不是 summary JSON。它显示 Tree 的收益来自 overlap: 远端 verification + RTT 等待期间，edge 并没有闲置，而是在生成下一轮候选分支；真正决定收益的是 full-acceptance offset 是否命中、预取 token 是否能被下一轮 base draft 复用，以及复用后是否仍保持 K=8 发送给 verifier。\n\n")

    
    # Mathematical notes
    f.write("## 数学说明\n\n")
    f.write("### 接受率定义\n\n")
    f.write("对于每个提示词，接受率计算如下:\n\n")
    f.write("$$\\text{acceptance\\_rate} = \\frac{\\text{accepted\\_draft\\_tokens}}{\\text{drafted\\_tokens}}$$\n\n")
    f.write("### 加速比定义\n\n")
    f.write("每个提示词的加速比是 Tree 相对于 Sync 的吞吐量 (tok/s) 比率:\n\n")
    f.write("$$\\text{speedup} = \\frac{\\text{tok/s}_{\\text{tree}}}{\\text{tok/s}_{\\text{sync}}}$$\n\n")
    f.write("### 95% 置信区间\n\n")
    f.write("使用正态近似，$z = 1.96$:\n\n")
    f.write("$$\\text{CI} = \\bar{x} \\pm 1.96 \\frac{s}{\\sqrt{n}}$$\n\n")
    f.write(f"其中 $\\bar{{x}} = {speedup_stats['mean']:.4f}$, $s = {speedup_stats['std']:.4f}$, $n = {speedup_stats['n']}$。\n\n")
    f.write("### 接受分布观察\n\n")
    f.write("接受长度分布呈现偏态:\n\n")
    f.write(f"- **主峰:** 8 tokens 完全接受, ~{sync_pct[8]:.1f}% 的轮次\n")
    f.write(f"- **长尾:** 0–2 tokens 短接受, ~{sum(sync_pct[:3]):.1f}% 的轮次\n")
    f.write(f"- **中间:** 3–7 tokens, ~{sum(sync_pct[3:8]):.1f}% 的轮次\n\n")
    f.write("完全接受是主导模式，短接受情况占次要比例。\n\n")

    # Figures
    f.write("## 图表\n\n")
    f.write("### 图 01: 接受长度分布\n\n")
    f.write("![图 01](fig01_acceptance_distribution.png)\n\n")
    f.write("### 图 02: 吞吐量比较\n\n")
    f.write("![图 02](fig02_throughput_comparison.png)\n\n")
    f.write("### 图 03: 关键路径延迟分解 (验证时间 / 网络 RTT / 草稿时间)\n\n")
    f.write("![图 03](fig03_time_breakdown.png)\n\n")
    f.write("### 图 04: 预取 Token 分布\n\n")
    f.write("![图 04](fig04_prefetch_distribution.png)\n\n")
    f.write("### 图 05: 树形分支选择偏移量分布\n\n")
    f.write("![图 05](fig05_tree_offset_distribution.png)\n\n")

print(f"Report written to {report_path}")
print("Done.")
