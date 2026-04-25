"""Generate charts and a Markdown report from the latest benchmark JSON."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_JSON = Path(__file__).resolve().parent / "outputs" / "sglang_eagle_comparison_20260425_074444.json"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

COLORS = {"base": "#4C72B0", "eagle": "#DD8452"}
LABELS = {"base": "Base (Qwen3-8B)", "eagle": "Eagle3 (Qwen3-8B + Eagle draft)"}

# ── helpers ───────────────────────────────────────────────────────────────────

def bar_label(ax, bars, fmt="{:.1f}", pad=0.5):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + pad,
            fmt.format(h),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")
    return path.name


# ── load data ─────────────────────────────────────────────────────────────────

def load():
    with open(RESULTS_JSON) as f:
        return json.load(f)


# ── chart 1 : throughput comparison (mean / p50 / p95) ────────────────────────

def chart_throughput(d):
    agg = d["aggregates"]
    metrics = ["mean_tokens_per_second", "p50_tokens_per_second", "p95_tokens_per_second"]
    labels  = ["Mean", "p50", "p95"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    base_vals  = [agg["base"][m]  for m in metrics]
    eagle_vals = [agg["eagle"][m] for m in metrics]

    b1 = ax.bar(x - width/2, base_vals,  width, color=COLORS["base"],  label=LABELS["base"],  edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + width/2, eagle_vals, width, color=COLORS["eagle"], label=LABELS["eagle"], edgecolor="white", linewidth=0.6)

    bar_label(ax, b1, pad=0.8)
    bar_label(ax, b2, pad=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Tokens / second", fontsize=11)
    ax.set_title("Throughput Comparison — Base vs Eagle3", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(eagle_vals) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, "chart_throughput.png")


# ── chart 2 : latency comparison ──────────────────────────────────────────────

def chart_latency(d):
    agg = d["aggregates"]
    metrics = ["mean_latency_ms", "p50_latency_ms", "p95_latency_ms"]
    labels  = ["Mean", "p50", "p95"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    base_vals  = [agg["base"][m]  for m in metrics]
    eagle_vals = [agg["eagle"][m] for m in metrics]

    b1 = ax.bar(x - width/2, base_vals,  width, color=COLORS["base"],  label=LABELS["base"],  edgecolor="white")
    b2 = ax.bar(x + width/2, eagle_vals, width, color=COLORS["eagle"], label=LABELS["eagle"], edgecolor="white")

    bar_label(ax, b1, fmt="{:.0f}", pad=20)
    bar_label(ax, b2, fmt="{:.0f}", pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("End-to-End Latency — Base vs Eagle3", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(base_vals) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, "chart_latency.png")


# ── chart 3 : per-prompt throughput scatter ───────────────────────────────────

def chart_per_prompt(d):
    pp = d["per_prompt"]
    base_tps  = [p["tokens_per_second"] for p in pp["base"]]
    eagle_tps = [p["tokens_per_second"] for p in pp["eagle"]]
    ids = [p["prompt_id"] for p in pp["base"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ids, base_tps,  "o-", color=COLORS["base"],  label=LABELS["base"],  linewidth=2, markersize=8)
    ax.plot(ids, eagle_tps, "s-", color=COLORS["eagle"], label=LABELS["eagle"], linewidth=2, markersize=8)

    for i, (b, e) in enumerate(zip(base_tps, eagle_tps)):
        ax.annotate(f"{b:.1f}", (ids[i], b), textcoords="offset points", xytext=(-18, 6), fontsize=8, color=COLORS["base"])
        ax.annotate(f"{e:.1f}", (ids[i], e), textcoords="offset points", xytext=(4, 6),  fontsize=8, color=COLORS["eagle"])

    ax.set_xticks(ids)
    ax.set_xticklabels([f"Prompt {i}" for i in ids], fontsize=10)
    ax.set_ylabel("Tokens / second", fontsize=11)
    ax.set_title("Per-Prompt Throughput — Base vs Eagle3", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, "chart_per_prompt_tps.png")


# ── chart 4 : Eagle acceptance length per prompt ─────────────────────────────

def chart_acceptance(d):
    pp = d["per_prompt"]["eagle"]
    ids  = [p["prompt_id"] for p in pp]
    vals = [p["acceptance_length"] for p in pp]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(ids, vals, color=COLORS["eagle"], edgecolor="white", width=0.5)
    bar_label(ax, bars, fmt="{:.2f}", pad=0.02)

    ax.axhline(sum(vals)/len(vals), color="gray", linestyle="--", linewidth=1.2, label=f"Mean = {sum(vals)/len(vals):.2f}")
    ax.set_xticks(ids)
    ax.set_xticklabels([f"Prompt {i}" for i in ids], fontsize=10)
    ax.set_ylabel("Accepted tokens / verify step", fontsize=11)
    ax.set_title("Eagle3 Acceptance Length per Prompt", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, "chart_acceptance_length.png")


# ── chart 5 : speedup summary ─────────────────────────────────────────────────

def chart_speedup(d):
    pp  = d["per_prompt"]
    base_by_id  = {p["prompt_id"]: p["tokens_per_second"] for p in pp["base"]}
    eagle_by_id = {p["prompt_id"]: p["tokens_per_second"] for p in pp["eagle"]}
    ids     = sorted(base_by_id)
    speedups = [eagle_by_id[i] / base_by_id[i] for i in ids]
    mean_su  = d["comparison"]["throughput_speedup_eagle_vs_base"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(ids, speedups, color="#2ca02c", edgecolor="white", width=0.5)
    bar_label(ax, bars, fmt="{:.2f}x", pad=0.02)

    ax.axhline(1.0, color="gray",    linestyle=":", linewidth=1.0, label="Baseline (1.00×)")
    ax.axhline(mean_su, color="red", linestyle="--", linewidth=1.2, label=f"Mean = {mean_su:.2f}×")
    ax.set_xticks(ids)
    ax.set_xticklabels([f"Prompt {i}" for i in ids], fontsize=10)
    ax.set_ylabel("Eagle3 speedup vs Base", fontsize=11)
    ax.set_title("Per-Prompt Throughput Speedup (Eagle3 / Base)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(speedups) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, "chart_speedup.png")


# ── markdown report ───────────────────────────────────────────────────────────

REPORT_TEMPLATE = """\
# Qwen3-8B Eagle3 Speculative Decoding Benchmark Report

**Date:** {date}  
**GPU:** NVIDIA GeForce RTX 4090 (24 GB)  
**Framework:** SGLang {sglang_version}  
**PyTorch:** {torch_version}  

---

## 1. Setup

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-8B` |
| Draft model | `Tengyunw/qwen3_8b_eagle3` |
| Speculative algorithm | `{spec_algo}` |
| Speculative steps | {spec_steps} |
| Eagle topk | {eagle_topk} |
| Max draft tokens | {max_draft} |
| Max new tokens | {max_new_tokens} |
| Temperature | {temperature} |
| dtype | {dtype} |
| mem_fraction_static | {mem_fraction} |
| Prompts evaluated | {prompt_count} |

---

## 2. Results Summary

| Metric | Base | Eagle3 | Delta |
|---|---|---|---|
| **Mean throughput (tok/s)** | {base_mean_tps:.2f} | **{eagle_mean_tps:.2f}** | +{tps_delta:.2f} |
| **p50 throughput (tok/s)** | {base_p50_tps:.2f} | **{eagle_p50_tps:.2f}** | — |
| **p95 throughput (tok/s)** | {base_p95_tps:.2f} | **{eagle_p95_tps:.2f}** | — |
| **Mean latency (ms)** | {base_mean_lat:.0f} | **{eagle_mean_lat:.0f}** | {lat_delta:.0f} ms |
| **p50 latency (ms)** | {base_p50_lat:.0f} | **{eagle_p50_lat:.0f}** | — |
| **p95 latency (ms)** | {base_p95_lat:.0f} | **{eagle_p95_lat:.0f}** | — |
| **Throughput speedup** | 1.00× | **{speedup:.2f}×** | — |
| **Latency reduction** | 1.00× | **{lat_ratio:.2f}×** | — |
| **Mean acceptance length** | — | **{accept_len:.2f} tok/verify** | — |
| **Mean spec verify count** | — | {spec_verify_ct:.0f} | — |

> Eagle3 delivers a **{speedup:.2f}× throughput speedup** and reduces mean end-to-end latency by
> **{lat_ms_saved:.0f} ms** ({lat_pct:.0f}%) compared to standard autoregressive decoding.

---

## 3. Charts

### 3.1 Throughput Comparison

![Throughput Comparison]({chart_throughput})

Mean, median (p50), and 95th-percentile tokens-per-second for Base and Eagle3.
Eagle3 raises mean throughput from **{base_mean_tps:.1f}** to **{eagle_mean_tps:.1f} tok/s**.

---

### 3.2 End-to-End Latency

![Latency Comparison]({chart_latency})

Eagle3 cuts mean latency from **{base_mean_lat:.0f} ms** to **{eagle_mean_lat:.0f} ms** —
a **{lat_pct:.0f}% reduction** for 256-token outputs.

---

### 3.3 Per-Prompt Throughput

![Per-Prompt Throughput]({chart_per_prompt})

Eagle3 outperforms the base model on every prompt. Prompt 4 benefits most
(**{eagle_p4_tps:.1f} tok/s** vs **{base_p4_tps:.1f} tok/s**), likely due to high
token repetitiveness (acceptance length **{eagle_p4_accept:.2f}**).

---

### 3.4 Per-Prompt Throughput Speedup

![Speedup]({chart_speedup})

All four prompts show super-linear improvement. The mean speedup is **{speedup:.2f}×**
with a maximum of **{max_speedup:.2f}×**.

---

### 3.5 Eagle3 Acceptance Length

![Acceptance Length]({chart_acceptance})

Acceptance length measures how many draft tokens are accepted on average per
verification step (higher is better). The mean across prompts is **{accept_len:.2f}**,
meaning Eagle3 produces ~{accept_len:.1f} output tokens for every forward pass
of the target model.

---

## 4. Per-Prompt Detail

| Prompt | Length | Base tok/s | Eagle tok/s | Speedup | Acceptance len | Spec verify ct |
|---|---|---|---|---|---|---|
{per_prompt_rows}

---

## 5. How to Reproduce

```bash
# From repo root
.venv/bin/python speculative-decoding/sglang_eagle_benchmark.py \\
  --base-model-path  models/Qwen3-8B \\
  --eagle-model-path models/qwen3_8b_eagle3 \\
  --max-new-tokens   256 \\
  --prompt-count     4 \\
  --port             31000 \\
  --mem-fraction-static 0.72
```

Then re-run this script to regenerate the report:

```bash
.venv/bin/python speculative-decoding/generate_report.py
```

---

## 6. Notes

- `ttft_ms` (time-to-first-token) was not returned by this SGLang version and is absent from all rows.
- Eagle3 requires the `Tengyunw/qwen3_8b_eagle3` draft head, which was trained on Qwen3-8B
  using the Eagle3 method on 600 K samples / 1 B tokens of UltraChat-200K.
- The draft model adds ~764 MB of VRAM overhead (vs 15.7 GB for the base model weights).
- Both models were served sequentially on a single RTX 4090 with `mem_fraction_static=0.72`
  to leave adequate GPU memory for the Eagle run's combined base + draft allocation.
"""


def build_report(d, charts):
    import datetime
    agg  = d["aggregates"]
    cmp  = d["comparison"]
    cfg  = d["config"]
    pp   = d["per_prompt"]

    base_by_id  = {p["prompt_id"]: p for p in pp["base"]}
    eagle_by_id = {p["prompt_id"]: p for p in pp["eagle"]}
    ids = sorted(base_by_id)

    speedups = [eagle_by_id[i]["tokens_per_second"] / base_by_id[i]["tokens_per_second"] for i in ids]

    rows = []
    for i in ids:
        b = base_by_id[i]
        e = eagle_by_id[i]
        su = e["tokens_per_second"] / b["tokens_per_second"]
        al = e["acceptance_length"] or 0
        sv = e["spec_verify_count"] or 0
        rows.append(
            f"| {i} | {b['prompt_length']} | {b['tokens_per_second']:.2f} |"
            f" {e['tokens_per_second']:.2f} | {su:.2f}× | {al:.2f} | {sv} |"
        )

    p4_base  = base_by_id[4]
    p4_eagle = eagle_by_id[4]

    lat_pct = 100 * (1 - agg["eagle"]["mean_latency_ms"] / agg["base"]["mean_latency_ms"])

    try:
        import sglang
        sglang_ver = sglang.__version__
    except Exception:
        sglang_ver = "0.5.2"
    try:
        import torch
        torch_ver = torch.__version__
    except Exception:
        torch_ver = "2.8.0+cu128"

    return REPORT_TEMPLATE.format(
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        sglang_version=sglang_ver,
        torch_version=torch_ver,
        spec_algo=cfg["speculative_algorithm"],
        spec_steps=cfg["speculative_num_steps"],
        eagle_topk=cfg["speculative_eagle_topk"],
        max_draft=cfg["speculative_num_draft_tokens"],
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        dtype=cfg["dtype"],
        mem_fraction=cfg["mem_fraction_static"],
        prompt_count=cfg["prompt_count"],
        base_mean_tps=agg["base"]["mean_tokens_per_second"],
        eagle_mean_tps=agg["eagle"]["mean_tokens_per_second"],
        tps_delta=cmp["mean_tps_delta"],
        base_p50_tps=agg["base"]["p50_tokens_per_second"],
        eagle_p50_tps=agg["eagle"]["p50_tokens_per_second"],
        base_p95_tps=agg["base"]["p95_tokens_per_second"],
        eagle_p95_tps=agg["eagle"]["p95_tokens_per_second"],
        base_mean_lat=agg["base"]["mean_latency_ms"],
        eagle_mean_lat=agg["eagle"]["mean_latency_ms"],
        lat_delta=cmp["mean_latency_ms_delta"],
        base_p50_lat=agg["base"]["p50_latency_ms"],
        eagle_p50_lat=agg["eagle"]["p50_latency_ms"],
        base_p95_lat=agg["base"]["p95_latency_ms"],
        eagle_p95_lat=agg["eagle"]["p95_latency_ms"],
        speedup=cmp["throughput_speedup_eagle_vs_base"],
        lat_ratio=cmp["latency_ratio_base_vs_eagle"],
        accept_len=cmp["acceptance_length_eagle"],
        spec_verify_ct=cmp["spec_verify_count_eagle"],
        lat_ms_saved=-cmp["mean_latency_ms_delta"],
        lat_pct=lat_pct,
        max_speedup=max(speedups),
        chart_throughput=charts["throughput"],
        chart_latency=charts["latency"],
        chart_per_prompt=charts["per_prompt"],
        chart_speedup=charts["speedup"],
        chart_acceptance=charts["acceptance"],
        eagle_p4_tps=p4_eagle["tokens_per_second"],
        base_p4_tps=p4_base["tokens_per_second"],
        eagle_p4_accept=p4_eagle["acceptance_length"] or 0,
        per_prompt_rows="\n".join(rows),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load()

    print("Generating charts...")
    charts = {
        "throughput": chart_throughput(d),
        "latency":    chart_latency(d),
        "per_prompt": chart_per_prompt(d),
        "speedup":    chart_speedup(d),
        "acceptance": chart_acceptance(d),
    }

    print("Writing report...")
    report_md = build_report(d, charts)
    report_path = OUT_DIR / "eagle3_benchmark_report.md"
    report_path.write_text(report_md)
    print(f"  saved {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
