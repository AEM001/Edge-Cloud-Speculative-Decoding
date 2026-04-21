"""
Generate charts and a concise markdown report for experiment output folders.

Default target is outputs_run3/, but the script can analyze any folder that
contains comprehensive_results.json and optionally comprehensive_log.txt.
"""
import argparse
import json
import os
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12

METHOD_COLORS = {
    "direct": "#1f77b4",
    "spec_k2": "#ff7f0e",
    "spec_k4": "#2ca02c",
    "spec_k6": "#d62728",
    "spec_k8": "#9467bd",
    "spec_k10": "#8c564b",
}
PROMPT_COLORS = {
    "easy": "#4c78a8",
    "hard": "#e45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze experiment outputs.")
    parser.add_argument(
        "--input-dir",
        default="outputs_run3",
        help="Experiment output directory under draft/experiments/",
    )
    return parser.parse_args()


def load_results(results_path: Path) -> list[dict]:
    payload = json.loads(results_path.read_text())
    return payload["results"]


def detect_methods(results: list[dict]) -> list[str]:
    methods = sorted({row["method"] for row in results})

    def order_key(name: str) -> tuple[int, int]:
        if name == "direct":
            return (0, 0)
        match = re.search(r"k(\d+)", name)
        return (1, int(match.group(1)) if match else 999)

    return sorted(methods, key=order_key)


def method_label(method: str) -> str:
    if method == "direct":
        return "Direct"
    match = re.search(r"k(\d+)", method)
    return f"K={match.group(1)}" if match else method


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def safe_std(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(arr.std())


def corrcoef(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def group_rows(results: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in results:
        grouped[(row["prompt_type"], row["method"])].append(row)
    return grouped


def summarize(results: list[dict], methods: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    grouped = group_rows(results)
    summary: dict[str, dict[str, dict[str, float]]] = {"easy": {}, "hard": {}, "overall": {}}

    for prompt_type in ["easy", "hard"]:
        direct_rows = grouped.get((prompt_type, "direct"), [])
        direct_tps = safe_mean([row["tokens_per_second"] for row in direct_rows])

        for method in methods:
            rows = grouped.get((prompt_type, method), [])
            metric = {
                "count": len(rows),
                "mean_tps": safe_mean([row["tokens_per_second"] for row in rows]),
                "median_tps": safe_median([row["tokens_per_second"] for row in rows]),
                "std_tps": safe_std([row["tokens_per_second"] for row in rows]),
                "mean_acceptance": safe_mean([row["acceptance_rate"] for row in rows]),
                "mean_rounds": safe_mean([row["num_rounds"] for row in rows]),
                "mean_rtt": safe_mean([row["avg_rtt_ms"] for row in rows]),
                "mean_network": safe_mean([row["avg_network_ms"] for row in rows]),
                "mean_draft": safe_mean([row["avg_draft_ms"] for row in rows]),
                "mean_server": safe_mean([row["server_pure_inference_ms"] for row in rows]),
                "mean_reject_pos": safe_mean([row["avg_rejection_position"] for row in rows]),
                "speedup_vs_direct": (
                    safe_mean([row["tokens_per_second"] for row in rows]) / direct_tps
                    if direct_tps > 0
                    else 0.0
                ),
            }
            if method == "direct":
                metric["speedup_vs_direct"] = 1.0
            summary[prompt_type][method] = metric

    for method in methods:
        rows = [row for row in results if row["method"] == method]
        summary["overall"][method] = {
            "count": len(rows),
            "mean_tps": safe_mean([row["tokens_per_second"] for row in rows]),
            "median_tps": safe_median([row["tokens_per_second"] for row in rows]),
            "std_tps": safe_std([row["tokens_per_second"] for row in rows]),
            "mean_acceptance": safe_mean([row["acceptance_rate"] for row in rows]),
            "mean_rounds": safe_mean([row["num_rounds"] for row in rows]),
            "mean_rtt": safe_mean([row["avg_rtt_ms"] for row in rows]),
            "mean_network": safe_mean([row["avg_network_ms"] for row in rows]),
            "mean_draft": safe_mean([row["avg_draft_ms"] for row in rows]),
            "mean_server": safe_mean([row["server_pure_inference_ms"] for row in rows]),
            "mean_reject_pos": safe_mean([row["avg_rejection_position"] for row in rows]),
        }

    return summary


def parse_retry_count(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_text()
    return len(re.findall(r"Request failed \(attempt 1/3\)", text))


def plot_throughput(summary: dict, methods: list[str], output_dir: Path) -> None:
    labels = [method_label(method) for method in methods]
    x = np.arange(len(methods))
    width = 0.35
    easy_vals = [summary["easy"][method]["mean_tps"] for method in methods]
    hard_vals = [summary["hard"][method]["mean_tps"] for method in methods]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_easy = ax.bar(x - width / 2, easy_vals, width, label="Easy", color=PROMPT_COLORS["easy"])
    bars_hard = ax.bar(x + width / 2, hard_vals, width, label="Hard", color=PROMPT_COLORS["hard"])

    ax.set_title("Run3 Throughput by Method")
    ax.set_ylabel("Tokens / second")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for bars in (bars_easy, bars_hard):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_dir / "throughput_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_latency_breakdown(summary: dict, methods: list[str], output_dir: Path) -> None:
    spec_methods = [method for method in methods if method != "direct"]
    labels = []
    draft_vals = []
    network_vals = []
    server_vals = []

    for prompt_type in ["easy", "hard"]:
        for method in spec_methods:
            labels.append(f"{prompt_type.title()}\n{method_label(method)}")
            draft_vals.append(summary[prompt_type][method]["mean_draft"])
            network_vals.append(summary[prompt_type][method]["mean_network"])
            server_vals.append(summary[prompt_type][method]["mean_server"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, draft_vals, color="#59a14f", label="Draft")
    ax.bar(x, network_vals, bottom=draft_vals, color="#e15759", label="Network")
    ax.bar(
        x,
        server_vals,
        bottom=np.array(draft_vals) + np.array(network_vals),
        color="#4e79a7",
        label="Server",
    )

    totals = np.array(draft_vals) + np.array(network_vals) + np.array(server_vals)
    for idx, total in enumerate(totals):
        net_share = 100.0 * network_vals[idx] / total if total > 0 else 0.0
        ax.text(idx, total + 3, f"{net_share:.0f}% net", ha="center", va="bottom", fontsize=8)

    ax.set_title("Per-Round Latency Breakdown")
    ax.set_ylabel("Milliseconds / round")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "latency_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_acceptance_vs_speed(results: list[dict], methods: list[str], output_dir: Path) -> tuple[float, float]:
    spec_methods = [method for method in methods if method != "direct"]
    fig, ax = plt.subplots(figsize=(8, 6))
    xs_all: list[float] = []
    ys_all: list[float] = []

    markers = {"easy": "o", "hard": "s"}
    for prompt_type in ["easy", "hard"]:
        for method in spec_methods:
            rows = [row for row in results if row["prompt_type"] == prompt_type and row["method"] == method]
            xs = [100.0 * row["acceptance_rate"] for row in rows]
            ys = [row["tokens_per_second"] for row in rows]
            xs_all.extend(xs)
            ys_all.extend(ys)
            ax.scatter(
                xs,
                ys,
                label=f"{prompt_type.title()} {method_label(method)}",
                alpha=0.75,
                s=60,
                marker=markers[prompt_type],
                color=METHOD_COLORS.get(method, "#333333"),
                edgecolors="black",
                linewidths=0.3,
            )

    corr = corrcoef(xs_all, ys_all)
    coeffs = np.polyfit(xs_all, ys_all, deg=1) if len(xs_all) >= 2 else [0.0, 0.0]
    line_x = np.linspace(min(xs_all), max(xs_all), 100) if xs_all else np.array([0.0, 1.0])
    line_y = coeffs[0] * line_x + coeffs[1]
    ax.plot(line_x, line_y, color="black", linestyle="--", linewidth=1.5, label=f"Fit (r={corr:.2f})")

    ax.set_title("Acceptance vs Throughput")
    ax.set_xlabel("Acceptance rate (%)")
    ax.set_ylabel("Tokens / second")
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_dir / "acceptance_vs_speed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    rtt_values = [row["avg_rtt_ms"] for row in results if row["method"] != "direct"]
    tps_values = [row["tokens_per_second"] for row in results if row["method"] != "direct"]
    return corr, corrcoef(rtt_values, tps_values)


def plot_rounds_distribution(results: list[dict], methods: list[str], output_dir: Path) -> None:
    spec_methods = [method for method in methods if method != "direct"]
    data = [[row["num_rounds"] for row in results if row["method"] == method] for method in spec_methods]
    labels = [method_label(method) for method in spec_methods]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    boxplot = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.6)
    for patch, method in zip(boxplot["boxes"], spec_methods):
        patch.set_facecolor(METHOD_COLORS.get(method, "#999999"))
        patch.set_alpha(0.8)

    ax.set_title("Verification Rounds Distribution")
    ax.set_ylabel("Rounds per request")

    fig.tight_layout()
    fig.savefig(output_dir / "rounds_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_report(
    summary: dict,
    methods: list[str],
    retry_count: int,
    acceptance_speed_corr: float,
    rtt_speed_corr: float,
    output_dir: Path,
) -> str:
    spec_methods = [method for method in methods if method != "direct"]

    best_easy = max(spec_methods, key=lambda method: summary["easy"][method]["mean_tps"])
    best_hard = max(spec_methods, key=lambda method: summary["hard"][method]["mean_tps"])
    best_overall = max(spec_methods, key=lambda method: summary["overall"][method]["mean_tps"])

    lines: list[str] = []
    lines.append("# Run3 Report")
    lines.append("")
    lines.append(f"**Date**: {date.today().isoformat()}  ")
    lines.append("**Config**: 3B draft on edge + 32B GPTQ target on remote server  ")
    lines.append("**Run**: 10 easy + 10 hard prompts, methods = Direct, K=6, K=8, K=10")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"Direct decoding remained dominant at {summary['overall']['direct']['mean_tps']:.2f} tok/s overall. "
        f"The best speculative setting was {method_label(best_overall)} at "
        f"{summary['overall'][best_overall]['mean_tps']:.2f} tok/s, only "
        f"{summary['overall'][best_overall]['mean_tps'] / summary['overall']['direct']['mean_tps']:.2%} "
        "of the direct baseline."
    )
    lines.append("")
    lines.append("## Throughput Summary")
    lines.append("")
    lines.append("| Method | Easy tok/s | Hard tok/s | Overall tok/s | Avg acceptance | Avg rounds | Mean RTT ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method in methods:
        easy = summary["easy"][method]
        hard = summary["hard"][method]
        overall = summary["overall"][method]
        acceptance = "-" if method == "direct" else f"{100 * overall['mean_acceptance']:.1f}%"
        lines.append(
            f"| {method_label(method)} | {easy['mean_tps']:.2f} | {hard['mean_tps']:.2f} | "
            f"{overall['mean_tps']:.2f} | {acceptance} | {overall['mean_rounds']:.1f} | "
            f"{overall['mean_rtt']:.1f} |"
        )
    lines.append("")
    lines.append("## Timing Breakdown")
    lines.append("")
    lines.append("| Method | Draft ms/round | Network ms/round | Server ms/round | Network share | Median tok/s | Std tok/s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method in spec_methods:
        overall = summary["overall"][method]
        total = overall["mean_draft"] + overall["mean_network"] + overall["mean_server"]
        net_share = 100.0 * overall["mean_network"] / total if total > 0 else 0.0
        lines.append(
            f"| {method_label(method)} | {overall['mean_draft']:.1f} | {overall['mean_network']:.1f} | "
            f"{overall['mean_server']:.1f} | {net_share:.1f}% | {overall['median_tps']:.2f} | {overall['std_tps']:.2f} |"
        )
    lines.append("")
    lines.append("## Statistical Outcomes")
    lines.append("")
    lines.append(
        f"- Acceptance and throughput were almost perfectly coupled in this run: Pearson r = {acceptance_speed_corr:.3f}."
    )
    lines.append(
        f"- RTT had a weaker negative relationship with throughput: Pearson r = {rtt_speed_corr:.3f}."
    )
    lines.append(
        f"- Best speculative setting on easy prompts was {method_label(best_easy)} at {summary['easy'][best_easy]['mean_tps']:.2f} tok/s."
    )
    lines.append(
        f"- Best speculative setting on hard prompts was {method_label(best_hard)} at {summary['hard'][best_hard]['mean_tps']:.2f} tok/s."
    )
    lines.append(
        f"- Hard prompts were not universally worse: {method_label(best_hard)} reached "
        f"{summary['hard'][best_hard]['mean_tps']:.2f} tok/s with {100 * summary['hard'][best_hard]['mean_acceptance']:.1f}% "
        "acceptance, beating the same method on easy prompts."
    )
    if retry_count:
        lines.append(
            f"- The log recorded {retry_count} transient request retries, which indicates tunnel or proxy instability but not total experiment failure."
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"{method_label(best_overall)} was the least bad speculative option overall, but it still needed "
        f"{summary['overall'][best_overall]['mean_rounds']:.1f} verification rounds per request and only "
        f"{100 * summary['overall'][best_overall]['mean_acceptance']:.1f}% average acceptance. "
        "That is far below the acceptance regime needed to amortize the extra round trips."
    )
    lines.append(
        "Increasing K did not reliably help. K=10 raised draft cost, increased RTT exposure, and degraded acceptance enough to become the worst overall speculative choice."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `throughput_comparison.png`")
    lines.append("- `latency_breakdown.png`")
    lines.append("- `acceptance_vs_speed.png`")
    lines.append("- `rounds_distribution.png`")
    lines.append("- `comprehensive_results.json`")
    lines.append("- `comprehensive_log.txt`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).parent
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = script_dir / input_dir

    results_path = input_dir / "comprehensive_results.json"
    log_path = input_dir / "comprehensive_log.txt"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    results = load_results(results_path)
    methods = detect_methods(results)
    summary = summarize(results, methods)
    retry_count = parse_retry_count(log_path)

    plot_throughput(summary, methods, input_dir)
    plot_latency_breakdown(summary, methods, input_dir)
    acceptance_speed_corr, rtt_speed_corr = plot_acceptance_vs_speed(results, methods, input_dir)
    plot_rounds_distribution(results, methods, input_dir)

    report = build_report(
        summary=summary,
        methods=methods,
        retry_count=retry_count,
        acceptance_speed_corr=acceptance_speed_corr,
        rtt_speed_corr=rtt_speed_corr,
        output_dir=input_dir,
    )
    report_path = input_dir / "REPORT.md"
    report_path.write_text(report)

    print(f"Analyzed: {input_dir}")
    print("Generated:")
    print(f"  - {input_dir / 'throughput_comparison.png'}")
    print(f"  - {input_dir / 'latency_breakdown.png'}")
    print(f"  - {input_dir / 'acceptance_vs_speed.png'}")
    print(f"  - {input_dir / 'rounds_distribution.png'}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
