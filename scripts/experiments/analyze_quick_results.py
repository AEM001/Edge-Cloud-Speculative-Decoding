#!/usr/bin/env python3
"""Analyze raw quick-test experiment outputs.

This script intentionally stops at data products: CSV tables and PNG charts.
The report is written manually after inspecting these outputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ecsd_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/ecsd_cache")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs_quick"
RESULTS_PATH = OUTPUT_DIR / "quick_test_results.json"
ROUNDS_PATH = OUTPUT_DIR / "quick_test_rounds.jsonl"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

METHOD_ORDER = ["direct", "sync_k8", "tree_k8_b3"]
METHOD_LABELS = {
    "direct": "Direct",
    "sync_k8": "Sync K=8",
    "tree_k8_b3": "Tree K=8 B=3",
}
COLORS = {
    "direct": "#4C78A8",
    "sync_k8": "#F58518",
    "tree_k8_b3": "#54A24B",
}


def load_results() -> pd.DataFrame:
    payload = json.loads(RESULTS_PATH.read_text())
    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    flat_rows = []
    for row in rows:
        speculative = row["speculative"]
        acceptance_length = speculative.get(
            "acceptance_length",
            speculative.get("accepted_draft_per_round", 0.0),
        )
        flat_rows.append(
            {
                "network": row["network"],
                "method": row["method"],
                "prompt_type": row["prompt_type"],
                "prompt_id": row["prompt_id"],
                "tokens": row["output"]["tokens_generated"],
                "total_ms": row["output"]["total_time_ms"],
                "tps": row["output"]["tokens_per_second"],
                "server_model_ms": row["timing"]["server_model_ms"],
                "http_rpc_ms": row["timing"]["http_rpc_ms"],
                "local_draft_ms": row["timing"]["local_draft_ms"],
                "sim_network_ms": row["timing"]["simulated_network_ms"],
                "avg_rtt_ms": row["timing"]["avg_rtt_ms"],
                "rounds": row["speculative"]["rounds"],
                "acceptance_length": acceptance_length,
                "generated_per_round": row["speculative"]["generated_per_round"],
                "branch_reused": row["async_detail"]["branch_reused"],
                "reused_tokens": row["async_detail"]["reused_tokens"],
                "predraft_window_ms": row["async_detail"]["predraft_window_ms"],
                "reuse_prep_time_ms": row["async_detail"]["reuse_prep_time_ms"],
            }
        )
    df = pd.DataFrame(flat_rows)
    df["method_label"] = df["method"].map(METHOD_LABELS)
    return df


def load_round_details() -> pd.DataFrame:
    rows = []
    with ROUNDS_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            detail = item["detail"]
            base = {
                "network": item["network"],
                "method": item["method"],
                "method_family": item["method_family"],
                "prompt_type": item["prompt_type"],
                "prompt_id": item["prompt_id"],
                "detail_index": item["detail_index"],
            }
            base.update(detail)
            rows.append(base)
    return pd.DataFrame(rows)


def add_speedups(df: pd.DataFrame) -> pd.DataFrame:
    direct = df[df["method"] == "direct"][
        ["network", "prompt_type", "prompt_id", "tps", "total_ms"]
    ].rename(columns={"tps": "direct_tps", "total_ms": "direct_total_ms"})
    out = df.merge(direct, on=["network", "prompt_type", "prompt_id"], how="left")
    out["speedup_vs_direct"] = out["tps"] / out["direct_tps"]
    out["ms_saved_vs_direct"] = out["direct_total_ms"] - out["total_ms"]
    return out


def write_tables(df: pd.DataFrame, rounds: pd.DataFrame) -> None:
    run_summary = (
        df.groupby(["network", "method"], observed=True)
        .agg(
            runs=("tps", "size"),
            tokens_per_second=("tps", "mean"),
            tps_std=("tps", "std"),
            total_ms=("total_ms", "mean"),
            total_ms_std=("total_ms", "std"),
            speedup_vs_direct=("speedup_vs_direct", "mean"),
            ms_saved_vs_direct=("ms_saved_vs_direct", "mean"),
            rounds=("rounds", "mean"),
            acceptance_length=("acceptance_length", "mean"),
            branch_reused=("branch_reused", "mean"),
            reused_tokens=("reused_tokens", "mean"),
            predraft_window_ms=("predraft_window_ms", "mean"),
            reuse_prep_time_ms=("reuse_prep_time_ms", "mean"),
        )
        .reset_index()
    )
    run_summary.to_csv(ANALYSIS_DIR / "table_run_summary.csv", index=False)

    prompt_summary = (
        df.groupby(["network", "prompt_type", "method"], observed=True)
        .agg(
            runs=("tps", "size"),
            tokens_per_second=("tps", "mean"),
            total_ms=("total_ms", "mean"),
            speedup_vs_direct=("speedup_vs_direct", "mean"),
            rounds=("rounds", "mean"),
            acceptance_length=("acceptance_length", "mean"),
            branch_reused=("branch_reused", "mean"),
            reused_tokens=("reused_tokens", "mean"),
        )
        .reset_index()
    )
    prompt_summary.to_csv(ANALYSIS_DIR / "table_prompt_type_summary.csv", index=False)

    if "accepted" in rounds:
        acceptance = (
            rounds[rounds["method"].isin(["sync_k8", "tree_k8_b3"])]
            .groupby(["network", "method", "accepted"], observed=True)
            .size()
            .reset_index(name="rounds")
        )
        acceptance["share"] = acceptance["rounds"] / acceptance.groupby(
            ["network", "method"], observed=True
        )["rounds"].transform("sum")
        acceptance.to_csv(ANALYSIS_DIR / "table_acceptance_distribution.csv", index=False)

    # Tree offsets table removed - slot_details no longer tracked in simplified metrics

    if "verify_input_len" in rounds:
        verify = rounds[rounds["method"].isin(["sync_k8", "tree_k8_b3"])].copy()
        verify["verify_ms"] = verify.get("server_time_ms", verify.get("verify_ms"))
        verify[["network", "method", "prompt_type", "prompt_id", "verify_input_len", "verify_ms", "rtt_ms"]].to_csv(
            ANALYSIS_DIR / "table_verify_rounds.csv", index=False
        )


def plot_tps(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    summary = (
        df.groupby(["network", "method"], observed=True)["tps"]
        .mean()
        .unstack("method")
        .reindex(columns=METHOD_ORDER)
    )
    summary.rename(columns=METHOD_LABELS).plot(
        kind="bar", ax=ax, color=[COLORS[m] for m in METHOD_ORDER], width=0.78
    )
    ax.set_ylabel("Tokens / second")
    ax.set_xlabel("Network")
    ax.set_title("Throughput by Method")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "chart_throughput_by_method.png", dpi=180)
    plt.close(fig)


def plot_speedup_by_prompt(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    subset = df[df["method"].isin(["sync_k8", "tree_k8_b3"])].copy()
    grouped = (
        subset.groupby(["network", "prompt_type", "method"], observed=True)["speedup_vs_direct"]
        .mean()
        .reset_index()
    )
    grouped["bucket"] = grouped["network"] + " / " + grouped["prompt_type"]
    pivot = grouped.pivot(index="bucket", columns="method", values="speedup_vs_direct")
    pivot = pivot[["sync_k8", "tree_k8_b3"]]
    pivot.rename(columns=METHOD_LABELS).plot(
        kind="bar", ax=ax, color=[COLORS["sync_k8"], COLORS["tree_k8_b3"]], width=0.75
    )
    ax.axhline(1.0, color="#333333", linewidth=1, linestyle="--")
    ax.set_ylabel("Speedup vs direct")
    ax.set_xlabel("")
    ax.set_title("Speedup by Prompt Type")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "chart_speedup_by_prompt_type.png", dpi=180)
    plt.close(fig)


def plot_acceptance(rounds: pd.DataFrame) -> None:
    spec = rounds[rounds["method"].isin(["sync_k8", "tree_k8_b3"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    for ax, network in zip(axes, sorted(spec["network"].unique())):
        sub = spec[spec["network"] == network]
        counts = (
            sub.groupby(["accepted", "method"], observed=True)
            .size()
            .unstack("method", fill_value=0)
            .reindex(range(0, 9), fill_value=0)
        )
        shares = counts.div(counts.sum(axis=0), axis=1)
        shares.rename(columns=METHOD_LABELS).plot(
            kind="bar",
            ax=ax,
            color=[COLORS.get(c, "#777777") for c in counts.columns],
            width=0.78,
        )
        ax.set_title(network)
        ax.set_xlabel("Accepted draft tokens")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Share of rounds")
    axes[1].legend(frameon=False)
    axes[0].legend().remove()
    fig.suptitle("Raw Round Acceptance Distribution")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ANALYSIS_DIR / "chart_acceptance_distribution.png", dpi=180)
    plt.close(fig)


def plot_verify_scaling(rounds: pd.DataFrame) -> None:
    spec = rounds[rounds["method"].isin(["sync_k8", "tree_k8_b3"])].copy()
    if "server_time_ms" in spec:
        spec["verify_ms"] = spec["server_time_ms"].fillna(spec.get("verify_ms"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method in ["sync_k8", "tree_k8_b3"]:
        sub = spec[spec["method"] == method]
        ax.scatter(
            sub["verify_input_len"],
            sub["verify_ms"],
            s=14,
            alpha=0.45,
            label=METHOD_LABELS[method],
            color=COLORS[method],
        )
    ax.set_xlabel("Verify input length (prefix + draft)")
    ax.set_ylabel("Verify model time (ms)")
    ax.set_title("Verifier Cost Grows With Prefix Length")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "chart_verify_scaling.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = add_speedups(load_results())
    rounds = load_round_details()
    df.to_csv(ANALYSIS_DIR / "raw_run_metrics.csv", index=False)
    rounds.to_csv(ANALYSIS_DIR / "raw_round_details.csv", index=False)
    write_tables(df, rounds)
    plot_tps(df)
    plot_speedup_by_prompt(df)
    plot_acceptance(rounds)
    plot_verify_scaling(rounds)
    print(f"Wrote analysis outputs to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
