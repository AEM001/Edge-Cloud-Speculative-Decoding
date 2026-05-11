#!/usr/bin/env python3
"""Explore acceptance-length dynamics from raw quick-test round records."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/picospec_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/picospec_cache")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "outputs_quick" / "analysis"
ROUND_DETAILS = ANALYSIS_DIR / "raw_round_details.csv"
OUT_DIR = ANALYSIS_DIR / "acceptance_dynamics"

METHOD_ORDER = ["sync_k8", "tree_k8_b3"]
METHOD_LABELS = {"sync_k8": "Sync K=8", "tree_k8_b3": "Tree K=8 B=3"}
STATE_ORDER = ["dead", "middle", "full"]
STATE_LABELS = {"dead": "Dead 0-2", "middle": "Middle 3-7", "full": "Full 8"}
COLORS = {"dead": "#D62728", "middle": "#F2A93B", "full": "#54A24B"}


def state_for_acceptance(accepted: float) -> str:
    if accepted <= 2:
        return "dead"
    if accepted >= 8:
        return "full"
    return "middle"


def load_rounds() -> pd.DataFrame:
    rounds = pd.read_csv(ROUND_DETAILS)
    spec = rounds[rounds["method"].isin(METHOD_ORDER)].copy()
    spec = spec.dropna(subset=["accepted", "verify_prefix_len", "detail_index"])
    spec["accepted"] = spec["accepted"].astype(int)
    spec["round_index"] = spec["detail_index"].astype(int)
    spec["state"] = spec["accepted"].map(state_for_acceptance)
    spec["run_key"] = (
        spec["network"]
        + "/"
        + spec["method"]
        + "/"
        + spec["prompt_type"]
        + "/"
        + spec["prompt_id"].astype(str)
    )
    spec = spec.sort_values(["network", "method", "prompt_type", "prompt_id", "round_index"])
    return spec


def add_lags(spec: pd.DataFrame) -> pd.DataFrame:
    grouped = spec.groupby(["network", "method", "prompt_type", "prompt_id"], observed=True)
    spec["prev_accepted"] = grouped["accepted"].shift(1)
    spec["prev_state"] = grouped["state"].shift(1)
    spec["next_state"] = grouped["state"].shift(-1)
    spec["next_accepted"] = grouped["accepted"].shift(-1)
    return spec


def write_tables(spec: pd.DataFrame) -> None:
    state_share = (
        spec.groupby(["method", "prompt_type", "state"], observed=True)
        .size()
        .reset_index(name="rounds")
    )
    state_share["share"] = state_share["rounds"] / state_share.groupby(
        ["method", "prompt_type"], observed=True
    )["rounds"].transform("sum")
    state_share.to_csv(OUT_DIR / "table_state_share_by_prompt_type.csv", index=False)

    transitions = spec.dropna(subset=["prev_state"]).copy()
    trans = (
        transitions.groupby(["method", "prompt_type", "prev_state", "state"], observed=True)
        .size()
        .reset_index(name="rounds")
    )
    trans["probability"] = trans["rounds"] / trans.groupby(
        ["method", "prompt_type", "prev_state"], observed=True
    )["rounds"].transform("sum")
    trans.to_csv(OUT_DIR / "table_state_transitions.csv", index=False)

    predict = (
        transitions.groupby(["method", "prev_state"], observed=True)
        .agg(
            rounds=("state", "size"),
            p_dead_next=("state", lambda s: (s == "dead").mean()),
            p_middle_next=("state", lambda s: (s == "middle").mean()),
            p_full_next=("state", lambda s: (s == "full").mean()),
            avg_next_accept=("accepted", "mean"),
        )
        .reset_index()
    )
    predict.to_csv(OUT_DIR / "table_dead_zone_predictability.csv", index=False)

    base_dead = (
        spec.groupby(["method", "prompt_type"], observed=True)
        .agg(base_dead_share=("state", lambda s: (s == "dead").mean()))
        .reset_index()
    )
    dead_next = (
        transitions[transitions["prev_state"] == "dead"]
        .groupby(["method", "prompt_type"], observed=True)
        .agg(
            after_dead_rounds=("state", "size"),
            dead_after_dead_share=("state", lambda s: (s == "dead").mean()),
            full_after_dead_share=("state", lambda s: (s == "full").mean()),
            middle_after_dead_share=("state", lambda s: (s == "middle").mean()),
        )
        .reset_index()
    )
    clustering = dead_next.merge(base_dead, on=["method", "prompt_type"], how="left")
    clustering["dead_clustering_lift"] = (
        clustering["dead_after_dead_share"] / clustering["base_dead_share"]
    )
    clustering.to_csv(OUT_DIR / "table_dead_zone_clustering.csv", index=False)

    run_rows = []
    for key, group in spec.groupby(["network", "method", "prompt_type", "prompt_id"], observed=True):
        prev = None
        length = 0
        for state in group["state"]:
            if state == prev:
                length += 1
            else:
                if prev is not None:
                    run_rows.append((*key, prev, length))
                prev = state
                length = 1
        if prev is not None:
            run_rows.append((*key, prev, length))
    runs = pd.DataFrame(
        run_rows, columns=["network", "method", "prompt_type", "prompt_id", "state", "run_length"]
    )
    runs.to_csv(OUT_DIR / "table_state_run_lengths.csv", index=False)

    position = (
        spec.groupby(["method", "prompt_type", "round_index"], observed=True)
        .agg(
            rounds=("accepted", "size"),
            avg_accept=("accepted", "mean"),
            dead_share=("state", lambda s: (s == "dead").mean()),
            middle_share=("state", lambda s: (s == "middle").mean()),
            full_share=("state", lambda s: (s == "full").mean()),
            avg_prefix_len=("verify_prefix_len", "mean"),
        )
        .reset_index()
    )
    position.to_csv(OUT_DIR / "table_acceptance_by_round_position.csv", index=False)

    prefix_bins = spec.copy()
    prefix_bins["prefix_bin"] = pd.cut(
        prefix_bins["verify_prefix_len"],
        bins=[0, 100, 150, 200, 250, 10_000],
        labels=["<=100", "101-150", "151-200", "201-250", ">250"],
        include_lowest=True,
    )
    prefix_summary = (
        prefix_bins.groupby(["method", "prompt_type", "prefix_bin"], observed=True)
        .agg(
            rounds=("accepted", "size"),
            avg_accept=("accepted", "mean"),
            dead_share=("state", lambda s: (s == "dead").mean()),
            middle_share=("state", lambda s: (s == "middle").mean()),
            full_share=("state", lambda s: (s == "full").mean()),
        )
        .reset_index()
    )
    prefix_summary.to_csv(OUT_DIR / "table_acceptance_by_prefix_bin.csv", index=False)

    middle = spec[spec["state"] == "middle"].copy()
    middle_summary = (
        middle.groupby(["method", "prompt_type", "accepted"], observed=True)
        .size()
        .reset_index(name="rounds")
    )
    middle_summary["share_within_middle"] = middle_summary["rounds"] / middle_summary.groupby(
        ["method", "prompt_type"], observed=True
    )["rounds"].transform("sum")
    middle_summary.to_csv(OUT_DIR / "table_middle_accept_lengths.csv", index=False)


def plot_state_timeline(spec: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.0), sharex=True, sharey=True)
    for ax, (method, prompt_type) in zip(axes.ravel(), [(m, p) for p in ["gsm8k", "humaneval"] for m in METHOD_ORDER]):
        sub = spec[(spec["method"] == method) & (spec["prompt_type"] == prompt_type)]
        pivot = sub.pivot_table(
            index=["network", "prompt_id"],
            columns="round_index",
            values="accepted",
            aggfunc="mean",
        )
        ax.imshow(pivot.fillna(-1), aspect="auto", cmap="RdYlGn", vmin=0, vmax=8)
        ax.set_title(f"{METHOD_LABELS[method]} / {prompt_type}")
        ax.set_ylabel("network/prompt")
        ax.set_xlabel("round index")
    fig.suptitle("Acceptance Length Timelines")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "chart_acceptance_timeline.png", dpi=180)
    plt.close(fig)


def plot_transition_heatmaps(spec: pd.DataFrame) -> None:
    transitions = spec.dropna(subset=["prev_state"]).copy()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), sharex=True, sharey=True)
    for ax, (method, prompt_type) in zip(axes.ravel(), [(m, p) for p in ["gsm8k", "humaneval"] for m in METHOD_ORDER]):
        sub = transitions[(transitions["method"] == method) & (transitions["prompt_type"] == prompt_type)]
        table = pd.crosstab(sub["prev_state"], sub["state"], normalize="index")
        table = table.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
        im = ax.imshow(table.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{METHOD_LABELS[method]} / {prompt_type}")
        ax.set_xticks(range(3), [STATE_LABELS[s] for s in STATE_ORDER], rotation=30, ha="right")
        ax.set_yticks(range(3), [STATE_LABELS[s] for s in STATE_ORDER])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{table.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.suptitle("Acceptance-State Transition Probabilities")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "chart_state_transitions.png", dpi=180)
    plt.close(fig)


def plot_position_effect(spec: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for ax, prompt_type in zip(axes, ["gsm8k", "humaneval"]):
        for method in METHOD_ORDER:
            sub = spec[(spec["method"] == method) & (spec["prompt_type"] == prompt_type)]
            summary = sub.groupby("round_index", observed=True)["accepted"].mean()
            ax.plot(summary.index, summary.values, marker="o", markersize=3, label=METHOD_LABELS[method])
        ax.set_title(prompt_type)
        ax.set_xlabel("round index")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean accepted draft tokens")
    axes[1].legend(frameon=False)
    fig.suptitle("Acceptance Drift Within a Completion")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "chart_acceptance_by_round_position.png", dpi=180)
    plt.close(fig)


def plot_prefix_bins(spec: pd.DataFrame) -> None:
    prefix_bins = spec.copy()
    prefix_bins["prefix_bin"] = pd.cut(
        prefix_bins["verify_prefix_len"],
        bins=[0, 100, 150, 200, 250, 10_000],
        labels=["<=100", "101-150", "151-200", "201-250", ">250"],
        include_lowest=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for ax, prompt_type in zip(axes, ["gsm8k", "humaneval"]):
        sub = prefix_bins[prefix_bins["prompt_type"] == prompt_type]
        summary = (
            sub.groupby(["prefix_bin", "state"], observed=True)
            .size()
            .unstack("state", fill_value=0)
            .reindex(columns=STATE_ORDER, fill_value=0)
        )
        summary = summary.div(summary.sum(axis=1), axis=0)
        summary.rename(columns=STATE_LABELS).plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[COLORS[s] for s in STATE_ORDER],
            width=0.78,
        )
        ax.set_title(prompt_type)
        ax.set_xlabel("verify prefix length")
        ax.grid(axis="y", alpha=0.25)
        ax.legend().remove()
    axes[0].set_ylabel("Share of rounds")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Acceptance State by Prefix Length")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "chart_acceptance_by_prefix_length.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = add_lags(load_rounds())
    spec.to_csv(OUT_DIR / "raw_acceptance_rounds_with_lags.csv", index=False)
    write_tables(spec)
    plot_state_timeline(spec)
    plot_transition_heatmaps(spec)
    plot_position_effect(spec)
    plot_prefix_bins(spec)
    print(f"Wrote acceptance-dynamics outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
