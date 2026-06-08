"""Analyze the most recent (or a specified) quick_test results file.

Usage
-----
  python scripts/experiments/analyze_quick_run.py                   # latest file in outputs/
  python scripts/experiments/analyze_quick_run.py path/to/file.json # explicit file
  RESULTS_PATH=path/to/file.json python scripts/experiments/analyze_quick_run.py

Output
------
Prints a compact markdown-style summary table to stdout and writes a
companion ``<stem>_summary.txt`` alongside the results JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_latest_results() -> Optional[Path]:
    candidates = sorted(OUTPUTS_DIR.glob("quick_test_results_*.json"))
    return candidates[-1] if candidates else None


def load_results(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _ms(v: float) -> str:
    return f"{v:8.1f} ms"


def _tps(v: float) -> str:
    return f"{v:6.2f} tok/s"


def _pct(num: float, den: float) -> str:
    if den <= 0:
        return "   n/a"
    return f"{100.0 * num / den:5.1f}%"


def _col(s: str, width: int) -> str:
    return s[:width].ljust(width)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _extract_row(result: Dict[str, Any]) -> Dict[str, Any]:
    out = result.get("output", {})
    timing = result.get("timing", {})
    spec = result.get("speculative", {})
    return {
        "method": result.get("method", "?"),
        "prompt_type": result.get("prompt_type", "?"),
        "prompt_id": result.get("prompt_id", 0),
        "tokens": int(out.get("tokens_generated", 0)),
        "wall_ms": float(out.get("total_time_ms", 0.0)),
        "tps": float(out.get("tokens_per_second", 0.0)),
        "draft_ms": float(timing.get("local_draft_ms", 0.0)),
        "server_ms": float(timing.get("server_model_ms", 0.0)),
        "net_ms": float(timing.get("simulated_network_ms", 0.0)),
        "avg_rtt_ms": float(timing.get("avg_rtt_ms", 0.0)),
        "rounds": int(spec.get("rounds", 0)),
        "accepted": int(spec.get("accepted_draft_tokens", 0)),
        "gen_per_round": float(spec.get("generated_per_round", 0.0)),
        "acceptance_len": float(spec.get("acceptance_length", 0.0)),
    }


def _time_breakdown(row: Dict[str, Any]) -> str:
    draft = row["draft_ms"]
    server = row["server_ms"]
    net = row["net_ms"]
    wall = row["wall_ms"]
    other = max(0.0, wall - draft - server - net)
    return (
        f"draft={draft:.0f} server={server:.0f} net={net:.0f} other={other:.0f}"
    )


def analyze(data: Dict[str, Any]) -> str:
    config = data.get("config", {})
    results = data.get("results", [])

    rows = [_extract_row(r) for r in results]
    direct_rows = [r for r in rows if r["method"] == "direct"]
    spec_rows = [r for r in rows if r["method"] != "direct"]

    lines: List[str] = []

    # Header
    lines.append("=" * 72)
    lines.append("  Quick Benchmark Summary")
    lines.append("=" * 72)
    lines.append(f"  draft_mode  : {config.get('draft_mode','?')}  "
                 f"nodes={config.get('draft_tree_nodes','?')}  "
                 f"depth={config.get('draft_tree_max_depth','?')}")
    lines.append(f"  dataset     : {config.get('dataset_split','?')}  "
                 f"input_tokens={config.get('prompt_input_tokens','?')}  "
                 f"max_tokens={config.get('max_tokens','?')}")
    lines.append(f"  prompts     : {config.get('prompt_count','?')}  "
                 f"types={config.get('prompt_types','?')}")
    lines.append(f"  retrieval   : chunk_size={config.get('retrieval_chunk_size','?')}  "
                 f"top_k={config.get('retrieve_top_k','?')}  "
                 f"every_n={config.get('retrieve_every_n_steps','?')}")
    lines.append("")

    # Per-method table
    col_w = [24, 7, 10, 10, 6, 7, 6]
    header = (
        _col("method", col_w[0])
        + _col("tokens", col_w[1])
        + _col("wall_ms", col_w[2])
        + _col("tok/s", col_w[3])
        + _col("rounds", col_w[4])
        + _col("acc/r", col_w[5])
        + _col("al", col_w[6])
    )
    sep = "-" * sum(col_w)
    lines.append(header)
    lines.append(sep)

    for row in rows:
        if row["method"] == "direct":
            extra = ""
        else:
            extra = f"  [{_time_breakdown(row)}]"
        line = (
            _col(row["method"], col_w[0])
            + _col(str(row["tokens"]), col_w[1])
            + _col(f"{row['wall_ms']:.1f}", col_w[2])
            + _col(f"{row['tps']:.2f}", col_w[3])
            + _col(str(row["rounds"]) if row["rounds"] else "-", col_w[4])
            + _col(f"{row['gen_per_round']:.2f}" if row["rounds"] else "-", col_w[5])
            + _col(f"{row['acceptance_len']:.2f}" if row["rounds"] else "-", col_w[6])
            + extra
        )
        lines.append(line)

    lines.append("")

    # Speedup vs direct (matched by prompt)
    if direct_rows and spec_rows:
        lines.append("  Speedup vs direct (wall time):")
        for sr in spec_rows:
            matched = [d for d in direct_rows if d["prompt_id"] == sr["prompt_id"]]
            if not matched:
                matched = direct_rows[:1]
            dr = matched[0] if matched else None
            if dr and dr["wall_ms"] > 0:
                speedup = dr["wall_ms"] / sr["wall_ms"]
                lines.append(f"    {sr['method']}: {speedup:.3f}x  "
                              f"({dr['wall_ms']:.1f} ms -> {sr['wall_ms']:.1f} ms)")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a quick_test results JSON.")
    parser.add_argument(
        "results_file",
        nargs="?",
        type=Path,
        default=None,
        help="Path to results JSON. Defaults to the latest file in outputs/.",
    )
    args = parser.parse_args()

    import os
    env_path = os.environ.get("RESULTS_PATH")

    if args.results_file is not None:
        path = args.results_file
    elif env_path:
        path = Path(env_path)
    else:
        path = find_latest_results()

    if path is None:
        print("No results file found in outputs/ — run quick.sh first.", file=sys.stderr)
        sys.exit(1)

    if not path.exists():
        print(f"Results file not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = load_results(path)
    summary = analyze(data)

    print(summary)

    summary_path = path.with_name(path.stem + "_summary.txt")
    summary_path.write_text(summary)
    print(f"\n  Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
