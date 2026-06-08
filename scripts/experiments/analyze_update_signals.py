"""Correlation analysis for observable stale-guidance signals.

Input is the per-round CSV written by analyze_guidance_reuse.py. The script
uses only metrics already available at the edge: recent accepted length,
staleness, selected-set change, and pipeline state. If reuse_R1 is present,
oracle_overlap is also included as a diagnostic label, not as an edge-side
signal.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from guidance_common import pearson


def _float(value: object) -> Optional[float]:
    if value in (None, "", "None", "n/a"):
        return None
    return float(value)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _group_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["prompt_type"], row["prompt_id"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["round"]))
    return grouped


def _feature_rows(rows: List[Dict[str, str]], window: int) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    grouped = _group_rows(rows)
    for group in grouped.values():
        accepted = [float(row["accepted_len"]) for row in group]
        overlaps = [_float(row.get("oracle_overlap")) for row in group]
        for idx, row in enumerate(group):
            start = max(0, idx - window + 1)
            recent = accepted[start : idx + 1]
            previous = accepted[max(0, start - window) : start]
            recent_accept = sum(recent) / len(recent)
            prev_accept = sum(previous) / len(previous) if previous else recent_accept
            accept_drop = prev_accept - recent_accept
            next_accept = accepted[idx + 1] if idx + 1 < len(accepted) else accepted[idx]
            next_accept_drop = accepted[idx] - next_accept
            overlap = overlaps[idx]
            stale_label = 1.0 if overlap is not None and overlap < 0.6 else 0.0
            out.append(
                {
                    "staleness_rounds": float(row["staleness_rounds"]),
                    "recent_acceptance_len": recent_accept,
                    "accepted_len_drop": accept_drop,
                    "pipeline_wait_ms": float(row["pipeline_wait_ms"]),
                    "pipeline_hit": float(row["pipeline_hit"]),
                    "pipeline_reuse": float(row["pipeline_reuse"]),
                    "next_accept_len_drop": next_accept_drop,
                    "oracle_overlap": overlap if overlap is not None else 1.0,
                    "stale_oracle_label": stale_label,
                }
            )
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["metric"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze update-trigger signal correlations.")
    parser.add_argument("round_csv", type=Path)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = _read_rows(args.round_csv)
    feature_rows = _feature_rows(rows, args.window)
    feature_names = [
        "staleness_rounds",
        "recent_acceptance_len",
        "accepted_len_drop",
        "pipeline_wait_ms",
        "pipeline_hit",
        "pipeline_reuse",
    ]
    labels = ["next_accept_len_drop", "oracle_overlap", "stale_oracle_label"]

    corr_rows: List[Dict[str, object]] = []
    for feature in feature_names:
        xs = [row[feature] for row in feature_rows]
        for label in labels:
            ys = [row[label] for row in feature_rows]
            corr = pearson(xs, ys)
            corr_rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "pearson": "" if corr is None else f"{corr:.6f}",
                    "samples": len(xs),
                    "window": args.window,
                }
            )

    output = args.output or args.round_csv.with_name("update_signal_correlations.csv")
    _write_csv(output, corr_rows)
    print(f"CORRELATIONS_CSV={output}")


if __name__ == "__main__":
    main()

