"""Offline threshold-policy simulation from per-round reuse data.

This script estimates when a simple edge-side policy would request guidance:

  if mean(accepted_len over recent window) < threshold: update

It does not re-run the model. It is for choosing promising thresholds before
implementing the policy in the live client.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from guidance_common import parse_int_list


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


def _simulate_group(rows: List[Dict[str, str]], threshold: float, window: int, update_cost_ms: float) -> Dict[str, float]:
    accepted = [float(row["accepted_len"]) for row in rows]
    updates = 0
    for idx in range(len(rows)):
        start = max(0, idx - window + 1)
        recent = accepted[start : idx + 1]
        if sum(recent) / len(recent) < threshold:
            updates += 1
    accepted_total = sum(accepted)
    added_cost = updates * update_cost_ms
    return {
        "rounds": float(len(rows)),
        "accepted_tokens": accepted_total,
        "updates": float(updates),
        "update_cost_ms": added_cost,
        "accepted_per_update": accepted_total / updates if updates > 0 else accepted_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate simple adaptive cloud-guidance update thresholds.")
    parser.add_argument("round_csv", type=Path)
    parser.add_argument("--thresholds", default="1,2,3,4,5,6")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--update-cost-ms", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = _read_rows(args.round_csv)
    grouped = _group_rows(rows)
    thresholds = parse_int_list(args.thresholds)
    output_rows: List[Dict[str, object]] = []

    for threshold in thresholds:
        totals = {
            "rounds": 0.0,
            "accepted_tokens": 0.0,
            "updates": 0.0,
            "update_cost_ms": 0.0,
        }
        for group in grouped.values():
            simulated = _simulate_group(group, float(threshold), args.window, args.update_cost_ms)
            for key in totals:
                totals[key] += simulated[key]
        output_rows.append(
            {
                "threshold": threshold,
                "window": args.window,
                "update_cost_ms_each": args.update_cost_ms,
                "rounds": int(totals["rounds"]),
                "accepted_tokens": int(totals["accepted_tokens"]),
                "updates": int(totals["updates"]),
                "total_update_cost_ms": f"{totals['update_cost_ms']:.3f}",
                "accepted_per_update": ""
                if totals["updates"] <= 0
                else f"{totals['accepted_tokens'] / totals['updates']:.3f}",
            }
        )

    output = args.output or args.round_csv.with_name("adaptive_policy_simulation.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"ADAPTIVE_POLICY_CSV={output}")


if __name__ == "__main__":
    main()

