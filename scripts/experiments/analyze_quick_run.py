#!/usr/bin/env python3
"""Build summaries and round-detail files for quick_test.py results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs_quick"
DEFAULT_RESULTS_PATH = OUTPUT_DIR / "quick_test_results.json"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "quick_test_summary.json"
DEFAULT_ROUNDS_PATH = OUTPUT_DIR / "quick_test_rounds.jsonl"
SUPPORTED_METHOD_FAMILIES = {"direct", "tree_async"}


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def acceptance_length(row: Dict[str, Any]) -> float:
    speculative = row.get("speculative", {})
    if "acceptance_length" in speculative:
        return speculative["acceptance_length"]
    if "accepted_draft_per_round" in speculative:
        return speculative["accepted_draft_per_round"]
    rounds = speculative.get("rounds") or 0
    accepted = speculative.get("accepted_draft_tokens") or 0
    return accepted / rounds if rounds else 0.0


def load_result_file(path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "results" in payload:
        config = payload.get("config", {})
        results = payload["results"]
    else:
        config = {}
        results = payload
    return config, [
        row for row in results
        if row.get("method_family") in SUPPORTED_METHOD_FAMILIES
    ]


def method_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    direct_tps = avg(row["output"]["tokens_per_second"] for row in rows if row["method"] == "direct")
    methods = sorted({row["method"] for row in rows})
    summary: Dict[str, Any] = {}

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        tps = avg(row["output"]["tokens_per_second"] for row in method_rows)
        summary[method] = {
            "runs": len(method_rows),
            "tokens_per_second": tps,
            "speedup_vs_direct": tps / direct_tps if direct_tps and method != "direct" else 1.0,
            "total_time_ms": avg(row["output"]["total_time_ms"] for row in method_rows),
            "tokens_generated": avg(row["output"]["tokens_generated"] for row in method_rows),
            "simulated_network_ms": avg(row["timing"]["simulated_network_ms"] for row in method_rows),
        }
        if method != "direct":
            summary[method].update(
                {
                    "rounds": avg(row["speculative"]["rounds"] for row in method_rows),
                    "acceptance_length": avg(acceptance_length(row) for row in method_rows),
                    "branch_reused": avg(float(row["async_detail"]["branch_reused"]) for row in method_rows),
                    "reused_tokens": avg(row["async_detail"]["reused_tokens"] for row in method_rows),
                    "predraft_window_ms": avg(row["async_detail"]["predraft_window_ms"] for row in method_rows),
                    "reuse_prep_time_ms": avg(row["async_detail"]["reuse_prep_time_ms"] for row in method_rows),
                }
            )
    return summary


def build_summary(config: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "config": config,
        "overall": method_summary(results),
        "by_prompt_type": {},
    }

    for prompt_type in sorted({row["prompt_type"] for row in results}):
        rows = [row for row in results if row["prompt_type"] == prompt_type]
        summary["by_prompt_type"][prompt_type] = method_summary(rows)

    return summary


def write_round_details(results: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        for result in results:
            base = {
                "method": result["method"],
                "method_family": result["method_family"],
                "network": result["network"],
                "prompt_type": result["prompt_type"],
                "prompt_id": result["prompt_id"],
            }
            details = (
                result["raw"].get("round_details")
                or result["raw"].get("slot_details")
                or [result["raw"].get("direct_timing", {})]
            )
            for index, detail in enumerate(details):
                f.write(json.dumps({**base, "detail_index": index, "detail": detail}) + "\n")


def print_summary(summary: Dict[str, Any]) -> None:
    overall = summary["overall"]
    direct = overall.get("direct", {})
    direct_tps = direct.get("tokens_per_second", 0.0)

    print("Quick Test Summary")
    print("==================")
    for method, row in overall.items():
        if method == "direct":
            print(f"{method:16} {row['tokens_per_second']:8.2f} tok/s")
        else:
            print(
                f"{method:16} {row['tokens_per_second']:8.2f} tok/s  "
                f"{row['speedup_vs_direct']:.3f}x vs direct ({direct_tps:.2f} tok/s)"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze quick_test.py output.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Path to quick_test_results.json.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH, help="Path for quick_test_summary.json.")
    parser.add_argument("--rounds", type=Path, default=DEFAULT_ROUNDS_PATH, help="Path for quick_test_rounds.jsonl.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, results = load_result_file(args.results)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.rounds.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(config, results)
    args.summary.write_text(json.dumps(summary, indent=2))
    write_round_details(results, args.rounds)
    print_summary(summary)
    print(f"\nSummary saved to: {args.summary}")
    print(f"Round details saved to: {args.rounds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
