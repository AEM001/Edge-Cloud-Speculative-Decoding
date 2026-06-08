"""Summarize guidance reuse sweep results.

Writes:
  <run_dir>/analysis/guidance_reuse_summary.csv
  <run_dir>/analysis/guidance_reuse_rounds.csv
  <run_dir>/analysis/guidance_reuse_report.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from guidance_common import (
    acceptance_ratio,
    jaccard,
    latest_quick_result,
    load_json,
    mean,
    method_interval,
    prompt_key,
    result_profile,
    result_rounds,
)


def _find_oracle_rounds(results: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[int, List[int]]]:
    oracle: Dict[Tuple[str, int], Dict[int, List[int]]] = {}
    for result in results:
        if result.get("method") != "reuse_R1":
            continue
        by_round: Dict[int, List[int]] = {}
        for rd in result_rounds(result):
            by_round[int(rd.get("round", 0))] = [int(v) for v in rd.get("selected_chunk_ids") or []]
        oracle[prompt_key(result)] = by_round
    return oracle


def _round_rows(result: Dict[str, Any], oracle: Dict[Tuple[str, int], Dict[int, List[int]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    method = str(result.get("method", "?"))
    profile = result_profile(result)
    interval = method_interval(method, profile)
    key = prompt_key(result)
    oracle_for_prompt = oracle.get(key, {})
    last_update_round: Optional[int] = None

    for rd in result_rounds(result):
        round_idx = int(rd.get("round", 0))
        retrieval = bool(rd.get("retrieval", False))
        if retrieval:
            last_update_round = round_idx
        selected = [int(v) for v in rd.get("selected_chunk_ids") or []]
        oracle_selected = oracle_for_prompt.get(round_idx)
        rows.append(
            {
                "method": method,
                "prompt_type": key[0],
                "prompt_id": key[1],
                "reuse_interval": interval,
                "round": round_idx,
                "staleness_rounds": 0 if last_update_round is None else round_idx - last_update_round,
                "draft_tokens": int(rd.get("draft_tokens", 0)),
                "accepted_len": int(rd.get("accepted_len", 0)),
                "retrieval_update": int(retrieval),
                "selected_count": len(selected),
                "oracle_overlap": jaccard(selected, oracle_selected) if oracle_selected is not None else None,
                "pipeline_hit": int(bool(rd.get("pipeline_hit", False))),
                "pipeline_reuse": int(bool(rd.get("pipeline_reuse", False))),
                "pipeline_wait_ms": float(rd.get("pipeline_wait_ms", 0.0) or 0.0),
            }
        )
    return rows


def _summary_row(result: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = result.get("output") or {}
    timing = result.get("timing") or {}
    spec = result.get("speculative") or {}
    raw = result.get("raw") or {}
    profile = result_profile(result)
    method = str(result.get("method", "?"))
    rounds = int(spec.get("rounds", 0) or 0)
    accepted = int(spec.get("accepted_draft_tokens", 0) or 0)
    draft_length = int(spec.get("draft_length", profile.get("draft_length", 0)) or 0)
    total_ms = float(output.get("total_time_ms", 0.0) or 0.0)
    net_ms = float(timing.get("simulated_network_ms", 0.0) or 0.0)
    retrieval_updates = int(raw.get("retrieval_updates", 0) or 0)
    accepted_ratio = acceptance_ratio(accepted, rounds, draft_length)

    return {
        "method": method,
        "prompt_type": result.get("prompt_type", "?"),
        "prompt_id": result.get("prompt_id", 0),
        "reuse_interval": method_interval(method, profile),
        "tokens_generated": int(output.get("tokens_generated", 0) or 0),
        "rounds": rounds,
        "draft_length": draft_length,
        "accepted_draft_tokens": accepted,
        "acceptance_length": float(spec.get("acceptance_length", 0.0) or 0.0),
        "acceptance_ratio": accepted_ratio,
        "total_time_ms": total_ms,
        "tokens_per_second": float(output.get("tokens_per_second", 0.0) or 0.0),
        "latency_per_accepted_token_ms": total_ms / accepted if accepted > 0 else None,
        "simulated_network_ms": net_ms,
        "retrieval_updates": retrieval_updates,
        "network_ms_per_update": net_ms / retrieval_updates if retrieval_updates > 0 else None,
        "mean_oracle_overlap": mean(row["oracle_overlap"] for row in rows),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Guidance Reuse Summary",
        "",
        "| method | R | acc_len | acc_ratio | latency/accepted ms | updates | oracle_overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda r: (r["reuse_interval"] is None, r["reuse_interval"] or 999999)):
        lines.append(
            "| {method} | {reuse_interval} | {acceptance_length:.3f} | {acceptance_ratio} | "
            "{latency_per_accepted_token_ms} | {retrieval_updates} | {mean_oracle_overlap} |".format(
                **{
                    **row,
                    "acceptance_ratio": "n/a"
                    if row["acceptance_ratio"] is None
                    else f"{row['acceptance_ratio']:.4f}",
                    "latency_per_accepted_token_ms": "n/a"
                    if row["latency_per_accepted_token_ms"] is None
                    else f"{row['latency_per_accepted_token_ms']:.2f}",
                    "mean_oracle_overlap": "n/a"
                    if row["mean_oracle_overlap"] is None
                    else f"{row['mean_oracle_overlap']:.4f}",
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze guidance reuse sweep results.")
    parser.add_argument("results_file", nargs="?", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    results_path = args.results_file or latest_quick_result()
    if results_path is None or not results_path.exists():
        raise FileNotFoundError("No quick_test results file found.")

    if args.run_dir is not None:
        run_dir = args.run_dir
    elif results_path.parent.name == "raw":
        run_dir = results_path.parent.parent
    else:
        run_dir = results_path.parent
    analysis_dir = run_dir / "analysis"

    data = load_json(results_path)
    results = [r for r in data.get("results", []) if r.get("method_family") == "specextend"]
    oracle = _find_oracle_rounds(results)

    round_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for result in results:
        rows = _round_rows(result, oracle)
        round_rows.extend(rows)
        summary_rows.append(_summary_row(result, rows))

    _write_csv(analysis_dir / "guidance_reuse_rounds.csv", round_rows)
    _write_csv(analysis_dir / "guidance_reuse_summary.csv", summary_rows)
    _write_report(analysis_dir / "guidance_reuse_report.md", summary_rows)

    print(f"SUMMARY_CSV={analysis_dir / 'guidance_reuse_summary.csv'}")
    print(f"ROUND_CSV={analysis_dir / 'guidance_reuse_rounds.csv'}")
    print(f"REPORT={analysis_dir / 'guidance_reuse_report.md'}")


if __name__ == "__main__":
    main()
