"""Analyze fixed SpecExtend observability results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from observability_common import REPO_ROOT, flatten_rounds, load_json, mean, write_csv, write_jsonl


def _num(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("method"), row.get("network"))].append(row)

    summaries: List[Dict[str, Any]] = []
    for (method, network), group in sorted(grouped.items()):
        accepted = sum(int(row.get("accepted_len") or 0) for row in group)
        updates = sum(1 for row in group if row.get("retrieval"))
        payload_bytes = sum(int(row.get("request_payload_bytes") or 0) + int(row.get("response_payload_bytes") or 0) for row in group)
        verify_ms = sum(_num(row.get("verify_elapsed_ms")) or 0.0 for row in group)
        summaries.append(
            {
                "method": method,
                "network": network,
                "rounds": len(group),
                "accepted_tokens": accepted,
                "guidance_updates": updates,
                "tokens_per_update": accepted / updates if updates else None,
                "latency_per_accepted_token_ms": verify_ms / accepted if accepted else None,
                "payload_bytes": payload_bytes,
                "mean_accept_ratio": mean(_num(row.get("accept_ratio")) for row in group),
                "mean_draft_entropy": mean(_num(row.get("draft_entropy")) for row in group),
                "mean_top1_confidence": mean(_num(row.get("draft_top1_confidence")) for row in group),
                "mean_attention_mass_covered": mean(_num(row.get("attention_mass_covered")) for row in group),
                "mean_guidance_jaccard": mean(_num(row.get("guidance_jaccard")) for row in group),
                "mean_selected_full_ratio": mean(_num(row.get("selected_full_ratio")) for row in group),
                "mean_kv_select_ms": mean(_num(row.get("kv_select_ms")) for row in group),
                "mean_tree_construct_ms": mean(_num(row.get("tree_construct_ms")) for row in group),
            }
        )
    return summaries


def _write_report(path: Path, summaries: List[Dict[str, Any]]) -> None:
    lines = [
        "# SpecExtend Observability Report",
        "",
        "Fixed parameters: `draft_tree_max_depth=6`, `retrieve_top_k=16`, `retrieve_every_n_steps=16`.",
        "",
        "| method | rounds | accepted | updates | tokens/update | latency/accepted ms | attn mass | guidance jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {method} | {rounds} | {accepted_tokens} | {guidance_updates} | {tokens_per_update} | "
            "{latency_per_accepted_token_ms} | {mean_attention_mass_covered} | {mean_guidance_jaccard} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _copy_latest_outputs(run_dir: Path, rows: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> None:
    latest_dir = REPO_ROOT / "Analysis"
    write_csv(latest_dir / "observability_latest_rounds.csv", rows)
    write_csv(latest_dir / "observability_latest_summary.csv", summaries)
    _write_report(latest_dir / "observability_latest_report.md", summaries)
    (latest_dir / "LATEST_RUN.txt").write_text(f"{run_dir.name}\n")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fixed SpecExtend observability outputs.")
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    data = load_json(args.results)
    run_dir = args.results.parent.parent
    analysis_dir = run_dir / "analysis"
    rows = flatten_rounds(data)
    summaries = _summarize(rows)

    write_jsonl(run_dir / "raw" / "round_observations.jsonl", rows)
    write_csv(analysis_dir / "rounds.csv", rows)
    write_csv(analysis_dir / "summary.csv", summaries)
    _write_report(run_dir / "report.md", summaries)
    _copy_latest_outputs(run_dir, rows, summaries)

    print(f"ROUNDS_CSV={analysis_dir / 'rounds.csv'}")
    print(f"SUMMARY_CSV={analysis_dir / 'summary.csv'}")
    print(f"REPORT={run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
