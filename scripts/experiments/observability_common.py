"""Shared helpers for the observability experiment scripts."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_ROOT = REPO_ROOT / "outputs" / "observability"
DEFAULT_QUICK_CONFIG = REPO_ROOT / "quick_benchmark_config.json"


def timestamp_run_id(prefix: str = "observability") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def result_rounds(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(((result.get("raw") or {}).get("round_details") or []))


def flatten_rounds(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in data.get("results", []):
        if result.get("method_family") != "specextend":
            continue
        profile = (result.get("raw") or {}).get("profile") or {}
        for detail in result_rounds(result):
            edge = detail.get("edge_observability") or {}
            cloud = detail.get("cloud_observability") or {}
            rows.append(
                {
                    "method": result.get("method"),
                    "network": result.get("network"),
                    "prompt_type": result.get("prompt_type"),
                    "prompt_id": result.get("prompt_id"),
                    "round": detail.get("round"),
                    "generated_token_offset": detail.get("generated_token_offset"),
                    "generation_phase": detail.get("generation_phase"),
                    "draft_tokens": detail.get("draft_tokens"),
                    "accepted_len": detail.get("accepted_len"),
                    "accept_ratio": detail.get("accept_ratio"),
                    "rejected_position": detail.get("rejected_position"),
                    "acceptance_trend_slope": detail.get("acceptance_trend_slope"),
                    "retrieval": detail.get("retrieval"),
                    "guidance_jaccard": detail.get("guidance_jaccard"),
                    "tokens_since_guidance_update": detail.get("tokens_since_guidance_update"),
                    "request_payload_bytes": detail.get("request_payload_bytes"),
                    "response_payload_bytes": detail.get("response_payload_bytes"),
                    "verify_elapsed_ms": detail.get("verify_elapsed_ms"),
                    "network_time_ms": detail.get("network_time_ms"),
                    "draft_entropy": edge.get("draft_entropy"),
                    "draft_top1_confidence": edge.get("draft_top1_confidence"),
                    "draft_top1_top2_margin": edge.get("draft_top1_top2_margin"),
                    "draft_tree_nodes": edge.get("draft_tree_nodes"),
                    "draft_tree_actual_depth": edge.get("draft_tree_actual_depth"),
                    "full_kv_tokens": edge.get("full_kv_tokens"),
                    "working_kv_tokens": edge.get("working_kv_tokens"),
                    "selected_full_ratio": edge.get("selected_full_ratio"),
                    "kv_append_ms": edge.get("kv_append_ms"),
                    "kv_select_ms": edge.get("kv_select_ms"),
                    "tree_construct_ms": edge.get("tree_construct_ms"),
                    "cuda_memory_allocated_mb": edge.get("cuda_memory_allocated_mb"),
                    "cloud_target_verify_time_ms": cloud.get("target_verify_time_ms"),
                    "cloud_guidance_generation_time_ms": cloud.get("guidance_generation_time_ms"),
                    "top_attention_token_indices": cloud.get("top_attention_token_indices"),
                    "attention_mass_covered": cloud.get("attention_mass_covered"),
                    "cloud_selected_count": cloud.get("selected_count"),
                    "cloud_selected_token_count": cloud.get("selected_token_count"),
                    "retrieve_every_n_steps": profile.get("retrieve_every_n_steps"),
                    "retrieve_top_k": profile.get("retrieve_top_k"),
                    "draft_tree_max_depth": profile.get("draft_tree_max_depth"),
                }
            )
    return rows
