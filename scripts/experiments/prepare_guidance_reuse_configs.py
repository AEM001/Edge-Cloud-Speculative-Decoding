"""Prepare config files for the cloud-guidance reuse experiments.

This script does not run models. It creates a timestamped experiment directory:

  outputs/guidance_reuse/<run_id>/
    configs/guidance_reuse_sweep.json
    manifest.json

The generated quick_test config contains one SpecExtend profile per reuse
window R, plus an optional no-update profile.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from guidance_common import (
    DEFAULT_QUICK_CONFIG,
    OUTPUT_ROOT,
    load_json,
    parse_int_list,
    timestamp_run_id,
    write_json,
)


def build_profiles(base: Dict[str, Any], reuse_windows: List[int], include_no_update: bool) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    for r in reuse_windows:
        profiles.append(
            {
                "name": f"reuse_R{r}",
                "label": f"Fixed-R cloud guidance update, R={r}",
                "retrieve_every_n_steps": r,
                "retrieve_on_first_round": True,
                "retrieve_top_k": int(base.get("retrieve_top_k", 16)),
                "retrieval_chunk_size": int(base.get("retrieval_chunk_size", 64)),
                "draft_length": int(base.get("draft_length", 8)),
                "draft_mode": base.get("draft_mode", "branching"),
                "draft_tree_nodes": int(base.get("draft_tree_nodes", 32)),
                "draft_tree_max_depth": int(base.get("draft_tree_max_depth", 8)),
            }
        )
    if include_no_update:
        profiles.append(
            {
                "name": "no_update",
                "label": "No cloud attention refresh after initialization",
                "retrieve_every_n_steps": 0,
                "retrieve_on_first_round": False,
                "retrieve_top_k": int(base.get("retrieve_top_k", 16)),
                "retrieval_chunk_size": int(base.get("retrieval_chunk_size", 64)),
                "draft_length": int(base.get("draft_length", 8)),
                "draft_mode": base.get("draft_mode", "branching"),
                "draft_tree_nodes": int(base.get("draft_tree_nodes", 32)),
                "draft_tree_max_depth": int(base.get("draft_tree_max_depth", 8)),
            }
        )
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Create guidance reuse sweep configs.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_QUICK_CONFIG)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--reuse-windows", default="1,2,4,8,16,32")
    parser.add_argument("--prompt-count", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--prompt-input-tokens", type=int, default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--include-direct", action="store_true")
    parser.add_argument("--no-no-update", action="store_true")
    args = parser.parse_args()

    base = load_json(args.base_config)
    reuse_windows = parse_int_list(args.reuse_windows)
    run_id = args.run_id or timestamp_run_id("guidance_reuse")
    run_dir = OUTPUT_ROOT / run_id
    config_path = run_dir / "configs" / "guidance_reuse_sweep.json"

    config: Dict[str, Any] = dict(base)
    profiles = build_profiles(base, reuse_windows, include_no_update=not args.no_no_update)
    config["spec_profiles"] = profiles
    config["methods"] = (["direct"] if args.include_direct else []) + [p["name"] for p in profiles]

    for key, value in {
        "prompt_count": args.prompt_count,
        "max_tokens": args.max_tokens,
        "prompt_input_tokens": args.prompt_input_tokens,
        "dataset_split": args.dataset_split,
    }.items():
        if value is not None:
            config[key] = value

    write_json(config_path, config)
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "purpose": "Guidance reuse window sweep for cloud-guided sparse edge KV.",
            "run_dir": str(run_dir),
            "config": str(config_path),
            "outputs": {
                "raw_quick_result": "Filled by run_guidance_reuse_sweep.py",
                "summary_csv": str(run_dir / "analysis" / "guidance_reuse_summary.csv"),
                "per_round_csv": str(run_dir / "analysis" / "guidance_reuse_rounds.csv"),
                "signal_correlations": str(run_dir / "analysis" / "update_signal_correlations.csv"),
                "adaptive_policy": str(run_dir / "analysis" / "adaptive_policy_simulation.csv"),
            },
            "reuse_windows": reuse_windows,
        },
    )

    print(f"RUN_DIR={run_dir}")
    print(f"CONFIG_PATH={config_path}")


if __name__ == "__main__":
    main()
