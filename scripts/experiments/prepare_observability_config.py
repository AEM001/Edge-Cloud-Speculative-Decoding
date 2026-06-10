"""Prepare the fixed observability experiment config.

This script does not run models. It creates:

  outputs/observability/<run_id>/
    configs/observability.json
    manifest.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from observability_common import DEFAULT_QUICK_CONFIG, OUTPUT_ROOT, load_json, timestamp_run_id, write_json


def build_config(base: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(base)
    profile = {
        "name": "observability_fixed_R16_k16",
        "label": "SpecExtend observability, fixed refresh R=16, top_k=16",
        "retrieve_every_n_steps": 16,
        "retrieve_on_first_round": True,
        "retrieve_top_k": 16,
        "retrieval_chunk_size": int(config.get("retrieval_chunk_size", 64)),
        "draft_length": 8,
        "draft_mode": "branching",
        "draft_tree_nodes": int(config.get("draft_tree_nodes", 32)),
        "draft_tree_max_depth": 6,
    }
    config.update(
        {
            "methods": (["direct"] if args.include_direct else []) + [profile["name"]],
            "draft_mode": "branching",
            "draft_length": 8,
            "draft_tree_max_depth": 6,
            "retrieve_top_k": 16,
            "retrieve_every_n_steps": 16,
            "spec_profiles": [profile],
        }
    )
    for key, value in {
        "prompt_count": args.prompt_count,
        "max_tokens": args.max_tokens,
        "prompt_input_tokens": args.prompt_input_tokens,
        "dataset_split": args.dataset_split,
    }.items():
        if value is not None:
            config[key] = value
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the fixed observability experiment config.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_QUICK_CONFIG)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prompt-count", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--prompt-input-tokens", type=int, default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--include-direct", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or timestamp_run_id()
    run_dir = OUTPUT_ROOT / run_id
    config_path = run_dir / "configs" / "observability.json"
    config = build_config(load_json(args.base_config), args)

    write_json(config_path, config)
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "purpose": "SpecExtend observability logging for fixed cloud guidance refresh R=16, top_k=16.",
            "run_dir": str(run_dir),
            "config": str(config_path),
            "fixed_parameters": {
                "draft_tree_max_depth": 6,
                "retrieve_top_k": 16,
                "retrieve_every_n_steps": 16,
            },
            "outputs": {
                "raw_quick_result": str(run_dir / "raw" / "quick_test_results.json"),
                "round_observations": str(run_dir / "raw" / "round_observations.jsonl"),
                "summary_csv": str(run_dir / "analysis" / "summary.csv"),
                "rounds_csv": str(run_dir / "analysis" / "rounds.csv"),
                "report": str(run_dir / "report.md"),
            },
        },
    )
    print(f"RUN_DIR={run_dir}")
    print(f"CONFIG_PATH={config_path}")


if __name__ == "__main__":
    main()
