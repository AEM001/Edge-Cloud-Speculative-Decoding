"""Run a prepared guidance reuse sweep with quick_test."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from guidance_common import load_json, write_json
from quick_test import load_config_file, run_quick_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick_test for a prepared guidance reuse config.")
    parser.add_argument("config", type=Path, help="Path from prepare_guidance_reuse_configs.py")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(args.config)

    run_dir = args.run_dir or args.config.parent.parent
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_file(args.config)
    result_path = run_quick_test(config)
    if result_path is None:
        raise RuntimeError("quick_test did not produce results")

    copied = raw_dir / result_path.name
    shutil.copy2(result_path, copied)

    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    manifest["raw_quick_result"] = str(copied)
    manifest["source_quick_result"] = str(result_path)
    write_json(manifest_path, manifest)

    print(f"RESULTS_PATH={copied}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()

