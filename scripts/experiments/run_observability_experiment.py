"""Run the fixed observability experiment and archive raw outputs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from observability_common import flatten_rounds, load_json, write_json, write_jsonl


def _repo_root() -> Path:
    return Path(__file__).parent.parent.parent


def _extract_results_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("RESULTS_PATH="):
            return Path(line.split("=", 1)[1].strip())
    raise RuntimeError("quick_test.py did not print RESULTS_PATH")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed SpecExtend observability experiment.")
    parser.add_argument("config", type=Path, help="Path from prepare_observability_config.py")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    run_dir = config_path.parent.parent
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_repo_root() / "scripts" / "experiments" / "quick_test.py"),
        "--config",
        str(config_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    source = _extract_results_path(completed.stdout).resolve()
    raw_result = raw_dir / "quick_test_results.json"
    shutil.copy2(source, raw_result)
    data = load_json(raw_result)
    write_jsonl(raw_dir / "round_observations.jsonl", flatten_rounds(data))

    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    manifest["raw_source_result"] = str(source)
    manifest["raw_quick_result"] = str(raw_result)
    write_json(manifest_path, manifest)

    if not args.skip_analysis:
        subprocess.run(
            [
                sys.executable,
                str(_repo_root() / "scripts" / "experiments" / "analyze_observability.py"),
                str(raw_result),
            ],
            cwd=_repo_root(),
            check=True,
        )

    print(f"RAW_RESULT={raw_result}")
    print(f"ROUND_OBSERVATIONS={raw_dir / 'round_observations.jsonl'}")


if __name__ == "__main__":
    main()
