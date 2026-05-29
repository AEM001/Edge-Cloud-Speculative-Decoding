#!/usr/bin/env python3
"""Download Hugging Face model snapshots with resume support."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ALIASES = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "Qwen3-8B"),
}

PRESETS = {
    "edge-cloud": ["qwen3-8b"],
    "separate": ["qwen3-1.7b", "qwen3-8b"],
}


def _load_token() -> str | None:
    token_file = Path(__file__).resolve().parent / "hf.txt"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token
    return None


def download_model(repo_id: str, local_dir: Path, token: str | None = None) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} to {local_dir}...")
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
    )
    print(f"Completed: {repo_id}")
    return Path(downloaded_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "Model alias or Hugging Face repo id. Can be provided multiple times. "
            f"Aliases: {', '.join(sorted(MODEL_ALIASES))}."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="edge-cloud",
        help=(
            "Preset to download when --model is not provided: edge-cloud downloads Qwen3-8B, "
            "separate downloads Qwen3-1.7B for draft and Qwen3-8B for verify."
        ),
    )
    return parser.parse_args()


def resolve_model(model: str) -> tuple[str, str]:
    key = model.lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    return model, model.split("/")[-1]


def main() -> None:
    args = parse_args()
    selected_models = args.models or PRESETS[args.preset]
    token = _load_token()

    for model in selected_models:
        repo_id, folder_name = resolve_model(model)
        local_path = args.base_dir / folder_name
        download_model(repo_id, local_path, token=token)

    print("All downloads completed!")


if __name__ == "__main__":
    main()
