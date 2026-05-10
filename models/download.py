#!/usr/bin/env python3
"""Download Hugging Face model snapshots with resume support."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct-AWQ": "Qwen2.5-1.5B-Instruct-AWQ",
    "Qwen/Qwen2.5-32B-Instruct-AWQ": "Qwen2.5-32B-Instruct-AWQ",
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
        help="Specific Hugging Face repo id to download. Can be provided multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_models = args.models or list(DEFAULT_MODELS.keys())
    token = _load_token()

    for repo_id in selected_models:
        folder_name = DEFAULT_MODELS.get(repo_id, repo_id.split("/")[-1])
        local_path = args.base_dir / folder_name
        download_model(repo_id, local_path, token=token)

    print("All downloads completed!")


if __name__ == "__main__":
    main()
