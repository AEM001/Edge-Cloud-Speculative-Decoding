#!/usr/bin/env python3
"""Download Hugging Face model snapshots with resume support."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ALIASES = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    "qwen3-4b": ("Qwen/Qwen3-4B", "Qwen3-4B"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "Qwen3-8B"),
    "qwen3-14b": ("Qwen/Qwen3-14B", "Qwen3-14B"),
    "qwen3-4b-awq": ("Qwen/Qwen3-4B-AWQ", "Qwen3-4B-AWQ"),
    "qwen3-14b-awq": ("Qwen/Qwen3-14B-AWQ", "Qwen3-14B-AWQ"),
}

PRESETS = {
    "smoke": ["qwen3-0.6b"],
    "edge-cloud": ["qwen3-4b"],
    "separate": ["qwen3-1.7b", "qwen3-4b"],
}


def _load_token() -> str | None:
    token_file = Path(__file__).resolve().parent / "hf.txt"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token
    return None


def download_model(repo_id: str, local_dir: Path, token: str | None = None, source: str = "huggingface") -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} from {source} to {local_dir}...")
    if source == "modelscope":
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
        downloaded_path = ms_snapshot_download(
            model_id=repo_id,
            local_dir=str(local_dir),
        )
    else:
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
            "Preset to download when --model is not provided: smoke downloads a tiny "
            "Qwen3, edge-cloud downloads Qwen3-4B, separate downloads Qwen3-1.7B "
            "for draft and Qwen3-4B for verify."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Download source: huggingface or modelscope.",
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
        download_model(repo_id, local_path, token=token, source=args.source)

    print("All downloads completed!")


if __name__ == "__main__":
    main()
