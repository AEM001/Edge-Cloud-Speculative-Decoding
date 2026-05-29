#!/usr/bin/env python3
"""Fast dataset downloader/preparer for SpecExtend experiments.

Examples:
  python scripts/download_datasets_fast.py --dataset pg19
  python scripts/download_datasets_fast.py --dataset longbench_v2
  python scripts/download_datasets_fast.py --dataset all --limit 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from datasets import DownloadConfig, load_dataset


REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data"


def _enable_fast_hf_transfer() -> None:
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_manifest(output_dir: Path, manifest: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _download_config(workers: int, force: bool) -> DownloadConfig:
    return DownloadConfig(
        max_retries=10,
        num_proc=max(1, workers),
        resume_download=True,
        force_download=force,
    )


def download_pg19(split: str, limit: Optional[int], workers: int, force: bool) -> None:
    repo_id = "emozilla/pg19"
    output_dir = DATA_DIR / "pg19"
    output_path = output_dir / f"{split}.jsonl"

    print(f"Downloading {repo_id} split={split} ...")
    ds = load_dataset(
        repo_id,
        split=split,
        download_config=_download_config(workers, force),
    )
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    def rows():
        for idx, item in enumerate(ds, start=1):
            text = item.get("text", "")
            yield {
                "id": idx,
                "original_id": item.get("short_book_title", idx),
                "source": "pg19",
                "text": text,
                "text_chars": len(text),
                "text_words": len(text.split()),
                "publication_date": item.get("publication_date"),
                "book_title": item.get("short_book_title"),
            }

    count = _write_jsonl(output_path, rows())
    _write_manifest(
        output_dir,
        {
            "dataset": repo_id,
            "split": split,
            "num_examples": count,
            "path": str(output_path.relative_to(REPO_DIR)),
        },
    )
    print(f"Saved {count} PG-19 rows to {output_path}")


def _make_longbench_v2_prompt(row: Dict) -> str:
    return (
        "Read the following context and answer the multiple-choice question.\n\n"
        "Context:\n"
        f"{row.get('context', '').strip()}\n\n"
        "Question:\n"
        f"{row.get('question', '').strip()}\n\n"
        "Choices:\n"
        f"A. {row.get('choice_A', '').strip()}\n"
        f"B. {row.get('choice_B', '').strip()}\n"
        f"C. {row.get('choice_C', '').strip()}\n"
        f"D. {row.get('choice_D', '').strip()}\n\n"
        "Answer with the best choice and explain your reasoning."
    )


def download_longbench_v2(split: str, limit: Optional[int], workers: int, force: bool) -> None:
    repo_id = "THUDM/LongBench-v2"
    output_dir = DATA_DIR / "longbench_v2"
    all_path = output_dir / f"{split}.jsonl"

    print(f"Downloading {repo_id} split={split} ...")
    ds = load_dataset(
        repo_id,
        split=split,
        download_config=_download_config(workers, force),
    )
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows: List[Dict] = []
    partitions: Dict[str, List[Dict]] = {}
    for idx, item in enumerate(ds, start=1):
        raw = dict(item)
        context = raw.get("context", "")
        prompt = _make_longbench_v2_prompt(raw)
        length = str(raw.get("length", "unknown") or "unknown").lower()
        row = {
            "id": idx,
            "original_id": raw.get("_id", idx),
            "source": "longbench_v2",
            "domain": raw.get("domain"),
            "sub_domain": raw.get("sub_domain"),
            "difficulty": raw.get("difficulty"),
            "length": length,
            "question": raw.get("question", ""),
            "choices": {
                "A": raw.get("choice_A", ""),
                "B": raw.get("choice_B", ""),
                "C": raw.get("choice_C", ""),
                "D": raw.get("choice_D", ""),
            },
            "answer": raw.get("answer", ""),
            "context": context,
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "prompt_words": len(prompt.split()),
            "context_chars": len(context),
            "context_words": len(context.split()),
        }
        rows.append(row)
        partitions.setdefault(length, []).append(row)

    count = _write_jsonl(all_path, rows)
    partition_manifest = {}
    for length, items in sorted(partitions.items()):
        path = output_dir / f"{length}.jsonl"
        _write_jsonl(path, items)
        partition_manifest[length] = {
            "path": str(path.relative_to(REPO_DIR)),
            "num_examples": len(items),
        }

    _write_manifest(
        output_dir,
        {
            "dataset": repo_id,
            "split": split,
            "num_examples": count,
            "path": str(all_path.relative_to(REPO_DIR)),
            "partitions": partition_manifest,
        },
    )
    print(f"Saved {count} LongBench v2 rows to {all_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download experiment datasets quickly.")
    parser.add_argument(
        "--dataset",
        choices=["pg19", "longbench_v2", "all"],
        default="pg19",
        help="Dataset to download and normalize.",
    )
    parser.add_argument("--split", default=None, help="Override split. Defaults: pg19=test, longbench_v2=train.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for quick local smoke data.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers for Hugging Face datasets.")
    parser.add_argument("--force", action="store_true", help="Force re-download instead of using cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _enable_fast_hf_transfer()
    limit = args.limit if args.limit > 0 else None

    if args.dataset in {"pg19", "all"}:
        download_pg19(args.split or "test", limit, args.workers, args.force)
    if args.dataset in {"longbench_v2", "all"}:
        download_longbench_v2(args.split or "train", limit, args.workers, args.force)


if __name__ == "__main__":
    main()
