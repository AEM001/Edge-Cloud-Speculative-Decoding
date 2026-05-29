#!/usr/bin/env python3
"""Download and prepare LongBench v2 for experiments."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download


_DATA_DIR = Path(__file__).parent.parent / "data"


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


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_source_rows(path: Path) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        for key in ("data", "train", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value]
    raise ValueError(f"Unsupported LongBench v2 JSON structure in {path}")


def download_longbench_v2(repo_id: str = "THUDM/LongBench-v2", split: str = "train") -> None:
    """Download LongBench v2 and normalize it into local jsonl files."""
    print(f"Downloading LongBench v2 dataset from {repo_id}...")

    output_dir = _DATA_DIR / "longbench_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="data.json",
            repo_type="dataset",
        )
    )
    source_rows = _read_source_rows(source_path)
    rows: List[Dict] = []
    partitions: Dict[str, List[Dict]] = {}

    for idx, item in enumerate(source_rows, start=1):
        raw = dict(item)
        prompt = _make_longbench_v2_prompt(raw)
        context = raw.get("context", "")
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

    all_path = output_dir / f"{split}.jsonl"
    _write_jsonl(all_path, rows)
    print(f"Saved {len(rows)} examples to {all_path}")

    partition_manifest = {}
    for length, items in sorted(partitions.items()):
        path = output_dir / f"{length}.jsonl"
        _write_jsonl(path, items)
        partition_manifest[length] = {
            "path": str(path.relative_to(_DATA_DIR.parent)),
            "num_examples": len(items),
            "min_prompt_chars": min(row["prompt_chars"] for row in items),
            "max_prompt_chars": max(row["prompt_chars"] for row in items),
        }
        print(f"Saved {len(items)} {length} examples to {path}")

    manifest = {
        "dataset": repo_id,
        "split": split,
        "num_examples": len(rows),
        "path": str(all_path.relative_to(_DATA_DIR.parent)),
        "partitions": partition_manifest,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")
    print("LongBench v2 download complete!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LongBench v2 into data/longbench_v2.")
    parser.add_argument(
        "--repo",
        default="THUDM/LongBench-v2",
        help="Hugging Face repo id for LongBench v2.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download.",
    )
    return parser.parse_args()


def download_pg19(repo_id: str = "emozilla/pg19", split: str = "test") -> None:
    """Download PG-19 books dataset and normalize into local jsonl files."""
    print(f"Downloading PG-19 dataset from {repo_id}...")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("`datasets` library is required. Install it: pip install datasets") from exc

    output_dir = _DATA_DIR / "pg19"
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(repo_id, split=split)
    rows = []
    for idx, item in enumerate(ds, start=1):
        text = item.get("text", "")
        row = {
            "id": idx,
            "original_id": item.get("short_book_title", idx),
            "source": "pg19",
            "text": text,
            "text_chars": len(text),
            "text_words": len(text.split()),
            "publication_date": item.get("publication_date"),
            "book_title": item.get("short_book_title"),
        }
        rows.append(row)

    all_path = output_dir / f"{split}.jsonl"
    _write_jsonl(all_path, rows)
    print(f"Saved {len(rows)} examples to {all_path}")

    manifest = {
        "dataset": repo_id,
        "split": split,
        "num_examples": len(rows),
        "path": str(all_path.relative_to(_DATA_DIR.parent)),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")
    print("PG-19 download complete!")


if __name__ == "__main__":
    args = parse_args()
    download_longbench_v2(args.repo, args.split)
