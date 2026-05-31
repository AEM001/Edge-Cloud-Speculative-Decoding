#!/usr/bin/env python3
"""Download GovReport and PG-19 datasets from Hugging Face."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset


_DATA_DIR = Path(__file__).parent.parent / "data"


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_govreport(split: str = "test", max_samples: int = 1000) -> None:
    """Download GovReport dataset from Hugging Face."""
    print(f"Downloading GovReport dataset (split={split}, max_samples={max_samples})...")
    
    output_dir = _DATA_DIR / "govreport"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset("ccdv/govreport-summarization", split=split)
    
    rows = []
    for idx, item in enumerate(dataset):
        if idx >= max_samples:
            break
        
        # GovReport has 'report' and 'summary' fields
        report_text = item.get("report", "")
        if not report_text:
            continue
            
        row = {
            "id": idx + 1,
            "text": report_text,
            "summary": item.get("summary", ""),
            "source": "govreport",
            "length": len(report_text),
        }
        rows.append(row)
    
    output_path = output_dir / f"{split}.jsonl"
    _write_jsonl(output_path, rows)
    print(f"Saved {len(rows)} examples to {output_path}")
    print("GovReport download complete!")


def download_pg19(split: str = "test", max_samples: int = 1000) -> None:
    """Download PG-19 dataset from Hugging Face."""
    print(f"Downloading PG-19 dataset (split={split}, max_samples={max_samples})...")
    
    output_dir = _DATA_DIR / "pg19"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use emozilla/pg19 which has parquet files (no loading script)
    dataset = load_dataset("emozilla/pg19", split=split)
    
    rows = []
    for idx, item in enumerate(dataset):
        if idx >= max_samples:
            break
        
        # PG-19 has 'text' field
        text = item.get("text", "")
        if not text:
            continue
            
        row = {
            "id": idx + 1,
            "text": text,
            "source": "pg19",
            "length": len(text),
            "title": item.get("title", ""),
        }
        rows.append(row)
    
    output_path = output_dir / f"{split}.jsonl"
    _write_jsonl(output_path, rows)
    print(f"Saved {len(rows)} examples to {output_path}")
    print("PG-19 download complete!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GovReport or PG-19 datasets.")
    parser.add_argument(
        "--dataset",
        choices=["govreport", "pg19", "all"],
        default="all",
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to download.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to download.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.dataset in ["govreport", "all"]:
        download_govreport(args.split, args.max_samples)
    
    if args.dataset in ["pg19", "all"]:
        download_pg19(args.split, args.max_samples)
