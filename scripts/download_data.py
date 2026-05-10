#!/usr/bin/env python3
"""Download and prepare datasets for experiments."""

import json
from pathlib import Path

from datasets import load_dataset


_DATA_DIR = Path(__file__).parent.parent / "data"


def download_gsm8k() -> None:
    """Download GSM8K dataset and convert to jsonl format."""
    print("Downloading GSM8K dataset...")
    
    # Create output directory
    output_dir = _DATA_DIR / "gsm8k"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from Hugging Face
    dataset = load_dataset("openai/gsm8k", "main")
    
    # Process test split
    test_data = dataset["test"]
    test_path = output_dir / "test.jsonl"
    
    with open(test_path, "w") as f:
        for item in test_data:
            row = {"question": item["question"]}
            f.write(json.dumps(row) + "\n")
    
    print(f"Saved {len(test_data)} test examples to {test_path}")
    
    # Process train split (optional, for future use)
    train_data = dataset["train"]
    train_path = output_dir / "train.jsonl"
    
    with open(train_path, "w") as f:
        for item in train_data:
            row = {"question": item["question"]}
            f.write(json.dumps(row) + "\n")
    
    print(f"Saved {len(train_data)} train examples to {train_path}")
    print("GSM8K download complete!")


def download_humaneval() -> None:
    """Download HumanEval dataset and convert to jsonl format."""
    print("Downloading HumanEval dataset...")
    
    # Create output directory
    output_dir = _DATA_DIR / "humaneval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from Hugging Face
    dataset = load_dataset("openai/openai_humaneval")
    
    # Process test split
    test_data = dataset["test"]
    test_path = output_dir / "test.jsonl"
    
    with open(test_path, "w") as f:
        for item in test_data:
            row = {"prompt": item["prompt"]}
            f.write(json.dumps(row) + "\n")
    
    print(f"Saved {len(test_data)} test examples to {test_path}")
    print("HumanEval download complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "humaneval", "all"],
        default="gsm8k",
        help="Dataset to download",
    )
    
    args = parser.parse_args()
    
    if args.dataset == "gsm8k":
        download_gsm8k()
    elif args.dataset == "humaneval":
        download_humaneval()
    elif args.dataset == "all":
        download_gsm8k()
        download_humaneval()
