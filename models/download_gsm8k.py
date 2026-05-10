#!/usr/bin/env python3
"""Download GSM8K dataset and convert to JSONL format."""
import json
from pathlib import Path
from datasets import load_dataset


def download_gsm8k():
    """Download GSM8K dataset and save as JSONL files."""
    data_dir = Path(__file__).parent.parent / "data" / "gsm8k"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading GSM8K dataset...")
    
    # Load GSM8K from Hugging Face
    dataset = load_dataset("openai/gsm8k", "main")
    
    # Save test split
    test_path = data_dir / "test.jsonl"
    with open(test_path, "w") as f:
        for item in dataset["test"]:
            f.write(json.dumps({"question": item["question"]}) + "\n")
    print(f"Saved {len(dataset['test'])} test examples to {test_path}")
    
    # Save train split
    train_path = data_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for item in dataset["train"]:
            f.write(json.dumps({"question": item["question"]}) + "\n")
    print(f"Saved {len(dataset['train'])} train examples to {train_path}")
    
    print("Done!")


if __name__ == "__main__":
    download_gsm8k()
