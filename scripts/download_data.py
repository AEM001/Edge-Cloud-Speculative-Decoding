#!/usr/bin/env python3
"""Download and prepare datasets for experiments."""

import json
import random
from pathlib import Path
from typing import Dict, List

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


def _word_bucket(count: int, prefix: str) -> str:
    """Return a stable coarse bucket name for approximate whitespace word counts."""
    if count < 2_000:
        return f"{prefix}_0_2k"
    if count < 4_000:
        return f"{prefix}_2k_4k"
    if count < 8_000:
        return f"{prefix}_4k_8k"
    if count < 16_000:
        return f"{prefix}_8k_16k"
    return f"{prefix}_16k_plus"


def _format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """Format user-side messages as a plain prompt for the current benchmark client."""
    user_parts = [m["content"].strip() for m in messages if m.get("role") == "user"]
    if not user_parts:
        return ""
    return "\n\n".join(user_parts)


def download_longwriter() -> None:
    """Download LongWriter-6k and convert it into one normalized jsonl file."""
    print("Downloading LongWriter-6k dataset...")

    output_dir = _DATA_DIR / "longwriter_6k"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("zai-org/LongWriter-6k")["train"]
    rows = []

    for idx, item in enumerate(dataset):
        messages = item["messages"]
        prompt = _format_chat_prompt(messages)
        targets = [m["content"].strip() for m in messages if m.get("role") == "assistant"]
        target = "\n\n".join(targets)

        prompt_words = len(prompt.split())
        target_words = len(target.split())
        prompt_bucket = _word_bucket(prompt_words, "prompt")
        target_bucket = _word_bucket(target_words, "target")

        row = {
            "id": idx + 1,
            "source": "longwriter_6k",
            "prompt": prompt,
            "target": target,
            "messages": messages,
            "prompt_chars": len(prompt),
            "target_chars": len(target),
            "prompt_words": prompt_words,
            "target_words": target_words,
            "prompt_bucket": prompt_bucket,
            "target_bucket": target_bucket,
        }
        rows.append(row)

    all_path = output_dir / "train.jsonl"
    with open(all_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} examples to {all_path}")

    manifest = {
        "dataset": "zai-org/LongWriter-6k",
        "split": "train",
        "num_examples": len(rows),
        "path": str(all_path.relative_to(_DATA_DIR.parent)),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")
    print("LongWriter-6k download complete!")


def _load_longwriter_rows() -> List[Dict]:
    path = _DATA_DIR / "longwriter_6k" / "train.jsonl"
    if not path.exists():
        download_longwriter()
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _take_words(text: str, count: int) -> str:
    words = text.split()
    return " ".join(words[:count])


def _build_context_pool(rows: List[Dict]) -> List[str]:
    pool = []
    for row in rows:
        target = row.get("target", "").strip()
        if len(target.split()) >= 1_000:
            pool.append(target)
    if not pool:
        raise ValueError("LongWriter target text pool is empty.")
    return pool


def _make_single_turn_prompt(base_prompt: str, context_text: str) -> str:
    return (
        "Use the reference material below as background context. The reference "
        "material may cover adjacent topics; write the requested response using "
        "the parts that are relevant.\n\n"
        "Reference material:\n"
        f"{context_text}\n\n"
        "User request:\n"
        f"{base_prompt.strip()}\n\n"
        "Write a complete, detailed response."
    )


def build_longwriter_single_turn(
    examples_per_level: int = 256,
    seed: int = 13,
    min_target_words: int = 4_000,
) -> None:
    """Create single-turn long-input partitions from LongWriter prompt/target pairs."""
    print("Building LongWriter single-turn long-input partitions...")

    rows = _load_longwriter_rows()
    context_pool = _build_context_pool(rows)
    rng = random.Random(seed)

    output_dir = _DATA_DIR / "longwriter_single_turn"
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = {
        "input_4k": 4_000,
        "input_6k": 6_000,
        "input_8k": 8_000,
        "input_10k": 10_000,
    }
    manifest = {
        "source": "data/longwriter_6k/train.jsonl",
        "construction": (
            "Single-turn prompts built by prepending LongWriter assistant outputs "
            "as reference material before an original LongWriter user request."
        ),
        "seed": seed,
        "examples_per_level": examples_per_level,
        "min_target_words": min_target_words,
        "partitions": {},
    }

    eligible = [
        row
        for row in rows
        if row.get("prompt", "").strip()
        and row.get("target", "").strip()
        and int(row.get("target_words", 0)) >= min_target_words
    ]
    if not eligible:
        raise ValueError(f"No LongWriter rows have target_words >= {min_target_words}.")
    for level_name, target_words in levels.items():
        level_eligible = [
            row
            for row in eligible
            if len(_make_single_turn_prompt(row["prompt"].strip(), "").split()) <= target_words
        ]
        selected = rng.sample(level_eligible, min(examples_per_level, len(level_eligible)))
        path = output_dir / f"{level_name}.jsonl"
        actual_words = []

        with open(path, "w") as f:
            for out_idx, row in enumerate(selected, start=1):
                base_prompt = row["prompt"].strip()
                overhead = len(_make_single_turn_prompt(base_prompt, "").split())
                needed_context_words = max(target_words - overhead, 0)

                context_parts = []
                while sum(len(part.split()) for part in context_parts) < needed_context_words:
                    context_parts.append(rng.choice(context_pool))
                context_text = _take_words(" ".join(context_parts), needed_context_words)
                prompt = _make_single_turn_prompt(base_prompt, context_text)
                prompt_words = len(prompt.split())
                actual_words.append(prompt_words)

                out = {
                    "id": out_idx,
                    "source": "longwriter_single_turn",
                    "level": level_name,
                    "base_id": row["id"],
                    "prompt": prompt,
                    "target": row["target"],
                    "prompt_words": prompt_words,
                    "target_words": row["target_words"],
                    "prompt_chars": len(prompt),
                    "target_chars": row["target_chars"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        manifest["partitions"][level_name] = {
            "path": str(path.relative_to(_DATA_DIR.parent)),
            "num_examples": len(selected),
            "target_input_words": target_words,
            "min_input_words": min(actual_words),
            "max_input_words": max(actual_words),
        }
        print(f"Saved {len(selected)} examples to {path}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")
    print("LongWriter single-turn build complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "humaneval", "longwriter", "longwriter-single-turn", "all"],
        default="gsm8k",
        help="Dataset to download",
    )
    parser.add_argument(
        "--examples-per-level",
        type=int,
        default=256,
        help="Examples per input-length level for longwriter-single-turn.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for derived dataset construction.",
    )
    parser.add_argument(
        "--min-target-words",
        type=int,
        default=4000,
        help="Minimum target words for longwriter-single-turn base examples.",
    )
    
    args = parser.parse_args()
    
    if args.dataset == "gsm8k":
        download_gsm8k()
    elif args.dataset == "humaneval":
        download_humaneval()
    elif args.dataset == "longwriter":
        download_longwriter()
    elif args.dataset == "longwriter-single-turn":
        build_longwriter_single_turn(
            args.examples_per_level,
            args.seed,
            args.min_target_words,
        )
    elif args.dataset == "all":
        download_gsm8k()
        download_humaneval()
        download_longwriter()
        build_longwriter_single_turn(
            args.examples_per_level,
            args.seed,
            args.min_target_words,
        )
