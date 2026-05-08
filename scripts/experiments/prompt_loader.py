"""Prompt loader supporting GSM8K and HumanEval datasets."""
import json
from pathlib import Path
from typing import List, Dict


_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_prompts(
    source: str = "gsm8k",
    count: int = 5,
    split: str = "test",
    min_length: int = 0,
    max_length: int = 99999,
) -> List[Dict]:
    """
    Load prompts from a supported dataset.

    Args:
        source: Dataset name — "gsm8k" or "humaneval".
        count: Number of prompts to return.
        split: Dataset split ("train" or "test", gsm8k only).
        min_length: Minimum prompt text length in characters.
        max_length: Maximum prompt text length in characters.

    Returns:
        List of dicts with keys: id, text, source.
    """
    if source == "gsm8k":
        prompts = _load_gsm8k(split)
    elif source == "humaneval":
        prompts = _load_humaneval()
    else:
        raise ValueError(f"Unknown source: {source}. Use: gsm8k or humaneval")

    filtered = [p for p in prompts if min_length <= len(p["text"]) <= max_length]

    return [
        {"id": i + 1, "text": p["text"], "source": source}
        for i, p in enumerate(filtered[:count])
    ]


def _load_gsm8k(split: str = "test") -> List[Dict]:
    path = _DATA_DIR / "gsm8k" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"GSM8K data not found at {path}. Download it first.")
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            prompts.append({"text": row["question"]})
    return prompts


def _load_humaneval() -> List[Dict]:
    path = _DATA_DIR / "humaneval" / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"HumanEval data not found at {path}. Download it first.")
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            prompts.append({"text": row["prompt"]})
    return prompts
