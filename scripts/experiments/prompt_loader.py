"""Prompt loader supporting GSM8K, HumanEval, and LongWriter datasets."""
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple


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
        source: Dataset name — "gsm8k", "humaneval", "longwriter",
            or "longwriter_single_turn:<partition>".
        count: Number of prompts to return.
        split: Dataset split ("train" or "test", gsm8k only).
        min_length: Minimum prompt text length in characters.
        max_length: Maximum prompt text length in characters.

    Returns:
        List of dicts with at least keys: id, text, source.
    """
    base_source, partition = _parse_source(source)

    if base_source == "gsm8k":
        prompts = _load_gsm8k(split)
    elif base_source == "humaneval":
        prompts = _load_humaneval()
    elif base_source == "longwriter":
        if partition:
            raise ValueError(
                "LongWriter raw partitions were removed. Use `longwriter` for "
                "the original data or `longwriter_single_turn:<partition>` for "
                "the derived long-input data."
            )
        prompts = _load_longwriter()
    elif base_source == "longwriter_single_turn":
        prompts = _load_longwriter_single_turn(partition)
    else:
        raise ValueError(
            f"Unknown source: {source}. Use: gsm8k, humaneval, longwriter, "
            "or longwriter_single_turn:<partition>"
        )

    filtered = [p for p in prompts if min_length <= len(p["text"]) <= max_length]

    # Random selection instead of taking first count
    selected = random.sample(filtered, min(count, len(filtered)))

    return [
        {"id": i + 1, "text": p["text"], "source": source, **_extra_fields(p)}
        for i, p in enumerate(selected)
    ]


def _parse_source(source: str) -> Tuple[str, Optional[str]]:
    if ":" not in source:
        return source, None
    base, partition = source.split(":", 1)
    return base, partition or None


def _extra_fields(prompt: Dict) -> Dict:
    return {k: v for k, v in prompt.items() if k not in {"text"}}


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


def _load_longwriter() -> List[Dict]:
    path = _DATA_DIR / "longwriter_6k" / "train.jsonl"

    if not path.exists():
        raise FileNotFoundError(
            f"LongWriter data not found at {path}. Run "
            "`python scripts/download_data.py --dataset longwriter` first."
        )

    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            prompts.append(
                {
                    "text": row["prompt"],
                    "target": row.get("target", ""),
                    "original_id": row.get("id"),
                    "prompt_chars": row.get("prompt_chars"),
                    "target_chars": row.get("target_chars"),
                    "prompt_words": row.get("prompt_words"),
                    "target_words": row.get("target_words"),
                    "prompt_bucket": row.get("prompt_bucket"),
                    "target_bucket": row.get("target_bucket"),
                }
            )
    return prompts


def _load_longwriter_single_turn(partition: Optional[str] = None) -> List[Dict]:
    if not partition:
        partition = "input_8k"
    path = _DATA_DIR / "longwriter_single_turn" / f"{partition}.jsonl"

    if not path.exists():
        available = _available_longwriter_single_turn_partitions()
        suffix = f" Available partitions: {', '.join(available)}." if available else ""
        raise FileNotFoundError(
            f"LongWriter single-turn data not found at {path}. Run "
            "`python scripts/download_data.py --dataset longwriter-single-turn` first."
            f"{suffix}"
        )

    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            prompts.append(
                {
                    "text": row["prompt"],
                    "target": row.get("target", ""),
                    "original_id": row.get("base_id"),
                    "prompt_chars": row.get("prompt_chars"),
                    "target_chars": row.get("target_chars"),
                    "prompt_words": row.get("prompt_words"),
                    "target_words": row.get("target_words"),
                    "prompt_bucket": row.get("level"),
                    "target_bucket": None,
                }
            )
    return prompts


def _available_longwriter_single_turn_partitions() -> List[str]:
    data_dir = _DATA_DIR / "longwriter_single_turn"
    if not data_dir.exists():
        return []
    return sorted(path.stem for path in data_dir.glob("input_*.jsonl"))
