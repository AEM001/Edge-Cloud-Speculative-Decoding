"""Prompt loader supporting GSM8K, HumanEval, LongWriter, and LongBench v2."""
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
            "longwriter_single_turn:<partition>", "longbench_v2:<partition>",
            "govreport", "pg19", or "prompts_2048".
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
    elif base_source == "longbench_v2":
        prompts = _load_longbench_v2(partition)
    elif base_source == "govreport":
        prompts = _load_govreport(split)
    elif base_source == "pg19":
        prompts = _load_pg19(split)
    elif base_source == "prompts_2048":
        prompts = _load_prompts_2048()
    else:
        raise ValueError(
            f"Unknown source: {source}. Use: gsm8k, humaneval, longwriter, "
            "longwriter_single_turn:<partition>, longbench_v2:<partition>, "
            "govreport, pg19, or prompts_2048"
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


def _load_longbench_v2(partition: Optional[str] = None) -> List[Dict]:
    if not partition:
        partition = "short"
    path = _DATA_DIR / "longbench_v2" / f"{partition}.jsonl"

    if not path.exists():
        available = _available_longbench_v2_partitions()
        suffix = f" Available partitions: {', '.join(available)}." if available else ""
        raise FileNotFoundError(
            f"LongBench v2 data not found at {path}. Run "
            "`python scripts/download_data.py` first."
            f"{suffix}"
        )

    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            prompt_text = row["prompt"]
            # Truncate prompt to ~1000 characters to fit within model max_len
            max_chars = 1000
            if len(prompt_text) > max_chars:
                prompt_text = prompt_text[:max_chars]
            prompts.append(
                {
                    "text": prompt_text,
                    "target": row.get("answer", ""),
                    "original_id": row.get("original_id"),
                    "domain": row.get("domain"),
                    "sub_domain": row.get("sub_domain"),
                    "difficulty": row.get("difficulty"),
                    "length": row.get("length"),
                    "question": row.get("question"),
                    "choices": row.get("choices"),
                    "prompt_chars": min(len(prompt_text), row.get("prompt_chars", len(prompt_text))),
                    "prompt_words": row.get("prompt_words"),
                    "context_chars": row.get("context_chars"),
                    "context_words": row.get("context_words"),
                    "prompt_bucket": row.get("length"),
                    "target_bucket": None,
                }
            )
    return prompts


def _available_longwriter_single_turn_partitions() -> List[str]:
    data_dir = _DATA_DIR / "longwriter_single_turn"
    if not data_dir.exists():
        return []
    return sorted(path.stem for path in data_dir.glob("input_*.jsonl"))


def _available_longbench_v2_partitions() -> List[str]:
    data_dir = _DATA_DIR / "longbench_v2"
    if not data_dir.exists():
        return []
    return sorted(path.stem for path in data_dir.glob("*.jsonl"))


def _load_govreport(split: str = "test") -> List[Dict]:
    path = _DATA_DIR / "govreport" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"GovReport data not found at {path}. Run "
            "`python scripts/download_govreport_pg19.py --dataset govreport` first.")
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            text = row["text"]
            # ~8000 chars ≈ 2048 tokens for English text
            text = text[:8000]
            prompts.append({"text": text, "summary": row.get("summary", "")})
    return prompts


def _load_pg19(split: str = "test") -> List[Dict]:
    path = _DATA_DIR / "pg19" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"PG-19 data not found at {path}. Run "
            "`python scripts/download_govreport_pg19.py --dataset pg19` first.")
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            text = row["text"]
            # ~8000 chars ≈ 2048 tokens for English text
            text = text[:8000]
            prompts.append({"text": text, "title": row.get("title", "")})
    return prompts


def _load_prompts_2048() -> List[Dict]:
    path = _DATA_DIR / "prompts" / "prompts_2048.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Prompts 2048 data not found at {path}.")
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            prompts.append({"text": row["prompt"]})
    return prompts
