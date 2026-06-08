"""Shared helpers for guidance reuse experiment scripts."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_ROOT = REPO_ROOT / "outputs" / "guidance_reuse"
DEFAULT_QUICK_CONFIG = REPO_ROOT / "quick_benchmark_config.json"


def timestamp_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def parse_int_list(value: str) -> List[int]:
    items = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            items.append(int(raw))
    if not items:
        raise ValueError("expected at least one integer")
    return items


def method_interval(method: str, profile: Optional[Dict[str, Any]] = None) -> Optional[int]:
    if profile and "retrieve_every_n_steps" in profile:
        try:
            return int(profile["retrieve_every_n_steps"])
        except (TypeError, ValueError):
            pass
    match = re.search(r"(?:reuse|fixed)_R(\d+)", method)
    if match:
        return int(match.group(1))
    if "no_update" in method:
        return 0
    return None


def selected_ids_from_round(round_detail: Dict[str, Any]) -> List[int]:
    ids = round_detail.get("selected_chunk_ids") or []
    return [int(v) for v in ids]


def jaccard(left: Sequence[int], right: Sequence[int]) -> Optional[float]:
    a = set(left)
    b = set(right)
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def acceptance_ratio(accepted: float, rounds: float, draft_length: float) -> Optional[float]:
    denom = rounds * draft_length
    if denom <= 0:
        return None
    return accepted / denom


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    vx = sum(x * x for x in dx)
    vy = sum(y * y for y in dy)
    if vx <= 0 or vy <= 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / ((vx * vy) ** 0.5)


def latest_quick_result() -> Optional[Path]:
    candidates = sorted((REPO_ROOT / "outputs").glob("quick_test_results_*.json"))
    return candidates[-1] if candidates else None


def result_profile(result: Dict[str, Any]) -> Dict[str, Any]:
    return ((result.get("raw") or {}).get("profile") or {})


def result_rounds(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(((result.get("raw") or {}).get("round_details") or []))


def prompt_key(result: Dict[str, Any]) -> Tuple[str, int]:
    return (str(result.get("prompt_type", "?")), int(result.get("prompt_id", 0)))

