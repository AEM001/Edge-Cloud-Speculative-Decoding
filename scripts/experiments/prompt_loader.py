"""Prompt loader using SPEED-Bench dataset qualitative split."""
from pathlib import Path
from typing import List, Dict, Optional


def load_speed_bench_prompts(
    count: int = 5,
    category: str = "coding",
    multiturn: bool = False,
    data_path: Optional[Path] = None
) -> List[Dict]:
    """
    Load prompts from SPEED-Bench qualitative split.

    Actual dataset columns: question_id, category, sub_category, turns, source, src_id, difficulty, multiturn

    Args:
        count: Number of prompts to load
        category: Category to filter (default: "coding")
        multiturn: Whether to include multiturn samples (default: False for single-turn)
        data_path: Optional local path to SPEED-Bench qualitative split. If None, loads from HuggingFace.

    Returns:
        List of prompt dictionaries with keys: id, text, category, multiturn, src_id, question_id
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library is required. Install with: pip install datasets")

    # Load from local path if provided, otherwise from HuggingFace
    if data_path and data_path.exists():
        dataset = load_dataset(str(data_path), "qualitative")
    else:
        # Use token from hf.txt if available
        hf_token_file = Path(__file__).parent.parent / "models" / "hf.txt"
        token = None
        if hf_token_file.exists():
            token = hf_token_file.read_text().strip()
        dataset = load_dataset("nvidia/SPEED-Bench", "qualitative", token=token)

    # Filter for category and multiturn (difficulty is None in actual data)
    filtered = dataset['test'].filter(
        lambda x: x['category'] == category and x['multiturn'] == multiturn
    )

    # Convert to list of dictionaries
    prompts = []
    for i, item in enumerate(filtered):
        if i >= count:
            break
        # Get the first turn's prompt
        prompt_text = item['turns'][0] if item['turns'] and len(item['turns']) > 0 else ""
        prompts.append({
            "id": i + 1,
            "text": prompt_text,
            "category": item['category'],
            "multiturn": item['multiturn'],
            "src_id": item['src_id'],
            "question_id": item['question_id']
        })

    return prompts
