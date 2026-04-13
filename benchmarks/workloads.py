"""Workload management for benchmarks."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_prompts_from_file(filepath: Path) -> List[str]:
    """
    Load prompts from a JSON file.
    
    Args:
        filepath: Path to JSON file containing prompts
    
    Returns:
        List of prompt strings
    """
    if not filepath.exists():
        logger.warning(f"Prompt file not found: {filepath}")
        return []
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract formatted prompts
    prompts = []
    for item in data:
        if 'formatted_prompt' in item:
            prompts.append(item['formatted_prompt'])
        elif 'prompt' in item:
            prompts.append(item['prompt'])
    
    logger.info(f"Loaded {len(prompts)} prompts from {filepath}")
    return prompts


def load_workloads(
    easy_file: str = None,
    hard_file: str = None
) -> Dict[str, List[str]]:
    """
    Load easy and hard workload prompts.
    
    Args:
        easy_file: Path to easy prompts file (relative to project root or absolute)
        hard_file: Path to hard prompts file (relative to project root or absolute)
    
    Returns:
        Dictionary with 'easy' and 'hard' keys containing prompt lists
    """
    # Try multiple locations for prompt files
    base_dir = Path(__file__).parent.parent
    
    # Default paths
    if easy_file is None:
        easy_candidates = [
            base_dir / "benchmarks" / "prompts_easy.json",
            base_dir.parent / "ubuntu-verify" / "prompts_easy.json",
            Path("prompts_easy.json"),
        ]
        easy_path = None
        for candidate in easy_candidates:
            if candidate.exists():
                easy_path = candidate
                break
        if easy_path is None:
            logger.error(f"Could not find prompts_easy.json in any of: {easy_candidates}")
            easy_path = easy_candidates[0]
    else:
        easy_path = Path(easy_file)
    
    if hard_file is None:
        hard_candidates = [
            base_dir / "benchmarks" / "prompts_hard.json",
            base_dir.parent / "ubuntu-verify" / "prompts_hard.json",
            Path("prompts_hard.json"),
        ]
        hard_path = None
        for candidate in hard_candidates:
            if candidate.exists():
                hard_path = candidate
                break
        if hard_path is None:
            logger.error(f"Could not find prompts_hard.json in any of: {hard_candidates}")
            hard_path = hard_candidates[0]
    else:
        hard_path = Path(hard_file)
    
    workloads = {
        'easy': load_prompts_from_file(easy_path),
        'hard': load_prompts_from_file(hard_path)
    }
    
    return workloads
