"""Centralized prompt loader using data/prompt.json."""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# Default path to the prompt data file
DEFAULT_PROMPT_FILE = Path(__file__).parent / "data" / "prompt.json"


def load_prompts_from_json(
    prompt_file: Path = DEFAULT_PROMPT_FILE,
    count: Optional[int] = None,
    category: Optional[str] = None,
    prompt_type: Optional[str] = None
) -> List[Dict]:
   
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, "r") as f:
        data = json.load(f)
    
    prompts = []
    prompt_id = 1
    
    # Iterate through categories
    for cat_name, cat_prompts in data.items():
        if category is not None and cat_name != category:
            continue
        
        for prompt_data in cat_prompts:
            p_type = prompt_data.get("Type", "Unknown")
            if prompt_type is not None and p_type != prompt_type:
                continue
            
            prompts.append({
                "id": prompt_id,
                "text": prompt_data["Prompt"],
                "type": p_type,
                "category": cat_name,
                "s_no": prompt_data.get("S.No.", None),
                "length": len(prompt_data["Prompt"])
            })
            prompt_id += 1
    
    # Apply count limit if specified
    if count is not None:
        prompts = prompts[:count]
    
    return prompts


def load_prompts_by_type(
    count_per_type: int = 5,
    prompt_file: Path = DEFAULT_PROMPT_FILE
) -> Tuple[List[Dict], List[Dict]]:
    
    simple = load_prompts_from_json(prompt_file, count=count_per_type, prompt_type="Simple")
    complex = load_prompts_from_json(prompt_file, count=count_per_type, prompt_type="Complex")
    
    # Re-index prompts starting from 1
    for i, p in enumerate(simple, 1):
        p["id"] = i
    for i, p in enumerate(complex, 1):
        p["id"] = i
    
    return simple, complex


def load_prompts_by_length(
    count: int = 5,
    target_length: int = 300,
    prompt_file: Path = DEFAULT_PROMPT_FILE
) -> List[Dict]:
    
    all_prompts = load_prompts_from_json(prompt_file, count=None)
    
    # Sort by absolute difference from target length
    sorted_prompts = sorted(
        all_prompts,
        key=lambda p: abs(p["length"] - target_length)
    )
    
    selected = sorted_prompts[:count]
    
    # Re-index prompts starting from 1
    for i, p in enumerate(selected, 1):
        p["id"] = i
    
    return selected
