"""Select and organize benchmark prompts from datasets for experiments."""
import json
from pathlib import Path
from typing import List, Dict, Any
import random

# Configuration
TARGET_PROMPT_LENGTH = 512  # tokens (approximate)
TARGET_COMPLETION_LENGTH = 128  # tokens
NUM_BENCHMARKS = 20  # Number of prompts to select

DATA_DIR = Path("data")
BENCHMARK_DIR = Path("benchmarks")
BENCHMARK_DIR.mkdir(exist_ok=True)


def load_datasets() -> Dict[str, List[Dict]]:
    """Load both datasets."""
    datasets = {}
    
    # Load Bespoke-Stratos
    with open(DATA_DIR / "bespoke_stratos.json") as f:
        datasets["bespoke_stratos"] = json.load(f)
    
    # Load ConversationChronicles
    with open(DATA_DIR / "conversation_chronicles.json") as f:
        datasets["conversation_chronicles"] = json.load(f)
    
    return datasets


def select_benchmarks(datasets: Dict[str, List[Dict]], num_samples: int = NUM_BENCHMARKS) -> List[Dict]:
    """Select diverse benchmarks from both datasets."""
    selected = []
    
    # Mix from both datasets
    bs_samples = datasets["bespoke_stratos"]
    cc_samples = datasets["conversation_chronicles"]
    
    # Randomly sample from each
    random.shuffle(bs_samples)
    random.shuffle(cc_samples)
    
    # Select roughly half from each
    bs_count = num_samples // 2
    cc_count = num_samples - bs_count
    
    # Filter for appropriate length
    def filter_by_length(samples: List[Dict], target: int, count: int) -> List[Dict]:
        """Filter samples close to target length."""
        # Sort by closeness to target
        sorted_samples = sorted(
            samples,
            key=lambda x: abs(x["prompt_length"] - target)
        )
        return sorted_samples[:count]
    
    selected_bs = filter_by_length(bs_samples, TARGET_PROMPT_LENGTH, bs_count)
    selected_cc = filter_by_length(cc_samples, TARGET_PROMPT_LENGTH, cc_count)
    
    selected.extend(selected_bs)
    selected.extend(selected_cc)
    
    # Shuffle final selection
    random.shuffle(selected)
    
    return selected


def truncate_to_length(text: str, max_chars: int) -> str:
    """Truncate text to approximate character limit."""
    # Rough approximation: 1 token ≈ 4 characters
    max_chars = min(max_chars, len(text))
    return text[:max_chars]


def format_benchmark_entry(entry: Dict, index: int) -> Dict:
    """Format a benchmark entry with fixed lengths."""
    # Truncate to target length
    user_msg = truncate_to_length(entry["user_message"], TARGET_PROMPT_LENGTH * 4)
    system_msg = truncate_to_length(entry["system"], TARGET_PROMPT_LENGTH * 2)
    
    # Rebuild formatted prompt
    parts = []
    if system_msg:
        parts.append(f"<|im_start|>system\n{system_msg}<|im_end|>")
    if user_msg:
        parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    
    return {
        "id": f"benchmark_{index}",
        "source": entry["source"],
        "original_id": entry["id"],
        "system_prompt": system_msg,
        "user_prompt": user_msg,
        "formatted_prompt": "\n".join(parts),
        "target_prompt_length": TARGET_PROMPT_LENGTH,
        "target_completion_length": TARGET_COMPLETION_LENGTH,
        "actual_prompt_length": len(user_msg),
    }


def save_benchmarks(benchmarks: List[Dict]):
    """Save benchmarks to file."""
    output_file = BENCHMARK_DIR / "prompts.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(benchmarks)} benchmarks to {output_file}")
    
    # Also save a text version for easy viewing
    text_file = BENCHMARK_DIR / "prompts.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(f"BENCHMARK PROMPTS\n")
        f.write(f"Total: {len(benchmarks)}\n")
        f.write(f"Target prompt length: {TARGET_PROMPT_LENGTH} tokens\n")
        f.write(f"Target completion length: {TARGET_COMPLETION_LENGTH} tokens\n")
        f.write(f"{'='*70}\n\n")
        
        for i, b in enumerate(benchmarks):
            f.write(f"[{i+1}] {b['id']}\n")
            f.write(f"Source: {b['source']}\n")
            f.write(f"Original ID: {b['original_id']}\n")
            f.write(f"Actual length: {b['actual_prompt_length']} chars\n")
            f.write(f"{'='*70}\n")
            f.write(f"System: {b['system_prompt'][:200]}...\n\n")
            f.write(f"User: {b['user_prompt'][:400]}...\n\n")
            f.write(f"Formatted (truncated):\n{b['formatted_prompt'][:300]}...\n\n")
    
    print(f"✅ Saved text version to {text_file}")
    
    # Save summary
    summary = {
        "total_benchmarks": len(benchmarks),
        "target_prompt_length": TARGET_PROMPT_LENGTH,
        "target_completion_length": TARGET_COMPLETION_LENGTH,
        "by_source": {
            "bespoke_stratos": sum(1 for b in benchmarks if b["source"] == "bespoke_stratos"),
            "conversation_chronicles": sum(1 for b in benchmarks if b["source"] == "conversation_chronicles"),
        },
        "avg_actual_length": sum(b["actual_prompt_length"] for b in benchmarks) / len(benchmarks),
    }
    
    with open(BENCHMARK_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary: {summary}")


if __name__ == "__main__":
    print("="*70)
    print("BENCHMARK PROMPT SELECTION")
    print("="*70)
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = load_datasets()
    print(f"  Bespoke-Stratos: {len(datasets['bespoke_stratos']):,} prompts")
    print(f"  ConversationChronicles: {len(datasets['conversation_chronicles']):,} prompts")
    
    # Select benchmarks
    print(f"\nSelecting {NUM_BENCHMARKS} benchmarks...")
    benchmarks = [format_benchmark_entry(entry, i) for i, entry in enumerate(select_benchmarks(datasets))]
    
    # Save
    save_benchmarks(benchmarks)
    
    print("\nDone!")
