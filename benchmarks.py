"""Select and organize benchmark prompts from datasets for experiments."""
import json
from pathlib import Path
from typing import List, Dict, Any
import random

# Configuration
TARGET_PROMPT_LENGTH = 512  # tokens (approximate)
TARGET_COMPLETION_LENGTH = 128  # tokens
NUM_BENCHMARKS = 100  # Total number of prompts to select
NUM_EASY = 50  # Number of easy prompts
NUM_HARD = 50  # Number of hard prompts

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


def classify_difficulty(entry: Dict) -> str:
    """Classify prompt as easy or hard based on content and source."""
    source = entry["source"]
    user_msg = entry["user_message"].lower()
    prompt_length = entry["prompt_length"]
    
    # Bespoke-Stratos is explicitly hard reasoning
    if source == "bespoke_stratos":
        return "hard"
    
    # ConversationChronicles - classify by content
    hard_keywords = [
        "solve", "calculate", "prove", "determine", "algorithm",
        "python", "code", "function", "math", "equation",
        "complex", "explain why", "step by step", "reasoning"
    ]
    
    easy_keywords = [
        "hello", "hi", "how are you", "what is your name",
        "thank you", "good", "nice", "simple", "basic"
    ]
    
    # Check for hard indicators
    for keyword in hard_keywords:
        if keyword in user_msg:
            return "hard"
    
    # Check for easy indicators
    for keyword in easy_keywords:
        if keyword in user_msg:
            return "easy"
    
    # Default: classify by length (longer = harder)
    if prompt_length > 800:
        return "hard"
    elif prompt_length < 300:
        return "easy"
    else:
        return "easy"  # Default to easy for mid-length


def select_benchmarks_by_difficulty(datasets: Dict[str, List[Dict]]) -> List[Dict]:
    """Select benchmarks separated by difficulty."""
    # Classify all prompts
    classified = {"easy": [], "hard": []}
    
    for source, samples in datasets.items():
        for sample in samples:
            difficulty = classify_difficulty(sample)
            sample["difficulty"] = difficulty
            classified[difficulty].append(sample)
    
    print(f"  Easy candidates: {len(classified['easy']):,}")
    print(f"  Hard candidates: {len(classified['hard']):,}")
    
    # Select from each difficulty
    def filter_and_select(samples: List[Dict], target_count: int, target_length: int) -> List[Dict]:
        """Filter by length and select random samples."""
        # Sort by closeness to target length
        sorted_samples = sorted(
            samples,
            key=lambda x: abs(x["prompt_length"] - target_length)
        )
        # Take top candidates closest to target
        candidates = sorted_samples[:target_count * 3]
        # Randomly select from candidates
        random.shuffle(candidates)
        return candidates[:target_count]
    
    selected_easy = filter_and_select(classified["easy"], NUM_EASY, TARGET_PROMPT_LENGTH)
    selected_hard = filter_and_select(classified["hard"], NUM_HARD, TARGET_PROMPT_LENGTH)
    
    selected = selected_easy + selected_hard
    random.shuffle(selected)
    
    print(f"  Selected: {len(selected_easy)} easy, {len(selected_hard)} hard")
    
    return selected


def truncate_to_length(text: str, max_chars: int) -> str:
    """Truncate text to approximate character limit."""
    # Rough approximation: 1 token ≈ 4 characters
    max_chars = min(max_chars, len(text))
    return text[:max_chars]


def tidy_text(text: str) -> str:
    """Tidy text by removing extra whitespace and normalizing."""
    # Remove leading/trailing whitespace
    text = text.strip()
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    # Normalize newlines
    text = text.replace("\n\n\n", "\n\n")
    return text


def format_benchmark_entry(entry: Dict, index: int) -> Dict:
    """Format a benchmark entry with fixed lengths and tidied text."""
    # Tidy the text first
    user_msg = tidy_text(entry["user_message"])
    system_msg = tidy_text(entry["system"])
    
    # Truncate to target length
    user_msg = truncate_to_length(user_msg, TARGET_PROMPT_LENGTH * 4)
    system_msg = truncate_to_length(system_msg, TARGET_PROMPT_LENGTH * 2)
    
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
        "difficulty": entry.get("difficulty", "unknown"),
        "system_prompt": system_msg,
        "user_prompt": user_msg,
        "formatted_prompt": "\n".join(parts),
        "target_prompt_length": TARGET_PROMPT_LENGTH,
        "target_completion_length": TARGET_COMPLETION_LENGTH,
        "actual_prompt_length": len(user_msg),
    }


def save_benchmarks(benchmarks: List[Dict]):
    """Save benchmarks to file, separated by difficulty."""
    # Separate by difficulty
    easy_benchmarks = [b for b in benchmarks if b["difficulty"] == "easy"]
    hard_benchmarks = [b for b in benchmarks if b["difficulty"] == "hard"]
    
    # Save combined
    output_file = BENCHMARK_DIR / "prompts.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(benchmarks)} benchmarks to {output_file}")
    
    # Save by difficulty
    for difficulty, subset in [("easy", easy_benchmarks), ("hard", hard_benchmarks)]:
        diff_file = BENCHMARK_DIR / f"prompts_{difficulty}.json"
        with open(diff_file, "w", encoding="utf-8") as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(subset)} {difficulty} benchmarks to {diff_file}")
    
    # Save text version for easy viewing
    text_file = BENCHMARK_DIR / "prompts.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(f"BENCHMARK PROMPTS\n")
        f.write(f"Total: {len(benchmarks)} (Easy: {len(easy_benchmarks)}, Hard: {len(hard_benchmarks)})\n")
        f.write(f"Target prompt length: {TARGET_PROMPT_LENGTH} tokens\n")
        f.write(f"Target completion length: {TARGET_COMPLETION_LENGTH} tokens\n")
        f.write(f"{'='*70}\n\n")
        
        # Group by difficulty
        for difficulty in ["easy", "hard"]:
            subset = easy_benchmarks if difficulty == "easy" else hard_benchmarks
            f.write(f"\n{'='*70}\n")
            f.write(f"{difficulty.upper()} BENCHMARKS ({len(subset)})\n")
            f.write(f"{'='*70}\n\n")
            
            for i, b in enumerate(subset[:10]):  # Show first 10 of each
                f.write(f"[{i+1}] {b['id']}\n")
                f.write(f"Source: {b['source']}\n")
                f.write(f"Original ID: {b['original_id']}\n")
                f.write(f"Actual length: {b['actual_prompt_length']} chars\n")
                f.write(f"{'='*70}\n")
                f.write(f"System: {b['system_prompt'][:150]}...\n\n")
                f.write(f"User: {b['user_prompt'][:300]}...\n\n")
    
    print(f"✅ Saved text version to {text_file}")
    
    # Save summary
    summary = {
        "total_benchmarks": len(benchmarks),
        "target_prompt_length": TARGET_PROMPT_LENGTH,
        "target_completion_length": TARGET_COMPLETION_LENGTH,
        "by_difficulty": {
            "easy": len(easy_benchmarks),
            "hard": len(hard_benchmarks),
        },
        "by_source": {
            "bespoke_stratos": sum(1 for b in benchmarks if b["source"] == "bespoke_stratos"),
            "conversation_chronicles": sum(1 for b in benchmarks if b["source"] == "conversation_chronicles"),
        },
        "avg_actual_length": sum(b["actual_prompt_length"] for b in benchmarks) / len(benchmarks),
        "avg_easy_length": sum(b["actual_prompt_length"] for b in easy_benchmarks) / len(easy_benchmarks) if easy_benchmarks else 0,
        "avg_hard_length": sum(b["actual_prompt_length"] for b in hard_benchmarks) / len(hard_benchmarks) if hard_benchmarks else 0,
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
    
    # Select benchmarks by difficulty
    print(f"\nSelecting {NUM_BENCHMARKS} benchmarks ({NUM_EASY} easy, {NUM_HARD} hard)...")
    selected = select_benchmarks_by_difficulty(datasets)
    benchmarks = [format_benchmark_entry(entry, i) for i, entry in enumerate(selected)]
    
    # Save
    save_benchmarks(benchmarks)
    
    print("\nDone!")
