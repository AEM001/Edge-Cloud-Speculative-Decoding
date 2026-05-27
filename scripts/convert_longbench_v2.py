#!/usr/bin/env python3
"""Convert LongBench v2 from Arrow format to partitioned JSONL files."""

import json
from pathlib import Path
from datasets import load_from_disk


def convert_longbench_v2():
    """Convert LongBench v2 dataset to partitioned JSONL files."""
    # Load the dataset
    dataset = load_from_disk('data/longbench_v2')
    train_data = dataset['train']
    
    # Create output directory
    output_dir = Path('data/longbench_v2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Partition by length
    partitions = {}
    for item in train_data:
        length = item['length']
        if length not in partitions:
            partitions[length] = []
        
        # Format for the prompt loader
        prompt_data = {
            'prompt': item['context'] + '\n\n' + item['question'],
            'answer': item['answer'],
            'original_id': item['_id'],
            'domain': item['domain'],
            'sub_domain': item['sub_domain'],
            'difficulty': item['difficulty'],
            'length': item['length'],
            'question': item['question'],
            'choices': {
                'A': item['choice_A'],
                'B': item['choice_B'],
                'C': item['choice_C'],
                'D': item['choice_D'],
            },
            'prompt_chars': len(item['context'] + '\n\n' + item['question']),
            'prompt_words': len((item['context'] + '\n\n' + item['question']).split()),
            'context_chars': len(item['context']),
            'context_words': len(item['context'].split()),
        }
        partitions[length].append(prompt_data)
    
    # Write partition files
    for length, items in partitions.items():
        output_path = output_dir / f"{length}.jsonl"
        with open(output_path, 'w') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')
        print(f"Written {len(items)} examples to {output_path}")
    
    # Also write a combined train file
    train_path = output_dir / "train.jsonl"
    all_items = []
    for items in partitions.values():
        all_items.extend(items)
    
    with open(train_path, 'w') as f:
        for item in all_items:
            f.write(json.dumps(item) + '\n')
    print(f"Written {len(all_items)} examples to {train_path}")
    
    print("\nConversion complete!")
    print(f"Available partitions: {', '.join(partitions.keys())}")


if __name__ == "__main__":
    convert_longbench_v2()