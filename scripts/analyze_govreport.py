#!/usr/bin/env python3
"""Analyze length distribution of GovReport dataset."""

import json
from pathlib import Path
from collections import Counter

_DATA_DIR = Path(__file__).parent.parent / "data"
GOVREPORT_PATH = _DATA_DIR / "govreport" / "test.jsonl"

def analyze_length_distribution():
    lengths = []
    with open(GOVREPORT_PATH) as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            lengths.append(len(text))
    
    lengths.sort()
    
    print(f"Total samples: {len(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {sum(lengths) / len(lengths):.0f}")
    print(f"Median length: {lengths[len(lengths) // 2]}")
    
    # Percentiles
    for p in [10, 25, 50, 75, 90, 95, 99]:
        idx = int(len(lengths) * p / 100)
        print(f"{p}th percentile: {lengths[idx]}")
    
    # Bucket distribution
    buckets = [0, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, float('inf')]
    bucket_counts = Counter()
    for length in lengths:
        for i in range(len(buckets) - 1):
            if buckets[i] <= length < buckets[i+1]:
                bucket_counts[f"{buckets[i]}-{buckets[i+1]}"] += 1
                break
    
    print("\nLength bucket distribution:")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {bucket}: {count}")

if __name__ == "__main__":
    analyze_length_distribution()
