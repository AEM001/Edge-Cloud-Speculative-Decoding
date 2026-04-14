# Experiment Report

**Date**: 2026-04-14  
**Config**: 3B Draft (Qwen2.5-3B) + 32B Target (Qwen2.5-32B-GPTQ-Int4)  
**Tests**: 50 easy + 50 hard prompts × 4 methods = 400 tests

## Results Summary

| Method | Easy (tok/s) | Hard (tok/s) | Avg Acceptance |
|--------|-------------|--------------|----------------|
| Direct | **69.8** | **69.3** | - |
| K=2 | 5.4 (0.08×) | 5.8 (0.08×) | 47.8% |
| K=4 | 5.7 (0.08×) | 5.9 (0.08×) | 23.4% |
| K=6 | 5.3 (0.08×) | 6.2 (0.09×) | 19.6% |

## Key Finding

**Speculative Decoding is ~10× slower than Direct** due to high network latency per round.

## Charts

- `outputs/comprehensive_main.png` - Main comparison
- `outputs/comprehensive_detailed.png` - Detailed analysis

## Raw Data

- `outputs/comprehensive_results.json`
- `outputs/comprehensive_log.txt`
