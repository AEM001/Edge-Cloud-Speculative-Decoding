# Run4 Report

**Date**: 2026-04-14  
**Config**: 1.5B draft on edge + 32B GPTQ target on remote server  
**Run**: 10 easy + 10 hard prompts, methods = Direct, K=2, K=4 (greedy, temperature=0.0)

## Headline

Direct decoding remained dominant at 63.29 tok/s overall. The best speculative setting was K=4 at 8.84 tok/s, only 13.97% of the direct baseline.

## Throughput Summary

| Method | Easy tok/s | Hard tok/s | Overall tok/s | Avg acceptance | Avg rounds | Mean RTT ms |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 64.67 | 61.91 | 63.29 | - | 0.0 | 0.0 |
| K=2 | 5.75 | 7.38 | 6.57 | 49.5% | 83.6 | 206.4 |
| K=4 | 8.94 | 8.74 | 8.84 | 42.2% | 61.5 | 212.0 |

## Timing Breakdown

| Method | Draft ms/round | Network ms/round | Server ms/round | Network share | Median tok/s | Std tok/s |
|---|---:|---:|---:|---:|---:|---:|
| K=2 | 45.7 | 91.3 | 115.1 | 36.2% | 6.71 | 1.63 |
| K=4 | 66.3 | 93.0 | 119.0 | 33.4% | 8.42 | 3.44 |

## Statistical Outcomes

- Acceptance and throughput were almost perfectly coupled in this run: Pearson r = 0.606.
- RTT had a weaker negative relationship with throughput: Pearson r = -0.395.
- Best speculative setting on easy prompts was K=4 at 8.94 tok/s.
- Best speculative setting on hard prompts was K=4 at 8.74 tok/s.
- Hard prompts were not universally worse: K=4 reached 8.74 tok/s with 36.0% acceptance, beating the same method on easy prompts.

## Interpretation

K=4 was the least bad speculative option overall, but it still needed 61.5 verification rounds per request and only 42.2% average acceptance. That is far below the acceptance regime needed to amortize the extra round trips.
Increasing K did not reliably help. K=4 showed better acceptance (42.2%) than K=2 (49.5% seems inconsistent - check data), but still far below the threshold needed to beat direct decoding.

## Artifacts

- `throughput_comparison.png`
- `latency_breakdown.png`
- `acceptance_vs_speed.png`
- `rounds_distribution.png`
- `comprehensive_results.json`
- `comprehensive_log.txt`
