# Run3 Report

**Date**: 2026-04-14  
**Config**: 3B draft on edge + 32B GPTQ target on remote server  
**Run**: 10 easy + 10 hard prompts, methods = Direct, K=6, K=8, K=10

## Headline

Direct decoding remained dominant at 68.47 tok/s overall. The best speculative setting was K=8 at 5.50 tok/s, only 8.04% of the direct baseline.

## Throughput Summary

| Method | Easy tok/s | Hard tok/s | Overall tok/s | Avg acceptance | Avg rounds | Mean RTT ms |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 69.66 | 67.28 | 68.47 | - | 0.0 | 0.0 |
| K=6 | 4.38 | 6.02 | 5.20 | 16.4% | 80.4 | 201.1 |
| K=8 | 5.04 | 5.97 | 5.50 | 17.0% | 76.0 | 208.5 |
| K=10 | 5.11 | 3.95 | 4.53 | 11.6% | 80.6 | 220.3 |

## Timing Breakdown

| Method | Draft ms/round | Network ms/round | Server ms/round | Network share | Median tok/s | Std tok/s |
|---|---:|---:|---:|---:|---:|---:|
| K=6 | 164.3 | 77.2 | 124.0 | 21.1% | 4.77 | 2.39 |
| K=8 | 204.4 | 81.2 | 127.4 | 19.7% | 3.89 | 3.89 |
| K=10 | 245.4 | 88.2 | 132.2 | 18.9% | 3.44 | 3.41 |

## Statistical Outcomes

- Acceptance and throughput were almost perfectly coupled in this run: Pearson r = 0.988.
- RTT had a weaker negative relationship with throughput: Pearson r = -0.169.
- Best speculative setting on easy prompts was K=10 at 5.11 tok/s.
- Best speculative setting on hard prompts was K=6 at 6.02 tok/s.
- Hard prompts were not universally worse: K=6 reached 6.02 tok/s with 22.3% acceptance, beating the same method on easy prompts.
- The log recorded 4 transient request retries, which indicates tunnel or proxy instability but not total experiment failure.

## Interpretation

K=8 was the least bad speculative option overall, but it still needed 76.0 verification rounds per request and only 17.0% average acceptance. That is far below the acceptance regime needed to amortize the extra round trips.
Increasing K did not reliably help. K=10 raised draft cost, increased RTT exposure, and degraded acceptance enough to become the worst overall speculative choice.

## Artifacts

- `throughput_comparison.png`
- `latency_breakdown.png`
- `acceptance_vs_speed.png`
- `rounds_distribution.png`
- `comprehensive_results.json`
- `comprehensive_log.txt`
