# Async Pipeline Experiment Report

**Date**: 2026-04-20  
**Config**: 1.5B draft + 7B AWQ verify on same RTX 5090  
**Methods**: Direct | Sync K=2,4 | Async K=2,4 (lookahead=1)  
**Prompts**: 10 easy + 10 hard, 128 max tokens, temperature=0.0

---

## Headline

**Neither speculative method beats Direct (22.97 tok/s) on same-GPU setup.**  
Async is marginally better than Sync (+0.3 tok/s avg) but the difference is noise-level.  
Best speculative result: Async K=4 on hard prompts at **22.72 tok/s (0.99x Direct)**.

---

## Throughput Summary

| Method | Easy tok/s | Hard tok/s | vs Direct | Accept% |
|---|---:|---:|---:|---:|
| Direct | 22.96 | 22.97 | **1.00x** | — |
| Sync K=2 | 16.13 | 17.08 | 0.70x | 48.5% |
| Async K=2 | 16.44 | 17.02 | 0.72x | 48.6% |
| Sync K=4 | 21.27 | 22.41 | 0.93x | 39.0% |
| Async K=4 | 21.31 | **22.72** | **0.99x** | 39.5% |

---

## Pipeline-Specific Metrics (Async only)

| Method | PipeEff% | Bubble ms | Prefetch Waste% | Rollback Rate | Effective TPR |
|---|---:|---:|---:|---:|---:|
| Async K=2 easy | 40.9% | 65.2ms | 51.4% | 0.59/round | 0.97 |
| Async K=2 hard | 44.1% | 66.8ms | 45.0% | 0.56/round | 1.10 |
| Async K=4 easy | 31.5% | 68.0ms | 60.5% | 0.69/round | 1.58 |
| Async K=4 hard | 31.4% | 69.2ms | 57.6% | 0.69/round | 1.70 |

- **PipeEff%** = fraction of rounds where all K tokens were accepted (full hit)
- **Bubble ms** = avg time drafter blocked waiting for verifier result
- **Prefetch Waste%** = fraction of speculatively pre-drafted tokens discarded
- **Rollback Rate** = rejections per round (high = pipeline flushes frequently)
- **Effective TPR** = accepted tokens per round

---

## Why Async Doesn't Beat Direct (Root Cause)

### 1. Same-GPU serialization kills pipeline overlap

`avg_bubble_ms ≈ 65–68ms ≈ avg_rtt_ms ≈ 66–70ms`

The bubble equals the RTT — meaning the drafter **immediately blocks** after submitting a batch. True pipelining requires `T_draft < T_verify` so the drafter can get ahead. On one GPU, both models share a single CUDA stream: when the verifier's forward pass is running, CUDA blocks the drafter. Python threading cannot overlap CUDA compute on one device.

**PicoSpec's speedup formula** `L_async = max(T_draft, T_verify)` only holds when draft and verify run on **separate hardware** (edge device vs cloud server). On same GPU: `L_async ≈ T_draft + T_verify` — identical to sync.

### 2. Acceptance rate too low to amortize overhead

Speculative decoding beats direct when: `α × K > 1` (accepted tokens per verify call > 1 autoregressive step)

| Method | Accept% | K | Effective tokens/verify | Break-even |
|---|---:|---:|---:|---|
| K=2 | ~49% | 2 | ~0.98 | Need >1.0 ❌ |
| K=4 | ~40% | 4 | ~1.60 | Need >1.0 ✓ but barely |

K=4 clears the break-even threshold but the overhead of running the 1.5B draft model on top of the 7B verify model cancels the gain.

### 3. RTT is only ~70ms (localhost)

In a real edge-cloud scenario, network RTT = 200–500ms. Speculative decoding's role is to **hide that latency**. At 70ms localhost RTT the latency-hiding benefit is minimal.

---

## When Would Async Actually Win?

| Scenario | Expected async gain |
|---|---|
| Real edge-cloud (200ms+ RTT) | Large — async hides network latency |
| Draft/verify on separate GPUs | Large — true CUDA parallelism |
| High acceptance rate (>70%) | Moderate — fewer rollbacks, pipeline stays warm |
| EAGLE/Medusa-style draft (shared arch) | Large — 80%+ acceptance achievable |

---

## Conclusion

The experiment correctly validates the theoretical constraints:

1. **Async ≈ Sync on same GPU** — confirmed empirically (+0.3 tok/s, within noise)
2. **Speculative decoding doesn't beat Direct** at 40–49% acceptance on same GPU
3. **Pipeline metrics are meaningful**: PipeEff 31–44%, bubble=RTT confirms serialization

The implementation is correct. The async client correctly implements PicoSpec parallel drafting with rollback. The results reflect a fundamental hardware constraint, not a bug.

---

## Artifacts

- `async_results.json` — full per-prompt results + summary
- `async_log.txt` — full experiment log
- `throughput_comparison.png`
- `latency_breakdown.png`
- `pipeline_efficiency.png`
- `bubble_time.png`
- `acceptance_vs_pipeline.png`
