# Experiment Run 2 Analysis: K=6, K=8, K=10

## Executive Summary

**Unexpected Result:** All speculative decoding configurations are 10-15x SLOWER than direct generation.

- Direct: ~67 tok/s
- K=6/8/10: ~4-6 tok/s

## Detailed Results

### Throughput Comparison

| Method | Easy Prompts (tok/s) | Hard Prompts (tok/s) | Speedup vs Direct |
|--------|---------------------|---------------------|-------------------|
| Direct | 67.1 | 67.8 | 1.0x |
| K=6 | 4.3 | 6.0 | 0.07x |
| K=8 | 5.1 | 6.1 | 0.08x |
| K=10 | 5.1 | 4.1 | 0.07x |

### Acceptance Rate Analysis

| K | Easy Avg | Hard Avg | Range |
|---|----------|----------|-------|
| 6 | ~8% | ~12% | 2% - 20% |
| 8 | ~25% | ~15% | 7% - 69% |
| 10 | ~9% | ~8% | 4% - 15% |

**Critical Threshold:** Theory requires α > 60% for speedup. Actual: 8-25%.

### Timing Breakdown (per round)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Draft (edge) | 170-240 | 25-35% |
| Verify (server) | 120-130 | 15-20% |
| Network RTT | 190-210 | 25-30% |
| Network overhead | 70-80 | 10-15% |

**Key Insight:** Network time (RTT + overhead) consumes 35-45% of total time per round.

## Root Cause Analysis

### 1. Draft-Target Mismatch
- **Draft:** Qwen2.5-3B (small, fast)
- **Target:** Qwen2.5-32B-GPTQ-Int4 (quantized, massive)
- **Gap:** 10x size difference with quantization distortion
- **Result:** Distribution divergence → low acceptance

### 2. Network Latency
- HTTP-based communication
- RTT ~200ms per round
- 60-100 rounds per request
- Cumulative network time: 12-20 seconds

### 3. Protocol Overhead
- Full token sequence transmitted each round
- No compression or delta encoding
- JSON serialization overhead

## Recommendations

### Immediate (Code Improvements)

1. **Add server_pure_inference_ms metric**
   - Current: verify_ms includes network
   - Need: separate server compute time from RTT

2. **Record per-token acceptance position**
   - Current: only overall acceptance rate
   - Need: distribution of rejection positions

3. **Add draft confidence histogram**
   - Track draft model's probability distribution
   - Correlate with acceptance rate

### System Level

1. **Draft Model Selection**
   - Use larger draft model (7B instead of 3B)
   - Or use same family with less quantization
   - Consider distillation from target

2. **Network Optimization**
   - Switch to gRPC/HTTP2 for persistent connection
   - Enable compression
   - Consider WebSocket for lower latency

3. **Adaptive K Selection**
   - Current: static K values
   - Need: dynamic based on recent acceptance rate
   - Formula: K_opt = f(α_recent, RTT, draft_speed)

4. **Batching**
   - Current: single request per round
   - Potential: batch multiple prompts

## Next Experiment Design

Based on this analysis, next experiment should:

1. **Test fewer K values:** Focus on K=4, K=6, K=8 (skip K=10)
2. **Reduce prompts:** 10 easy + 10 hard is sufficient
3. **Add new metrics:**
   - server_pure_inference_ms
   - draft_confidence_avg
   - rejection_position_avg
4. **Test with better draft model** if available

## Data Quality

All 80 tests completed successfully.
- 10 easy prompts × 4 methods = 40 tests
- 10 hard prompts × 4 methods = 40 tests
- Total time: ~35 minutes
- No failures or timeouts

## Raw Data

- Full results: `comprehensive_results.json`
- Log: `comprehensive_log.txt`
