# Performance Issue: Draft Model Too Slow

## Problem
Speculative decoding is **5x slower** than direct server calls on current hardware.

## Root Cause
- **Draft generation**: 227ms per round (K=2)
- **Server verify**: 45ms per round
- **Total**: 273ms per round
- **Expected speedup**: 0.20x (slower, not faster!)

## Why?
Draft model (Qwen2.5-0.5B-Instruct-4bit on MLX) is too slow:
- 0.5B parameters with 4-bit quantization
- Running on Mac GPU
- Already using KV cache (mlx_lm.generate)
- Inherent hardware limitation

## Baseline Comparison
- Direct server generation: ~6s for 128 tokens
- Speculative decoding: ~29s for 128 tokens

## Conclusion
**Speculative decoding requires draft model to be much faster than target model.**
Current setup: draft (227ms) > server (45ms) → no benefit.

## Next Steps

### Option 1: Try 2-bit Quantization
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-2bit-mlx`
- Expected: ~2x faster than 4-bit
- Risk: Lower quality, worse acceptance rate
- Worth testing: Yes

### Option 2: Accept Current Setup
- Document as experimental finding
- Focus on algorithm correctness, not speed
- Run small-scale experiment (3-5 prompts)

### Option 3: Skip Draft Model
- Direct server calls only
- Faster for current hardware
- Loses speculative decoding benefits
