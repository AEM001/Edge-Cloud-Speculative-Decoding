The performance degradation with CUDA graph in your experiment vs my standalone test likely stems from **workload pattern differences**:

**My standalone test:**
- Simple `/generate` requests with consistent prompt length (247 chars)
- Single request pattern, same input shape each time
- CUDA graph can optimize this well

**Your experiment workload:**
- Verify server receives `/verify` requests with varying:
  - Prefix lengths (verify_prefix varies: 114, 132 tokens)
  - Draft lengths (draft tokens vary)
  - Input shapes change between rounds
- Speculative decoding creates **variable-length inputs** per verification round
- CUDA graph has **overhead for shape mismatches** - when input size doesn't match captured graph, it falls back to eager execution

**Core issue:** CUDA graph optimization requires **consistent input shapes**. Speculative decoding verification has naturally varying input lengths (prefix + draft changes each round), causing frequent graph misses and fallback to eager execution, which is slower than just staying in eager mode.

The verify server logs show CUDA graph capturing succeeded, but during actual speculative decoding, the varying input shapes likely trigger frequent graph recompilation or fallbacks, adding overhead that negates any benefits.

**Solution:** Keep `VERIFY_ENFORCE_EAGER=1` for verify server since speculative decoding's variable input patterns don't benefit from CUDA graph.