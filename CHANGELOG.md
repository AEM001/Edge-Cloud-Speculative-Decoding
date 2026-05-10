# Changelog

## 2026-05-10

- Updated hardware configuration from 2x RTX 4090 to 2x RTX 3090.
- Changed draft model from Qwen2.5-3B-Instruct-AWQ to Qwen2.5-1.5B-Instruct-AWQ.
- Changed verify model from Qwen2.5-14B-Instruct-AWQ to Qwen2.5-32B-Instruct-AWQ.
- Updated config.py, README.md, and verify_server.py to reflect new model paths and hardware.
- Sharded the Qwen2.5-32B verifier across both RTX 3090 GPUs by default with vLLM tensor parallel size 2.
- Reduced verifier GPU memory utilization to 0.55 per GPU and draft GPU memory utilization to 0.25 so the 1.5B draft model can coexist on GPU 1.
- Added configurable `DRAFT_GPU_ID`, `VERIFY_TENSOR_PARALLEL_SIZE`, prefix-caching, and eager-mode environment defaults.
- Defaulted the verify server to FLASH_ATTN, prefix caching enabled, and eager mode disabled for CUDA graph execution.
- Added quick-test logging for `verify_ms` and simulated network time separately from end-to-end RTT.


- Added a streaming draft-token API to `VLLMDraftGenerator`.
- Updated tree async branch pre-drafting to publish partial branch tokens while verification is running.
- Reused streamed branch snapshots when verification returns, avoiding waits for full fixed-K branch drafts.
- Added `streamed_branch_tokens` to per-round tree async metrics.
- Serialized draft model calls to avoid concurrent vLLM offline generation during streaming.
- Fixed streamed branch threads to keep per-round state isolated across tree async rounds.
- Fixed prefetched tree base drafts to top up partial streamed reuse back to the policy `K` before verification.
- Analyzed quick-test results: vanilla `K=7` accepted lengths are bimodal rather than smooth, with about half of rounds accepting all 7 tokens and about one third accepting 0-2 tokens.
- Changed tree branch offsets to prioritize the full-acceptance offset first, then a lower/mid acceptance anchor, instead of centering only on the mean acceptance-ratio estimate.
- Verified changed Python files with `py_compile` and a lightweight smoke test.

22:14
- the cuda_graph is making it very slow, need to fix problems
- and temporarily using only the good network
- make the verify into two GPUs
