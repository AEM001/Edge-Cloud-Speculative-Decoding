# Changelog

## 2026-05-10

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
