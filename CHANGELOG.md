# Changelog

## 2026-05-10

- Added a streaming draft-token API to `VLLMDraftGenerator`.
- Updated tree async branch pre-drafting to publish partial branch tokens while verification is running.
- Reused streamed branch snapshots when verification returns, avoiding waits for full fixed-K branch drafts.
- Added `streamed_branch_tokens` to per-round tree async metrics.
- Serialized draft model calls to avoid concurrent vLLM offline generation during streaming.
- Fixed streamed branch threads to keep per-round state isolated across tree async rounds.
- Verified changed Python files with `py_compile` and a lightweight smoke test.
