# Changelog

## 2026-05-11

08:55
- Added `scripts/experiments/analyze_quick_results.py` to analyze raw quick-test `quick_test_results.json` and `quick_test_rounds.jsonl` records directly.
- Generated raw-data analysis CSV tables and PNG charts under `scripts/experiments/outputs_quick/analysis/`.
- Wrote `scripts/experiments/outputs_quick/analysis/report.md` manually from inspected raw tables/charts, including throughput, prompt-type, acceptance, tree-offset, and verifier-scaling observations.
- Noted that the raw run contains completed `good` and `medium` records only; bursty entries in the summary are zero-filled and excluded from the report.

09:06
- Added `scripts/experiments/analyze_acceptance_dynamics.py` to study raw acceptance-length dynamics separately from the main summary analysis.
- Generated acceptance-dynamics charts/tables for state timelines, dead-zone clustering, transition probabilities, prefix-length effects, round-position drift, and middle-acceptance lengths.
- Extended the raw-data report with acceptance-dynamics observations: dead zones cluster by prompt/phase, recent acceptance is predictive, prefix length alone is not a reliable acceptance predictor, and middle acceptance should be handled as conditional branch-policy value.

- Updated quick-test analysis to derive Tree Async diagnostics from raw `quick_test_results.json` / `quick_test_rounds.jsonl` records instead of relying only on aggregated `quick_test_summary.json`.
- Reframed the Tree Async objective as using the edge draft pipeline to absorb cloud verification + RTT wait, rather than saying draft time disappears.
- Added raw pipeline diagnostics for `base_draft_ms`, `branch_draft_ms`, `total_wait_ms`, `exposed_branch_ms`, `prefetched_tokens`, and `selected_offset`.
- Fixed the time-breakdown chart/report so Tree draft time is taken from raw branch draft timing instead of near-zero exposed tail time.
- Found from raw slot results that the previous K=8/B=3 offset heuristic effectively used speculative offsets `[8, 4]`, while offset `4` was never selected.
- Changed the Tree Async branch-offset heuristic to prioritize the full-acceptance branch and the strongest short-acceptance mode, making K=8/B=3 use speculative offsets `[8, 1]`.
- Verified the updated tree async client with `py_compile` and a direct `_tree_offsets(8, 8)` check.

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

22:57 
- back to the two 4090s, very successful
- used the randomly picking prompts
- for the first time, the medium network is working!
- the tree based-asynchronous way is significently beating the vanilla way! 
- so fucking good, I'm so happy for this.
