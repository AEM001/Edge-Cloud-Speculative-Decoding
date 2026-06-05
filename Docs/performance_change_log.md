# Performance Change Log

## 2026-05-29

- Added PG-19 support to `load_prompts(source="pg19")` and populated
  `data/pg19/test.jsonl`.
- Added fixed input-token truncation with `--prompt-input-tokens`; PG-19 runs
  now use exactly 2048 tokenizer input tokens when configured.
- Replaced direct target generation's token-by-token full-prefix forwards with
  `model.generate(..., use_cache=True)`.
- Reworked linear target verification to score a whole draft chain with cached
  KV state instead of recomputing the full prefix for every draft token.
- Added cache rewind on rejection, preserving the shared prefix cache and
  replaying only the last shared token when the speculative path diverges.
- Switched normal model loading to SDPA attention by default. Eager attention is
  no longer required for the fast verification path.
- Replaced full-sequence retrieval attention with cached last-token attention,
  avoiding large attention tensors during retrieval updates.
- Made `start_verify.sh` default to `VERIFY_ATTN_IMPLEMENTATION=eager` so
  target-attention retrieval is actually available. SDPA remains useful for
  no-retrieval timing runs but does not return attentions in Transformers.
- Added Qwen3-0.6B model alias and `fast-draft` preset.
- Added retrieval-backed draft working context: selected retrieval chunks plus a
  configurable recent suffix via `DRAFT_RECENT_TOKENS`.
- Updated `quick.sh` defaults for PG-19, 2048 input tokens, 256 output tokens,
  Qwen3-1.7B draft, SDPA attention, and retrieval every 8 rounds.
- Added an asynchronous edge pipeline. Cloud verification runs in a background
  worker while the draft backend builds candidate next drafts at configurable
  accept offsets (`SPECEXTEND_PIPELINE_OFFSETS`, default `full,half`). Reuse is
  exact only: accept length must match and the cloud correction token must equal
  the first prefetched token.

## 2026-05-30 01:44

- Switched runtime Python from `.venv` to system `miniconda3` base environment;
  installed missing dependencies (`transformers`, `uvicorn`, `fastapi`,
  `autoawq`, etc.) with network acceleration.
- Changed verify server and quick-test scripts to use `python3` directly instead
  of `.venv/bin/python3`.
- Separated launch scripts: `start_verify.sh` starts the verify server,
  `run_quick.sh` runs the experiment only (assumes server already up),
  `run_2x3090.sh` orchestrates both. Default port changed to `6008` to avoid
  conflicts.
- Raised `RETRIEVE_EVERY_N_STEPS` default from `4` to `16`.
- **Fixed sparse KV cache rebuild on every round** (`ensure_sparse_cache` +
  `generate_token_ids_cached_sparse`): after each decode pass the draft tokens
  appended to `_sparse_cache` are now cropped out, leaving only the context
  prefill KV. On the next round only newly accepted recent-window tokens are
  forwarded incrementally instead of the whole sparse sequence being re-prefilled
  from scratch.
- Added stale-tail crop in `ensure_sparse_cache`: when the retrieved-chunk set
  changes, the common prefix of the sparse cache is preserved via `DynamicCache.crop`
  instead of a full reset.

## 2026-05-30 02:01 — Measurements (2× RTX 3090, Qwen3-14B-AWQ target @ GPU 0 / Qwen3-1.7B draft @ GPU 1)

Environment: two RTX 3090, Qwen3-14B-AWQ target (GPU 0, eager attention),
Qwen3-1.7B draft (GPU 1, SDPA), PG-19 prompt, 2048 input tokens, 256 output
tokens, nodes=32, depth=8, retrieve_every_n_steps=16.

- Direct (14B-AWQ): **18.6 s, 13.7 tok/s**
- SpecExtend: **10.6 s, 24.4 tok/s** — **1.78× faster than direct**
  - local draft time: 1.8 s (down from 18.9 s before sparse-cache fix, −90%)
  - server verify time: 5.1 s
  - rounds: 37, accepted draft tokens: 223/256, acceptance length: 6.03

## Measurements

Environment: one Tesla V100-SXM2-32GB, Qwen3-8B target, PG-19 prompt,
2048 tokenizer input tokens, 256 output tokens.

- Optimized direct 8B baseline: about 12.2s, 20.9 tok/s.
- Qwen3-0.6B draft, draft length 4, retrieval every 8 rounds:
  about 35.2s, 7.3 tok/s; acceptance length about 1.25.
- Qwen3-1.7B draft, draft length 8, retrieval every 8 rounds:
  about 21.1s, 12.2 tok/s; acceptance length about 4.0.
- Qwen3-1.7B draft, draft length 16, retrieval every 8 rounds:
  about 73.3s, 3.5 tok/s; acceptance length about 1.27.
- Qwen3-1.7B draft, draft length 8, retrieval every 8 rounds, async offsets
  `full,half`: about 43.7s, 5.9 tok/s; acceptance length about 2.39. The
  pipeline hid response wait time, but single-GPU target/draft contention and
  low candidate reuse made it slower in this environment.
- Qwen3-1.7B draft, draft length 8, real target-attention retrieval every 2
  rounds with eager verify attention, async offsets `full,half`, 64 output
  tokens: about 15.2s, 4.35 tok/s; 11 retrieval rounds, selected chunks changed
  from target attention, acceptance length about 1.75.

Conclusion: the backend is now functional and much faster than the original
full-prefix implementation, but on one V100 the optimized direct baseline is
still faster than speculative decoding. The speculative path needs separate
draft/target GPUs, higher draft-target agreement, or tensor-level retrieval KV
compaction before it can beat cached direct generation for this setup.

## 2026-05-31 20:33 — Code Change, Not Yet Benchmarked

- Replaced the draft-side sparse replay cache with a SpecExtend-style tensor KV
  cache implementation. The draft backend now owns a full draft KV cache and
  rebuilds the active working KV cache by indexing selected retrieval chunks
  from the retrieval state, matching the core `full_draft_kv -> draft_stable_kv`
  pattern from the original SpecExtend implementation.
- Added local Qwen3 KV model support inside this repo:
  `scripts/core/modeling_qwen3_kv.py` and `scripts/core/qwen_kv_cache.py`.
  The draft backend no longer imports custom model/cache code from sibling
  repos.
- Changed the Qwen draft backend to load the local custom `Qwen3ForCausalLM`
  path for draft generation so the cache tensors are visible and mutable by the
  SpecExtend cache manager. The target verifier path still uses the existing
  verifier runtime.
- Disabled async pipeline candidate prefetch for this draft backend because
  speculative prefixes are not yet committed into the full draft KV cache; using
  them would violate the exact cache ownership/update discipline.
- Removed the prior `DRAFT_RECENT_TOKENS` recent-window union. The working cache
  is now driven by the same chunk selection semantics as SpecExtend: initial
  selected chunks are chosen by the retrieval state, later retrieval updates
  replace that selected chunk set, and the draft cache does not independently
  append a recent-token suffix.
- Validation so far is code-level only: `py_compile` and
  `tests/test_specextend_core.py` pass. No performance run has been executed for
  this change yet.

## 2026-06-01 00:21 — Working Cache Rebuild Optimization + Pipeline Candidate Infrastructure

### Problem diagnosed
After the 2026-05-31 refactor to the explicit full-draft-kv + retrieval working-cache
architecture, benchmarks showed a severe regression:

- `local_draft_ms`: 18,092 ms (was 1,813 ms before refactor, ×10 slower)
- Acceptance length: 2.51 / 8 (was 6.03 / 8)
- Throughput: 8.33 tok/s (was 24.42 tok/s)
- Rounds: 73 (was 37)

Root causes identified by comparing against the sparse-kv reference
(`specextend/application/model_classic.py`):

1. **`_rebuild_working_cache` re-allocated GPU tensors on every call** via
   `_allocate_kv_cache()`, creating new `KVCache` objects and zeroing large
   buffers each round. This is the single largest overhead.

2. **`select_working_tokens` rebuilt the working cache every round** even when
   the retrieval chunk selection had not changed. The comparison included newly
   appended verified-token indices from `append_prefix`, which are always different
   from the incoming retrieval chunk indices, triggering an unnecessary rebuild
   after every verification step.

3. **`build_draft_tree` called `append_prefix` before `select_working_tokens`**,
   so the retrieval rebuild always happened after the new tokens were appended,
   wasting the KV written by `append_prefix` into working cache.

4. **`_generate_linear_from_working_cache` finally block called
   `_rebuild_working_cache`** (expensive full index_select) just to undo the
   draft token KV appended during generation.

### Changes made

- **`_rebuild_working_cache`: in-place reuse of existing buffer.** Removed the
  `_allocate_kv_cache()` call inside `_rebuild_working_cache`. The method now
  writes directly into the pre-allocated `working_cache` tensors via
  `index_select + copy_` in-place, and updates `current_length` only. No GPU
  memory is allocated on the hot path.

- **`select_working_tokens`: skip rebuild when chunk selection is unchanged.**
  Added a comparison that strips the newly-appended verified-token tail from
  `working_token_indices` before comparing against incoming retrieval indices.
  If the chunk portion is identical, only `working_token_indices` is reassigned;
  no `_rebuild_working_cache` call is made.

- **Reordered `build_draft_tree`: `select_working_tokens` before
  `append_prefix`.** Mirrors the sparse-kv pattern
  (`update_working_cache_retrieval_main` → incremental forward of new tokens).
  Retrieval rebuild (if needed) happens on the old working cache, then
  `append_prefix` does a single incremental forward of only the new verified
  tokens on top of the correctly-selected working cache.

- **`_generate_linear_from_working_cache` finally: snapshot/restore
  `current_length` only.** Replaced the `working_token_indices` reset +
  `_rebuild_working_cache` call with `_snapshot_working_lengths` /
  `_restore_working_lengths`, which only fills scalar CPU tensors. The KV data
  in the buffer is not touched; stale draft-token KV past the restored length
  boundary is invisible to the model because `KVCache` uses `current_length` to
  control the active window.

- **`build_draft_candidate` implemented** (async pipeline prefetch). Snapshots
  working cache lengths, temporarily forwards candidate prefix tokens onto the
  working cache, generates a draft tree, then fully restores the cache state
  (lengths, `full_token_ids`, `working_token_indices`, `last_prefix_logits`).
  Does not write to `full_draft_kv`, so the main cache is unaffected.

- **`supports_pipeline_candidates` kept `False`** for now. Benchmarking with
  `True` showed the candidate forward runs on the same GPU (cuda:1) as the main
  draft forward. Because `_build_pipeline_candidates` is called synchronously
  on the main thread after submitting the verify future, the candidate and the
  next-round draft are serialized on GPU1, adding latency instead of hiding it.
  The implementation is in place for future evaluation on a setup where the
  candidate can be offloaded to a separate thread/process.

### Measurements (2026-06-01, same hardware as 2026-05-30 02:01)

Environment: 2× RTX 3090, Qwen3-14B-AWQ target (GPU 0, eager), Qwen3-1.7B draft
(GPU 1, SDPA), PG-19, 2048 input tokens, 256 output tokens, nodes=32, depth=8,
chunk_size=64, top_k=16, retrieve_every_n_steps=16.

| | Before (broken, 06-01 00:08) | After (06-01 00:19) | Best prior (05-30) |
|---|---|---|---|
| Throughput | 8.33 tok/s | **12.84 tok/s** | 24.42 tok/s |
| Total time | 30,715 ms | **20,174 ms** | 10,648 ms |
| local_draft_ms | 18,092 ms | **11,482 ms** | 1,813 ms |
| Rounds | 73 | **49** | 37 |
| Acceptance length | 2.51 / 8 | **4.29 / 8** | 6.03 / 8 |

The working-cache rebuild fixes alone reduced draft time by 37% and improved
acceptance rate significantly. The remaining gap to the 05-30 best is due to
the working cache having ~1024 tokens of past KV per round (retrieval chunks),
making each draft attention pass materially more expensive than the 05-30
sparse replay cache which only maintained a small recent-token DynamicCache.
That trade-off is inherent to the full-draft-kv + retrieval working-cache design
and matches the sparse-kv reference architecture's cost model.

## 2026-06-01 00:40 — Async Pipeline + DynamicCache Investigation (Reverted)

### Attempts and findings

**Attempt 1: Async pipeline with two-executor parallel execution.**
Submitted verify to one `ThreadPoolExecutor` and candidate drafts to a second executor
simultaneously. Result: 4.34 tps (worse than 12.84 tps baseline). Root cause: both
executors mapped onto the same GPU (cuda:1). Python's GIL and CUDA stream serialization
mean two threads issuing CUDA kernels on the same device do not run in parallel — the
candidate forward runs after the main draft, adding latency instead of hiding it. The
`threading.RLock` added to protect the shared working_cache also caused contention.
Reverted to single-executor design where candidate runs synchronously on main thread
while verify awaits network response.

**Attempt 2: Fixed-size working cache (trim verified-token tail after append_prefix).**
After each `append_prefix`, reset `working_cache.current_length` back to `chunk_size`,
keeping working cache at a constant ~1024-token size across rounds. Result: 6.15 tps,
acceptance 1.44/8 (much worse). Root cause: standard causal attention requires continuous
context. Trimming drops the recently-verified-token KV from the working cache, so the
draft model cannot attend to the last few accepted tokens. In sparse-kv this works because
their custom CUDA kernel supports non-contiguous position IDs (sparse attention); the
standard `scaled_dot_product_attention` does not. Reverted.

**Attempt 3: Replace KVCache with DynamicCache as working_cache.**
Hypothesis: custom `KVCache` attention path in `modeling_qwen3_kv.py` is slower than
the native SDPA path used with `DynamicCache`. Investigation showed:
- `Qwen3SdpaAttention` already supports `DynamicCache` via `isinstance(past_key_value, Cache)` branch (line 245).
- Both paths use `scaled_dot_product_attention`.
- `DynamicCache.update()` does not re-apply RoPE; KV is stored post-RoPE in both cases.
- Switching to `DynamicCache` required truncating the cache in `select_working_tokens`
  early-return path (dropping verified-token tail), which reproduced the same context
  loss as Attempt 2. Result: 8.96 tps, acceptance 2.50/8. Reverted to KVCache.

### Root cause of the remaining gap vs 05-30 best

The 05-30 best (24.42 tps, 49 ms/round) used a sparse **replay** cache
(`_sparse_cache`, `DynamicCache`): only accepted tokens were forwarded each round,
so past_kv grew by ~5 tokens per round (accepted length). Working cache was tiny and
grew linearly — still fast at 256 output tokens.

Current architecture: working_cache = chunk_tokens (~1024) + verified_tokens (growing).
Each draft forward does attention over ~1024+ tokens. At 234 ms/round this is unavoidable
given that:
- Qwen3-1.7B with 1024-token past_kv does ~200–250 ms/forward on one RTX 3090.
- Verify server (14B AWQ) takes ~350 ms/round server-side.
- Total latency per round ≈ draft_ms + verify_ms ≈ 234 + 350 = 584 ms (serialized).
- Effective tps = (accepted_len * 1000) / round_ms ≈ (4.29 * 1000) / 584 ≈ 7.3 tok/s
  ... but actual is 12.84 because pipeline partially hides verify latency.

### What would actually close the gap

1. **Flash attention / sliding window on the draft model**: reduce per-token attention
   cost in the 1024-token working cache. `flash_attn` is not installed in this env.
2. **Batched draft forward (multiple candidate tokens at once)**: instead of token-by-token
   draft generation, forward all 32 draft positions in one batched call (tree mask).
   This amortizes the 1024-token attention over 32 tokens instead of 32 separate calls.
3. **Reduce working cache size without hurting acceptance**: requires non-contiguous
   position ID support (sparse attention kernel), which is what sparse-kv provides.
   Not feasible with standard transformers without custom CUDA.
4. **Async pipeline across requests**: only beneficial if verify RTT >> draft time.
   Currently draft ≈ verify so pipeline provides minimal gain.

### Current code state

- `supports_pipeline_candidates = False` (pipeline implemented but disabled: overhead
  exceeds benefit at current draft/verify latency ratio).
- `build_draft_candidate` is implemented and correct; can be enabled when draft time
  drops significantly (e.g. after flash attention or batched drafting).
- Working cache remains KVCache (custom pre-allocated buffer, in-place index_select).
- Best measured throughput: **12.84 tok/s** (06-01 00:19 run).

## 2026-06-01 00:05 — SpecExtend KV Semantics Cleanup, Not Yet Benchmarked

- Tightened the draft-side cache path to avoid fallback behavior that was not
  faithful to SpecExtend. Removed the inactive sparse replay cache code
  (`_sparse_cache`, `ensure_sparse_cache`, and
  `generate_token_ids_cached_sparse`) so the draft backend must use the explicit
  full draft KV cache plus retrieval-selected working KV cache.
- Fixed retrieval chunk growth semantics to match SpecExtend more closely:
  when appending verified tokens creates a new chunk, that new chunk is appended
  to the selected working chunk set. Existing selected tail chunks are still
  refreshed as their end position grows.
- Fixed draft seeding after working-cache rebuild. The first draft-token logits
  now come from the logits produced while forwarding the newly appended verified
  prefix/correction token into the draft model, matching the original
  SpecExtend flow, instead of recomputing from the last selected working-cache
  token.
- Removed the approximate `DRAFT_TREE_MODE=branching` implementation because it
  recomputed paths from token ids rather than drafting through the explicit
  working KV cache. The backend now raises if non-linear tree mode is requested
  until branching is implemented on top of the strict KV path.
- Validation so far is code-level only: `py_compile` and
  `tests/test_specextend_core.py` pass. No performance run has been executed for
  this cleanup yet.

## 2026-06-05 — Chunk-Level Working Cache Selection

### Motivation
`SpecExtendDraftKVCache` previously accepted a flat list of token indices
(`retrieval_token_indices`) from the edge client to rebuild the working KV cache.
This was different from the original sparse-kv reference, which manages chunks
internally and only accepts chunk IDs from the caller. Moving to chunk-level
selection makes the draft cache manager responsible for chunk bookkeeping,
keeps the chunk→token expansion logic in one place, and aligns the draft
backend with the sparse-kv architecture.

### Changes made

- **`SpecExtendDraftKVCache`**: added internal chunk bookkeeping fields
  (`chunk_size`, `chunks`, `selected_chunks`).

- **Replaced `select_working_tokens(indices)` with `select_chunks(chunk_ids)`**.
  The method now accepts a list of chunk IDs, looks up the corresponding
  `RetrievalChunk` objects, and expands them into `working_token_indices` via
  `_update_working_indices_from_chunks`. If `chunk_ids` is `None`, all chunks
  are selected (full cache).

- **Added `_update_chunks_from_full()`**: rebuilds the chunk list from
  `full_token_ids` after every `append_prefix`, returning `True` when new
  chunks were created.

- **Added `_refresh_selected_tail()`**: keeps the last selected chunk in sync
  with the latest full-cache tail, matching sparse-kv's behavior.

- **`append_prefix` now maintains chunk bookkeeping automatically**.
  On first call it selects all chunks; on subsequent calls it appends newly
  created tail chunks to the selected set and refreshes the tail chunk end
  position before rebuilding the working cache.

- **Updated `build_draft_tree` / `build_draft_candidate` signatures** in
  `QwenSpecExtendDraftBackend` and the `SpecExtendDraftBackend` Protocol:
  parameter renamed from `retrieval_token_indices` to `retrieval_chunk_ids`.

- **Updated `specextend_edge_client.py`**: the client now passes
  `self.retrieval.selected_chunk_ids()` instead of
  `self.retrieval.selected_token_indices()` to the draft backend.

- **`position_ids` semantics are unchanged**: `context()` still returns
  `working_token_indices` (original global positions) as `position_ids`,
  preserving the correct RoPE encoding that distinguishes this backend from
  sparse-kv's local-position approach.

- **Also fixed `quick_test.py` warmup call** to use the new parameter name.

### Validation so far

Code-level checks pass (`py_compile`, existing tests). No performance benchmark
has been run yet for this refactor.
