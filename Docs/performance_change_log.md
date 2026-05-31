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
  plus the recent suffix, matching the core `full_draft_kv -> draft_stable_kv`
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
- Validation so far is code-level only: `py_compile` and
  `tests/test_specextend_core.py` pass. No performance run has been executed for
  this change yet.
