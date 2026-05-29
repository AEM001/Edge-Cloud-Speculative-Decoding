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
