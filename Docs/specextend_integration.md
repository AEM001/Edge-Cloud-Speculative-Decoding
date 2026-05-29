# SpecExtend Edge-Cloud Integration

This repo now uses a custom Qwen3 backend for the active SpecExtend path. The
implementation is split between an edge draft backend and a cloud target
backend, connected by the existing HTTP protocol.

## Active Path

- `scripts/core/qwen_specextend_backend.py`
  - `QwenSpecExtendDraftBackend`: builds draft trees and maintains visible full
    and working cache token state for retrieval.
  - `QwenSpecExtendTargetBackend`: verifies draft trees, returns accepted path
    indices, correction tokens, and target attention scores.
- `scripts/client/specextend_edge_client.py`
  - Orchestrates draft tree construction, cloud verification, accepted-token
    commits, and retrieval updates.
- `scripts/server/verify_server.py`
  - Exposes `/specextend/verify`, `/generate`, `/verify`, and `/health`.

## Wire Contract

`SpecExtendTreeRequest` contains `request_id`, `prefix_ids`, `tree_input_ids`,
`tree_position_ids`, `parent_indices`, `tree_attention_mask`,
`retrieve_attn_scores`, `retrieval_chunk_size`, `retrieve_top_k`, and optional
`metadata`.

`SpecExtendTreeResponse` contains `request_id`, `accepted_len`,
`correction_token_id`, `accepted_tree_indices`, `server_verify_time_ms`,
optional `target_attn_scores`, optional `selected_chunk_ids`, and timing fields.

The edge reconstructs accepted token IDs from `tree_input_ids` and
`accepted_tree_indices`; the cloud returns indices and correction only.

## Correctness Checks

Validate in this order:

1. Protocol serialization round trip.
2. Retrieval chunk selection with fixed attention scores.
3. Draft-tree parent and attention-mask construction.
4. Target verification on a tiny Qwen-shaped model or a small local Qwen3.
5. Edge-cloud parity for one prompt through `/specextend/verify`.
6. `quick.sh` direct vs `specextend` smoke run.

Track acceptance length, generated tokens per round, edge draft time, cloud
verify time, network time, selected chunk IDs, retrieval update count, and
output text.

## Current Limitations

- Batch size is 1.
- The fast path uses linear draft chains by default. Set
  `DRAFT_TREE_MODE=branching` only for correctness experiments; branching still
  performs many draft forwards.
- The target and draft backends now maintain Transformers dynamic KV caches.
  Rejections rewind to the shared prefix instead of recomputing the whole prompt.
- Retrieval is active through cached last-token target attention. Start the
  verify server with `VERIFY_ATTN_IMPLEMENTATION=eager`; SDPA is faster but does
  not expose attention weights in Transformers. The cloud selects top-k chunk
  IDs from those scores by default and returns only the compact ID list; set
  request metadata `return_attention_scores=true` for full score vectors. The
  draft backend builds its sparse working context from selected retrieval chunks
  plus a recent suffix controlled by `DRAFT_RECENT_TOKENS`, preserving original
  token positions for RoPE.
- The edge client supports an asynchronous verification pipeline. While a draft
  tree is in flight to the cloud, the edge builds candidate next drafts at
  `SPECEXTEND_PIPELINE_OFFSETS` such as `full,half`. A candidate is reused only
  when the returned accept length and correction token make it an exact match.
- On a single V100 with Qwen3-8B target and Qwen3-1.7B draft, fully cached
  direct generation remains faster in current measurements. SpecExtend needs a
  cheaper draft, better acceptance, separate GPUs, or deeper tensor-level
  retrieval/KV compaction to beat the optimized direct baseline.
