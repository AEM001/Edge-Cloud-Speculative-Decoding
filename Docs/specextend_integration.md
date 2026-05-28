# SpecExtend Edge-Cloud Integration

This document describes how the full SpecExtend method should fit into this
edge-cloud repository. It is written as an implementation contract: the current
vLLM quick path remains runnable, while the full SpecExtend path is deliberately
separated until a custom model backend is available.

## What Is Runnable Now

The runnable baseline is:

- edge draft model: vLLM `TokensPrompt` generation
- cloud target model: vLLM `/verify` and `/generate`
- speculative protocol: linear `prefix_ids + draft_ids`
- verification: greedy top-1 match using `prompt_logprobs=1`
- experiments: `quick.sh` and `scripts/experiments/quick_test.py`

This is useful for measuring edge-cloud round trips, vLLM prefix caching, HTTP
overhead, and local proactive pre-drafting. It is not full SpecExtend.

## Why Full SpecExtend Needs Another Backend

SpecExtend depends on model internals that vLLM does not expose through the
current offline `LLM.generate` interface:

- target-side tree attention masks
- accepted tree-path verification, not only linear draft verification
- target last-layer attention scores for retrieval
- edge-side full draft KV cache and selected working KV cache
- explicit cache position/index manipulation after accepted paths are chosen

For this reason, `/specextend/verify` exists but returns `501` on the current
vLLM server. That endpoint is a capability boundary, not a fake implementation.

## New Code Boundaries

The SpecExtend-facing code is intentionally backend-agnostic.

`scripts/core/protocol.py`

- `SpecExtendTreeRequest`: cloud verification request for a whole draft tree.
- `SpecExtendTreeResponse`: verification result plus retrieval feedback.

`scripts/core/specextend_backend.py`

- `SpecExtendDraftBackend`: edge model contract for building a draft tree.
- `SpecExtendTargetBackend`: cloud model contract for verifying that tree.

`scripts/core/specextend_retrieval.py`

- Tracks retrieval chunks over the full draft cache.
- Selects chunks by mean target attention score.
- Returns token indices that a real draft backend can use to materialize its
  working KV cache.

`scripts/client/specextend_edge_client.py`

- Coordinates draft tree construction.
- Sends `SpecExtendTreeRequest` to the cloud.
- Commits accepted tree-path tokens plus correction token.
- Updates retrieval state from target attention scores or selected chunk IDs.

`scripts/server/verify_server.py`

- Keeps the current `/verify` and `/generate` vLLM endpoints.
- Adds `/specextend/verify` with an explicit `501` on the current backend.
- Reports `runtime.specextend_tree_verify=false` in `/health`.

## Wire Contract

`SpecExtendTreeRequest` contains:

- `request_id`
- `prefix_ids`
- `tree_input_ids`
- `tree_position_ids`
- `parent_indices`
- `tree_attention_mask`
- `retrieve_attn_scores`
- `retrieval_chunk_size`
- `retrieve_top_k`
- optional `metadata`

`SpecExtendTreeResponse` contains:

- `request_id`
- `accepted_len`
- `correction_token_id`
- `accepted_tree_indices`
- `server_verify_time_ms`
- optional `target_attn_scores`
- optional `selected_chunk_ids`
- optional timing fields

The edge reconstructs accepted token IDs from `tree_input_ids` and
`accepted_tree_indices`. The cloud should only return the path indices and the
correction token.

## Porting Plan

### LLaMA/Vicuna Path

Use this path for the shortest full-SpecExtend implementation because the
original SpecExtend repo already patches LLaMA-style modules.

1. Port the target verifier from
   `SpecExtend/specextend/shared/modeling_llama_kv_target.py`.
2. Port draft KV retrieval from
   `SpecExtend/specextend/application/model_classic.py`.
3. Port or adapt tree construction from
   `SpecExtend/specextend/shared/opt_tree.py`.
4. Implement `SpecExtendDraftBackend.build_draft_tree`.
5. Implement a cloud `verify_tree` backend and replace the `501` endpoint.

### Qwen Path

Use this path if the edge-cloud experiments must stay on Qwen models.

1. Start from `specedge/src/model/qwen3.py`, because it already exposes a graph
   model with explicit cache indices and tree-like execution.
2. Add target attention extraction equivalent to SpecExtend's last-layer
   retrieval signal.
3. Add draft full-cache and working-cache selection using
   `SpecExtendRetrievalState`.
4. Implement the same `SpecExtendDraftBackend` and cloud `verify_tree` contracts.

The Qwen path is a real model-port task. It is not a small wrapper around vLLM.

## Correctness Checks

Before performance experiments, validate in this order:

1. Protocol serialization round trip for `SpecExtendTreeRequest` and
   `SpecExtendTreeResponse`.
2. Retrieval chunk selection with fixed attention scores.
3. Single-process backend parity against original SpecExtend on one prompt.
4. Edge-cloud parity against the single-process backend for the same prompt and
   tree.
5. Full experiment comparison:
   - direct cloud generation
   - current vLLM tree async
   - SpecExtend without retrieval
   - SpecExtend with retrieval

Track acceptance length, generated tokens per round, edge draft time, cloud
verify time, network time, selected chunk IDs, and output text equality.

## Current Limitations

- `/specextend/verify` is intentionally not implemented for vLLM.
- `SpecExtendEdgeClient` requires a backend that implements
  `SpecExtendDraftBackend`.
- The retrieval module tracks selected token indices but does not own tensor KV
  memory. Actual KV gathering belongs in the model backend.
- Existing quick-test summaries do not yet include SpecExtend-specific metrics
  such as selected chunk IDs or retrieval update count.
