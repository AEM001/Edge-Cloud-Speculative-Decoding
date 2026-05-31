# Edge-Cloud SpecExtend

Research system for real SpecExtend-style edge-cloud speculative decoding. A
custom Qwen3 draft backend runs on the edge GPU and a custom Qwen3 target
backend runs on the cloud GPU. The active path uses draft trees, target tree
verification, last-layer target attention scores, and retrieval-driven draft
cache bookkeeping.

## Current Status

- Runnable: direct cloud generation and real `specextend` tree verification on
  the `good` network profile.
- Backend: Hugging Face Qwen3 with eager attention, exposed through custom
  backend classes in `scripts/core/qwen_specextend_backend.py`.
- Protocol: `SpecExtendTreeRequest` and `SpecExtendTreeResponse` over HTTP.
- Scope: batch size 1, correctness-first implementation. CUDA graph and custom
  kernel optimization can be added after correctness/performance baselines are
  stable.

## Hardware Defaults

Single GPU setup running both draft and target models.

Override with `VERIFY_MODEL_PATH`, `VERIFY_DEVICE`, `DRAFT_MODEL_PATH`, and
`DRAFT_DEVICE`.

## Project Structure

```text
draft/
├── scripts/
│   ├── core/
│   │   ├── protocol.py
│   │   ├── qwen_specextend_backend.py
│   │   ├── specextend_backend.py
│   │   └── specextend_retrieval.py
│   ├── client/
│   │   ├── http_cloud_client.py
│   │   └── specextend_edge_client.py
│   ├── server/
│   │   └── verify_server.py
│   └── experiments/
│       ├── network_conditions.py
│       ├── prompt_loader.py
│       ├── quick_test.py
│       └── analyze_quick_run.py
├── Docs/
│   ├── specextend_integration.md
│   └── metrics_system.md
└── models/
```

## Quick Start

Start the target verify server:

```bash
bash start_verify.sh
```

Run the quick experiment:

```bash
./quick.sh
```

Defaults:

- prompt source: `prompts_2048`
- prompt count: `1`
- max generated tokens: `512`
- draft tree nodes: `32`
- max tree depth: `8`
- retrieval chunk size: `32`
- retrieval top-k chunks: `32`

Results are written to:

```text
scripts/experiments/outputs_quick/quick_test_results_*.json
```

## SpecExtend Runtime

The edge client:

- encodes the prompt
- builds a draft tree through `QwenSpecExtendDraftBackend`
- keeps full draft-cache token bookkeeping and a retrieval-selected working view
- sends `SpecExtendTreeRequest` to the cloud
- commits accepted tree-path tokens plus the target correction token
- updates retrieval state from target attention scores

The cloud server:

- loads the target Qwen3 model
- verifies each request through `/specextend/verify`
- returns accepted tree indices and a correction token
- returns last-layer target attention scores when requested
- keeps `/generate` for direct baseline generation

The compatibility `/verify` endpoint maps a linear draft request to a degenerate
SpecExtend tree and uses the same custom target backend.
