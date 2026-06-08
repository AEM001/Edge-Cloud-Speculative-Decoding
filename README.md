# Edge-Cloud SpecExtend

Edge-cloud speculative decoding: a Qwen3 draft model on the edge GPU builds
draft trees; a Qwen3 target model on the cloud GPU verifies them. The draft
backend uses a SpecExtend-style full draft KV cache plus retrieval-selected
working KV cache so that only selected chunks are active during drafting.

## Hardware Defaults

```text
GPU 0  → cloud / verify server   models/Qwen3-14B-AWQ  (eager attention)
GPU 1  → edge  / draft backend   models/Qwen3-1.7B     (SDPA)
```

Override with `VERIFY_MODEL_PATH`, `VERIFY_DEVICE`, `DRAFT_MODEL_PATH`, `DRAFT_DEVICE`.

## Quick Start

```bash
# Terminal 1 — start verify server (stays running)
bash start_verify.sh

# Terminal 2 — run experiment
bash run_quick.sh
```

Or launch both together:

```bash
bash run_2x3090.sh
```

Results: `outputs/quick_test_results_<timestamp>.json`  (summary: `outputs/quick_test_results_<timestamp>_summary.txt`)

## Latest Results (2026-05-30, 2× RTX 3090)

PG-19, 2048 input tokens, 256 output tokens, nodes=32, depth=8.

| Method | Time | Tok/s | Acceptance length |
|--------|------|-------|-------------------|
| direct (14B-AWQ) | 18.6 s | 13.7 | — |
| specextend (1.7B draft) | **10.6 s** | **24.4** | 6.03 |

SpecExtend is **1.78× faster** than direct generation.

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VERIFY_MODEL_PATH` | `models/Qwen3-14B-AWQ` | Target model path |
| `VERIFY_GPU_ID` | `0` | Target GPU |
| `DRAFT_MODEL_PATH` | `models/Qwen3-1.7B` | Draft model path |
| `DRAFT_GPU_ID` | `1` | Draft GPU |
| `RETRIEVE_EVERY_N_STEPS` | `16` | Retrieval update frequency |
| `NODES` | `32` | Draft tree node budget |
| `MAX_TOKENS` | `256` | Max output tokens |

## Docs

- `Docs/performance_change_log.md` — change history and measurements
- `Docs/metrics_system.md` — output JSON schema
- `Docs/server_runbook.md` — setup and troubleshooting
