# KV Tier Validation Report - 2026-06-08

Historical note: this report records an older KV-tier smoke run. It is not the
current observability setup. The current fixed observability setup is documented
in `Docs/observability.md` and uses `draft_tree_max_depth=6`, `retrieve_top_k=16`,
and `retrieve_every_n_steps=16`.

## Goal

This experiment validates whether the current sparse block selection setting
supports the two premises needed for the next research step:

1. During sparse selection / update, selected KV blocks can reside outside GPU
   memory, specifically in CPU memory or SSD.
2. Loading KV from CPU memory or SSD has a measurable impact on end-to-end
   performance.

If both premises hold, then continuing the research on sparse selected KV
placement and load cost is meaningful.

## Setup

- Hardware: 2 x RTX 3090.
- Target verifier: Qwen3-14B-AWQ on GPU 0.
- Draft model: Qwen3-1.7B on GPU 1.
- Dataset: PG-19, `pg19_2K`.
- Prompt input limit: 256 tokens.
- Output limit: 256 tokens.
- Draft mode: branching, 32 tree nodes, max depth 4, draft length 12.
- Retrieval config: chunk size 64, top-k 16, retrieve every 16 steps.
- Result file: `scripts/experiments/outputs_quick/quick_test_results_20260608_115624.json`.
- Logs: `verify_server_2x3090.log`, `bench_4k_final.log`.

Note: this was a fast smoke run, not the final 4K multi-prompt benchmark.

## Results

| Method | Throughput | Total Time | KV Load Time | CPU KV Updates | SSD KV Updates | Selected CPU Chunks | Selected SSD Chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 16.85 tok/s | 15.20 s | 0.00 ms | 0 | 0 | 0 | 0 |
| specextend_gpu | 9.14 tok/s | 28.34 s | 17.28 ms | 0 | 0 | 0 | 0 |
| specextend_kvload_cpu | 8.34 tok/s | 31.07 s | 145.29 ms | 5 | 0 | 38 | 0 |
| specextend_kvload_ssd | 7.60 tok/s | 34.06 s | 336.97 ms | 0 | 5 | 0 | 38 |

The SpecExtend rows all reported `specextend_tree_verify: true`, confirming
that the branching/tree verification path was enabled for this run.

## Validation of the Two Premises

### Premise 1: selected blocks can be in CPU memory or SSD

Validated.

In the CPU-tier profile, the run recorded:

- `updates_with_cpu_kv = 5`
- `selected_cpu_chunks = 38`
- `cpu_load_ms = 145.294`

In the SSD-tier profile, the run recorded:

- `updates_with_ssd_kv = 5`
- `selected_ssd_chunks = 38`
- `ssd_load_ms = 336.967`

This shows that, under the sparse selected KV setup, selected blocks can indeed
come from CPU memory or SSD, and the code path accounts for loading those KV
blocks.

### Premise 2: loading KV from CPU/SSD affects overall performance

Validated.

Compared with the all-GPU SpecExtend profile:

- all-GPU SpecExtend: 28.34 s, 9.14 tok/s
- CPU KV-load profile: 31.07 s, 8.34 tok/s
- SSD KV-load profile: 34.06 s, 7.60 tok/s

The CPU profile added 145.29 ms of measured KV load time and reduced throughput
from 9.14 tok/s to 8.34 tok/s. The SSD profile added 336.97 ms of measured KV
load time and reduced throughput further to 7.60 tok/s.

The end-to-end slowdown is larger than the isolated KV-load timer alone, so the
effect likely includes secondary runtime changes such as draft/cache rebuild
cost variation and extra synchronization. Still, the direction is clear: CPU
and SSD KV loading are visible in whole-run performance.

## Conclusion

Both research premises are supported by this validation run:

1. Sparse selected KV blocks can be placed in CPU memory or SSD and later loaded.
2. Loading those KV blocks has a measurable impact on end-to-end performance,
   with SSD producing a larger slowdown than CPU in this run.

Therefore, it is reasonable to continue the research direction. The next step
should be a larger benchmark, preferably with 4K input and multiple prompts, to
measure whether the same CPU/SSD KV-load effect remains stable at the target
experimental scale.
