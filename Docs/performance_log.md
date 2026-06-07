# Performance Log

## Environment

- **Hardware**: 2× NVIDIA GeForce RTX 3090 (24 GB VRAM each)
- **Target**: Qwen3-14B-AWQ on GPU 0 (loaded via `device_map="auto"`, ~10 GB)
- **Draft**: Qwen3-1.7B on GPU 1 (FP16, ~14 GB)
- **Network**: Simulated "good" (RTT 15 ms, 240 Mbps down / 60 Mbps up)
- **Dataset**: PG-19
- **Settings**: max_tokens=256, nodes=32, max_depth=8, threshold=0.7,
  retrieval_chunk_size=32, retrieve_top_k=32, retrieve_every_n_steps=8

---

## 2026-06-07 — 4K Input Benchmark (pg19_4K, 4096 input tokens)

| Prompt | Method | Tokens | Total (ms) | Throughput (t/s) |
|--------|--------|--------|------------|------------------|
| 1 | direct | 256 | 18,515 | **13.8** |
| 1 | specextend_n32 | 256 | 101,804 | 2.5 |
| 2 | direct | 256 | 18,461 | **13.9** |
| 2 | specextend_n32 | 256 | 109,039 | 2.3 |
| 3 | direct | 256 | 18,606 | **13.8** |
| 3 | specextend_n32 | 256 | 107,232 | 2.4 |

**Direct average**: 13.8 t/s
**SpecExtend average**: 2.4 t/s

---

## 2026-06-07 — 8K Input Benchmark (pg19_8K, 8192 input tokens)

| Prompt | Method | Tokens | Total (ms) | Throughput (t/s) |
|--------|--------|--------|------------|------------------|
| 1 | direct | 256 | 18,671 | **13.7** |
| 1 | specextend_n32 | 256 | 106,373 | 2.4 |
| 2 | direct | 256 | 18,945 | **13.5** |
| 2 | specextend_n32 | 256 | 108,289 | 2.4 |
| 3 | direct | 256 | 18,690 | **13.7** |
| 3 | specextend_n32 | 257 | 105,301 | 2.4 |

**Direct average**: 13.6 t/s
**SpecExtend average**: 2.4 t/s

---

## Observations

- **Direct generation** is consistent across 4K and 8K input lengths (~13.7 t/s),
  indicating the 14B-AWQ model is compute-bound rather than memory-bound at
  these sequence lengths.
- **SpecExtend is ~5.7× slower** than direct in this configuration. The primary
  bottleneck is the number of rounds: with `max_depth=8`, only 8 draft tokens
  are proposed per round, requiring ~32 rounds to generate 256 tokens. Each
  round incurs:
  - Draft forward time on GPU 1 (~200–250 ms for 1024-token working cache)
  - HTTP round-trip to localhost (~1–2 ms, negligible here)
  - Verify forward time on GPU 0 (~350 ms for 14B AWQ)
- **Retrieval overhead**: `retrieve_every_n_steps=8` triggers attention-score
  extraction and chunk selection every 8 rounds, adding marginal cost.
- To close the gap, `max_depth` should be raised significantly (e.g., 32–64)
  so fewer rounds are needed, or batched draft generation should be used to
  amortize the working-cache attention cost across multiple draft positions.

---

## 2026-06-07 — 4K & 8K Quick Test (max_tokens=512, retrieve_every_n_steps=0)

| Config | Method | Tokens | Total (ms) | Throughput (t/s) | Acceptance |
|--------|--------|--------|------------|------------------|------------|
| pg19_4K | direct | 512 | 35,471 | **14.43** | — |
| pg19_4K | specextend_n32 | 512 | 125,316 | 4.09 | 0.77 |
| pg19_8K | direct | 512 | 35,482 | **14.43** | — |
| pg19_8K | specextend_n32 | 513 | 120,782 | 4.25 | 0.80 |

Settings: nodes=32, max_depth=8, threshold=0.7, retrieval_chunk_size=32, retrieve_top_k=32, retrieve_every_n_steps=0.
