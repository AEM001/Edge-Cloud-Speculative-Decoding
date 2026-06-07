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

## 2026-06-08 — Tree Cache/Verify Fix Retest (pg19_2K, 256 input tokens)

Environment: 2x RTX 3090, Qwen3-14B-AWQ target on GPU 0, Qwen3-1.7B draft on
GPU 1, PG-19 prompt 1, 256 tokenizer input tokens, 256 output tokens, simulated
"good" network. The verify server normalizes the local custom Qwen backend from
requested `flash_attention_2` to effective `sdpa` because this repo does not
implement a FlashAttention2 Qwen3 layer.

| Method | Tokens | Total (ms) | Throughput (t/s) | Rounds | Acceptance |
|--------|--------|------------|------------------|--------|------------|
| direct | 256 | 17,878 | **14.32** | — | — |
| specextend branching, broken baseline | 256 | 126,107 | 2.03 | 86 | 1.98 |
| specextend linear, broken baseline | 256 | 93,751 | 2.73 | 198 | 0.29 |
| specextend branching, fixed, nodes=32 depth=4 | 258 | 28,908 | 8.92 | 77 | 2.35 |

Fixed SpecExtend timing breakdown:

- `local_draft_ms`: 15,261 ms
- `server_model_ms`: 12,146 ms
- `kv_load_ms`: 16.7 ms
- retrieval selected chunks reached `[0, 1, 2, 3, 4, 5, 6, 7]`

Result: the minutes-scale regression is fixed (126.1s -> 28.9s), but direct
generation is still faster on this short 256-token input. Remaining cost is
mostly the number of verify rounds and Qwen3-1.7B draft tree construction.

The fastest short-run sweep for 64 output tokens was `nodes=32, depth=4`:

| Tree config | Tokens | Total (ms) | Throughput (t/s) | Rounds | Acceptance |
|-------------|--------|------------|------------------|--------|------------|
| nodes=16 depth=4 | 66 | 8,769 | 7.53 | 23 | 1.87 |
| nodes=24 depth=4 | 64 | 8,364 | 7.65 | 23 | 1.78 |
| nodes=32 depth=4 | 64 | 7,161 | **8.94** | 19 | 2.37 |
| nodes=32 depth=5 | 64 | 10,501 | 6.09 | 22 | 1.91 |
| nodes=32 depth=6 | 66 | 9,055 | 7.29 | 21 | 2.14 |

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
