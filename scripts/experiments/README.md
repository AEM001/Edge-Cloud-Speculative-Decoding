# Network-Constrained Speculative Decoding Experiments

Compares **direct generation** and **synchronous speculative decoding** (K = 3, 5, 7)
under simulated mobile network conditions. Designed for the edge-cloud split
research scenario: the draft model runs on the edge device (GPU 1), the verify
model runs on the cloud server (GPU 0).

---

## Files

| File | Role |
|------|------|
| `network_conditions.py` | Network simulation — wraps any `cloud_client`, injects RTT, bandwidth throttle, and bursty spike state machine |
| `metrics_collector.py` | Unified data model (`ExperimentResult`), converters, aggregation, JSON/CSV export, and table printing |
| `prompt_loader.py` | Prompt loading from SPEED-Bench dataset |
| `quick_test.py` | Fast sanity check: direct vs sync_k7 vs tree_async (for quick validation) |
| `outputs_network/report.md` | Full analysis report with tables and root-cause findings |

**Dependencies:**
- `core.protocol` - Data structures (EdgeRequest, CloudResponse, DraftRequest, DraftResponse)
- `core.model_manager` - Model loading and GPU management
- `core.draft_generator` - Draft token generation
- `client.edge_client` - Synchronous edge client
- `client.http_cloud_client` - HTTP transport to verify server

---

## Hardware Setup (2× RTX 4090)

```
GPU 0  →  verify server  (Qwen2.5-14B-Instruct-AWQ, 0.85 mem util)
GPU 1  →  draft model    (Qwen2.5-3B-Instruct-AWQ,  0.60 mem util)
```

Start order:

```bash
# Terminal 1 — verify server on GPU 0
CUDA_VISIBLE_DEVICES=0 VERIFY_GPU_MEM=0.85 \
  .venv/bin/python -m server.verify_server --port 6006

# Terminal 2 — experiment on GPU 1
CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python -m experiments.run_network_experiment --prompts 5
```

Or for a quick sanity check:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python quick_test.py
```

---

## Quick Start

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m experiments.run_network_experiment \
    --prompts 5 \
    --max-tokens 128 \
    --k-values 3 5 7
```

Writes results to `experiments/outputs_network/`.

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `http://localhost:6006` | Verify server URL |
| `--max-tokens` | `128` | Max new tokens per call |
| `--prompts` | `5` | Prompts per type (simple + complex) |
| `--k-values` | `3 5 7` | Draft lengths K to sweep |
| `--profiles` | all | `good`, `medium`, `bursty` |
| `--no-direct` | — | Skip direct generation baseline |

---

## Network Profiles

| Profile | RTT | Download | Upload | Notes |
|---------|-----|----------|--------|-------|
| `good` | 25 ms | 120 Mbps | 30 Mbps | Strong 5G / WiFi |
| `medium` | 55 ms | 35 Mbps | 20 Mbps | Typical 4G |
| `bursty` | 40 ms normal / 250 ms spike | 30/5 Mbps | 15/2 Mbps | 4G with 2 s congestion bursts every 8 s |

Bursty cycle: 8 s normal → 2 s spike → repeat.

### Adding a custom profile

```python
from experiments.network_conditions import NetworkCondition

satellite = NetworkCondition(
    name="satellite",
    rtt_ms=600.0, download_mbps=10.0, upload_mbps=3.0,
)
```

---

## Metrics Tracked

### Core (all methods)

| Metric | Field |
|--------|-------|
| Tokens/second | `tokens_per_second` |
| Total latency | `total_latency_ms` |
| Tokens generated | `tokens_generated` |

### Speculative-only

| Metric | Field | Why it matters |
|--------|-------|----------------|
| Acceptance ratio | `acceptance_ratio` | Quality of draft model |
| **Net useful tokens/round** | derived: `accepted / rounds` | Actual amortisation per RTT |
| Total rounds | `total_rounds` | Determines total network overhead |
| Draft time | `total_draft_time_ms` | Edge compute cost |
| Verify time | `total_verify_time_ms` | Should be ~30 ms/round after fix |
| Average RTT | `average_rtt_ms` | verify + network |
| Simulated overhead | `simulated_overhead_ms` | Pure injected network delay |

---

## Output Files

```
outputs_network/
├── results_YYYYMMDD_HHMMSS.json
├── results_YYYYMMDD_HHMMSS.csv
├── summary_YYYYMMDD_HHMMSS.json
├── results_latest.json          ← symlink
├── results_latest.csv           ← symlink
└── report.md                    ← full analysis
```

---

## Architecture

```
EdgeClient (GPU 1 — edge)
  │  draft_generator (3B)
  │  cloud_client = ThrottledCloudClient
  │      └─► HTTP /verify ──► verify server (GPU 0 — cloud)
  │                               └─► CloudVerifier.verify()
  │                                     one prefill(prefix+draft) + 1 decode
  │                                     ≈ 30 ms/round  (vs 196 ms before fix)
  └─► metrics
```

**Key implementation detail — `verify()` method:**
Feed `prefix_ids + draft_ids` as a single prompt with `prompt_logprobs=K,
max_tokens=1`. vLLM runs **one prefill** over all tokens (prefix KV cache hit)
then **one decode** for the correction token. Total GPU work per round ≈ 30 ms,
vs the broken approach of `generate(prefix, max_tokens=K+1)` which did K+1
sequential decode steps ≈ 196 ms/round.

---

## Performance Results (2× 4090, 3B draft → 14B verify)

### After verify fix (prompt_logprobs single-prefill approach)

| Network | direct tok/s | spec_k7 tok/s | speedup | verify ms/round |
|---------|-------------|---------------|---------|-----------------|
| good    | 37          | 24            | 0.65×   | ~30 ms          |
| medium  | 37          | 21            | 0.57×   | ~30 ms          |
| bursty  | 37          | 22            | 0.60×   | ~30 ms          |

Simple prompts reach **0.77–0.96×** of direct. Complex prompts (500+ token prefix)
remain slower due to O(n) attention over the growing KV cache.

### Break-even analysis

For spec_k7 to reach parity with direct (37 tok/s):

```
net_useful_toks/round ≥ verify_rtt_ms × direct_tps / 1000
≥ 30 ms × 37 tok/s / 1000 = 1.11 tok/round   ← already satisfied (we get ~4)

But also: total_spec_time = rounds × (draft + verify + net)
         total_direct_time = tokens / direct_tps

For 128 tokens, ~25 rounds:
  spec:   25 × (28ms draft + 30ms verify + 25ms net) = 25 × 83ms = 2075ms
  direct: 128 / 37 × 1000 = 3459ms
```

Wait — the math says spec should win. Why doesn't it in practice?  
The **complex prompt RTT is 250–400 ms**, not 30 ms. The long prefix attention
(500–900 tokens) inflates verify time from 30 ms (short prefix) to 200+ ms.

---

## What to Change to Make Speculative Decoding Win

### Option 1 — Better draft model (highest impact, easy)

The 3B draft achieves ~55–64% acceptance. A stronger draft model in the same
memory budget would increase `net_useful_toks/round` and reduce rounds:

| Draft model | VRAM | Expected acceptance | Rounds (128 tok) |
|-------------|------|---------------------|-----------------|
| Qwen2.5-1.5B-AWQ | ~2 GB | ~45% | ~35 |
| **Qwen2.5-3B-AWQ** (current) | ~4 GB | ~57% | ~27 |
| Qwen2.5-7B-AWQ | ~8 GB | ~70–75% | ~20 |
| Qwen2.5-14B-AWQ (same as verify) | ~12 GB | ~90%+ | ~15 |

**Recommendation:** try `Qwen2.5-7B-Instruct-AWQ` as draft on GPU 1.
At 70–75% acceptance and ~20 rounds, total spec time ≈ 20 × 83ms = 1660ms
vs direct 3460ms → **2.1× speedup**.

### Option 2 — Sliding window / chunked attention for long prefixes

The O(n) attention on 500+ token prefixes dominates verify time for complex prompts.
Setting `--max-tokens 64` (shorter responses) reduces rounds proportionally.

### Option 3 — Larger verify model (research value, not perf)

If the research goal is to study the network overhead under realistic cloud
conditions, use a 72B verify model (needs tensor parallelism or quantisation).
Per-token latency rises to ~150 ms → spec wins more easily, but requires
multi-GPU cloud setup.

### Option 4 — Reduce prefix payload size (network optimisation)

Currently `prefix_ids` (the full growing token list) is sent every round. For
a 500-token prompt at round 25, that's ~2 KB uplink per round. Sending only a
delta (the newly accepted tokens) would cut uplink by ~20×. This requires a
stateful server that maintains per-request KV state, but would also eliminate
the re-prefill cost entirely.

### TL;DR recommendation for current 2× 4090 setup

> **Swap the draft model to `Qwen2.5-7B-Instruct-AWQ`.**  
> It fits on GPU 1 (~8 GB), should raise acceptance to ~70–75%, reduce rounds
> by ~25%, and push spec_k7 to ~1.5–2× faster than direct on simple prompts.
