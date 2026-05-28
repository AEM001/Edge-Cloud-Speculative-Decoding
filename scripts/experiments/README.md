# Network-Constrained Speculative Decoding Experiments

Compares **direct generation**, **synchronous speculative decoding**, and the
tree async speculative baseline under simulated mobile network conditions.
The draft model runs on the edge device (GPU 1); the verify model runs on the
cloud server (GPU 0).

---

## Files

| File | Role |
|------|------|
| `network_conditions.py` | Network simulation — wraps any `cloud_client`, injects RTT, bandwidth throttle, and bursty spike state machine |
| `prompt_loader.py` | Prompt loading from SPEED-Bench dataset |
| `tree_async_client.py` | Tree async speculative client used by `quick_test.py` |
| `quick_test.py` | Fast sanity check with normalized direct/sync/tree metrics |
| `outputs_quick/` | Normalized results, raw round details, and grouped summaries |

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
  .venv/bin/python -m experiments.quick_test
```

Or for a quick sanity check:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m experiments.quick_test
```

---

## Quick Start

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m experiments.quick_test
```

Writes results to `experiments/outputs_quick/`.

---

## Configuration

`quick_test.py` currently uses module-level constants:

| Constant | Default | Description |
|----------|---------|-------------|
| `SERVER_URL` | `http://localhost:6006` | Verify server URL |
| `MAX_TOKENS` | `128` | Max new tokens per call |
| `K_VALUES` | `[7]` | Draft lengths K to test |
| `PROMPT_COUNT` | `2` | Prompts per type |

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

`quick_test.py` normalizes every method into the same schema:

| Section | Meaning |
|---------|---------|
| `output` | Generated tokens, total wall time, tokens/sec |
| `timing` | Local draft time, server model time, HTTP/RPC overhead, simulated UL/DL/network, RTT |
| `speculative` | Rounds, K, drafted tokens, accepted draft tokens, correction tokens, acceptance length |
| `async_detail` | Branch launch/readiness/reuse, prefetched tokens, exposed branch time |
| `verify_runtime` | Prefix/draft/input lengths and vLLM runtime settings |
| `raw` | Direct timing or per-round details |

---

## Output Files

```
outputs_quick/
├── quick_test_results.json      # normalized rows
├── quick_test_rounds.jsonl      # raw direct/round/slot details
└── quick_test_summary.json      # grouped averages and speedups
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
Feed `prefix_ids + draft_ids` as a single prompt with `prompt_logprobs=1,
max_tokens=1`. vLLM runs **one prefill** over all tokens (prefix KV cache hit)
then **one decode** for the correction token. The verifier is greedy
(`temperature=0`): it only needs the target top-1 token at each draft position.
Stochastic speculative sampling would require target probabilities and
rejection resampling logic.

The `/verify` wire protocol omits draft logprobs and does not return accepted
token IDs. The edge reconstructs accepted tokens from `draft_ids[:accepted_len]`.

---
