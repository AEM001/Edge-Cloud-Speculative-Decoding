# Edge-Cloud Speculative Decoding

Research system measuring the effect of network latency and bandwidth on
speculative decoding when the draft and verify models are physically separated.

A small draft model runs on the **edge device** (GPU 1); a large verify model
runs on the **cloud server** (GPU 0). Network conditions are simulated with
a calibrated throttle wrapper so every method sees identical link parameters.

---

## Hardware

```
GPU 0 (RTX 4090)  →  cloud / verify server   Qwen2.5-14B-Instruct-AWQ
GPU 1 (RTX 4090)  →  edge  / draft model     Qwen2.5-3B-Instruct-AWQ
```

---

## Project Structure

```
draft/
├── scripts/
│   ├── core/                      # Core infrastructure (shared across codebase)
│   │   ├── protocol.py            # EdgeRequest / CloudResponse dataclasses + wire format
│   │   ├── model_manager.py       # Model loader / GPU assignment
│   │   └── draft_generator.py     # vLLM draft token generator (TokensPrompt, prefix caching)
│   ├── client/                    # Edge clients
│   │   ├── edge_client.py         # Synchronous speculative decoding loop
│   │   ├── async_edge_client.py   # Pipelined async client (PicoSpec-style, lookahead=1)
│   │   └── http_cloud_client.py   # HTTP/keep-alive transport to verify server
│   ├── server/                    # Cloud verification server
│   │   └── verify_server.py       # FastAPI server: /verify (speculative) + /generate (direct)
│   └── experiments/               # Experiment-specific utilities
│       ├── network_conditions.py  # Throttle simulation: good / medium / bursty profiles
│       ├── metrics_collector.py   # Data model, JSON export, summary tables
│       ├── prompt_loader.py       # Prompt dataset (simple + complex categories)
│       ├── run_network_experiment.py  # Full K-sweep runner
│       ├── quick_test.py          # Fast sanity check: direct vs sync_k7 vs cqt_k7
│       └── outputs_network/
│           ├── results_latest.json
│           └── report.md
├── config.py                      # Model paths, GPU memory fractions
└── models/                        # Model download scripts and config
```

---

## Quick Start

### 1. Start the verify server (GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 VERIFY_GPU_MEM=0.85 \
  .venv/bin/python -m server.verify_server --port 6006
```

### 2. Run the quick sanity check (GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m experiments.quick_test
```

Runs **direct** vs **sync_k7** vs **cqt_k7** (async with CQT prefix selection)
over 2 simple prompts × 3 network conditions. Prints tok/s, speedup, net useful
tokens/round, rollbacks, and bubble time per row.

### 3. Run the full network sweep

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m experiments.run_network_experiment \
    --prompts 5 --k-values 3 5 7
```

---

## Verify Method — Single Prefill Pass

The verifier runs **one prefill + one decode** per round, not K+1 decode steps:

```python
SamplingParams(temperature=0.0, max_tokens=1,
               logprobs=1, prompt_logprobs=K)
llm.generate([TokensPrompt(prefix_ids + draft_ids)], ...)
# prompt_logprobs[prefix_len + j] = verify distribution at draft position j
```

vLLM prefix caching means the `prefix_ids` KV entries are reused from the
previous round — only the K new draft positions are computed from scratch.

- **Current cost:** ~30 ms/round on 14B AWQ (K=7, simple prompts)
- **Previous broken approach:** `generate(prefix, max_tokens=K+1)` — 196 ms/round (6.5×)

---

## Async Pipeline — PicoSpec-style (lookahead = 1)

```
Sync:   [draft 0] → [verify 0] → [draft 1] → [verify 1] → ...
        Latency per round = T_draft + T_verify

Async:  [draft 0] ──────────────────────────────────────► ...
                       [verify 0] → result ──────────────► ...
        Latency per round ≈ max(T_draft, T_verify)  (full-hit case)
```

A background thread handles all verify calls; the drafter optimistically
advances the speculative prefix and generates the next batch immediately.

On **full hit**: committed prefix advances, pipeline stays warm.
On **rejection**: speculative prefix rolls back, all in-flight slots are flushed.

### CQT Prefix Selection (active in `quick_test.py`)

Naïve async always assumes the *entire* K draft was accepted before drafting
the next batch. This over-predicts the prefix on high-RTT links, causing
expensive rollbacks.

**Conservative Quantile Tracking (CQT)** maintains a rolling window of the
last 50 accepted lengths and sets the next draft's starting prefix to the
**10th percentile** of that history:

```python
prefix_len = percentile(history_accepted_lens[-50:], p=0.10)
# draft from: committed_prefix + first min(spec_ahead, prefix_len) tokens
```

- Guarantees ≥90% of async drafts start from a prefix the verifier will confirm
- Automatically adapts as acceptance rate changes across network conditions
- Zero GPU overhead — O(1) sort of a 50-element list per round
- Falls back to full optimistic advance for the first round (empty history)

The `CQTAsyncEdgeClient` subclass in `quick_test.py` overrides `_pipeline_loop`
only; the base `AsyncEdgeClient` is unchanged.

---

## Transport Overhead — Current Architecture and Options

The current stack is: **edge → HTTP/JSON → FastAPI/Pydantic → vLLM subprocess → ZMQ**.

Each verify round pays:
- TCP round-trip (keep-alive session reused, no per-call handshake)
- JSON serialisation of prefix_ids + draft_ids (grows with prefix length)
- Pydantic validation on the server
- ZMQ IPC between FastAPI worker and vLLM engine (~15–20 ms)

This overhead is **intentional** when the research goal is measuring realistic
cloud API latency. The five main architectural options, in order of impact:

| Option | Description | Tradeoff |
|--------|-------------|----------|
| **vLLM OpenAI-compatible API** | Use `--served-model-name` + persistent HTTP connection; GPU batching is automatic; edge client becomes a standard `openai` call | Loses custom `prefix_ids` / `prompt_logprobs` control; requires re-engineering the verify logic |
| **Batch multiple rounds per HTTP call** | Send 3–5 draft batches in one request; server verifies all and returns all results; amortises HTTP + IPC overhead across rounds | Increases latency per individual result; complicates the async pipeline |
| **Raw socket / gRPC** | Replace FastAPI with a binary framing protocol; send token-ID arrays as `int32` arrays; eliminates Pydantic + JSON serialisation | High implementation cost; loses HTTP ecosystem (retries, health checks, load balancers) |
| **Run vLLM in-process** | Embed vLLM directly in the verify server process instead of subprocess IPC; eliminates ZMQ (~15–20 ms/round) | Single-process; harder to isolate GPU memory between draft and verify |
| **Accept HTTP as realistic cost** | If the research goal is measuring *network impact* on speculative decoding, HTTP overhead is part of realistic cloud-server latency — measure it, don't remove it | Means reported numbers include transport cost, not just model cost |

---

## Current Results (3B draft → 14B verify, good network, K=7)

| Method | tok/s | vs direct | verify ms/round | notes |
|--------|-------|-----------|-----------------|-------|
| direct | ~37 | 1.00× | — | single /generate call |
| sync_k7 | 28–36 | 0.77–0.96× | ~30 ms | simple prompts |
| sync_k7 | 12–18 | 0.32–0.49× | 200–400 ms | complex prompts (long prefix) |
| cqt_k7 | TBD | TBD | ~30 ms | async + CQT p10, lookahead=1 |

Simple prompts nearly reach parity with direct. Complex prompts suffer from
O(n) attention over long prefixes (500–900 tokens) inflating verify time past
the 30 ms baseline — prefix caching helps but does not fully eliminate this.

---

## Known Bottlenecks and Next Steps

| Priority | Change | Expected gain |
|----------|--------|---------------|
| **High** | Upgrade draft to `Qwen2.5-7B-Instruct-AWQ` | Acceptance rate ~57% → ~72%; fewer rounds; spec wins ~2× on simple prompts |
| **High** | Send token delta only (not full prefix_ids) per round | Cuts uplink payload ~20×; eliminates re-prefill for accepted tokens |
| Medium | Evaluate CQT p10 vs p20 vs fixed-margin (avg × 0.7) on bursty network | Quantify rollback reduction vs bubble-time increase |
| Medium | Run vLLM in-process to eliminate ZMQ IPC | Saves ~15–20 ms/round; simplifies deployment |
| Low | Shorter max_tokens (64 instead of 128) | Fewer rounds, less accumulated prefix overhead |
| Low | Batch 3–5 verify rounds per HTTP call | Amortises transport cost; useful on very high-latency links |

See `experiments/outputs_network/report.md` for full per-condition analysis.
