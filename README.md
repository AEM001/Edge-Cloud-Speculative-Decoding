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
│   │   ├── http_cloud_client.py   # HTTP/keep-alive transport to verify server
│   ├── server/                    # Cloud verification server
│   │   └── verify_server.py       # FastAPI server: /verify (speculative) + /generate (direct)
│   └── experiments/               # Experiment-specific utilities
│       ├── network_conditions.py  # Throttle simulation: good / medium / bursty profiles
│       ├── prompt_loader.py       # Prompt dataset (simple + complex categories)
│       ├── tree_async_client.py   # Tree-based async speculative baseline
│       ├── quick_test.py          # Fast sanity check: direct vs sync_k7 vs tree_k7_b3
│       └── outputs_quick/
│           ├── quick_test_results.json
│           ├── quick_test_rounds.jsonl
│           └── quick_test_summary.json
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

Runs **direct** vs **sync_k7** vs **tree_k7_b3** (tree-based with branch prefetch)
over the configured prompt/network subset. It writes normalized result rows,
raw per-round details, and grouped summaries to `scripts/experiments/outputs_quick/`.

## Verify Method — Greedy Single Prefill Pass

The current verifier implements **greedy** speculative decoding (`temperature=0`).
It checks whether each draft token equals the target model's top-1 token. On the
first mismatch, it returns the target top-1 correction token. This is not the
stochastic speculative sampling algorithm; stochastic rejection requires target
probabilities and resampling from the corrected distribution.

The verifier runs **one prefill + one decode** per round, not K+1 decode steps:

```python
SamplingParams(temperature=0.0, max_tokens=1,
               logprobs=1, prompt_logprobs=1)
llm.generate([TokensPrompt(prefix_ids + draft_ids)], ...)
# prompt_logprobs[prefix_len + j] contains the top-1 target token at draft position j
```

vLLM prefix caching means the `prefix_ids` KV entries are reused from the
previous round — only the K new draft positions are computed from scratch.

The `/verify` protocol is intentionally slim:

- Request sends `request_id`, `round_id`, `prefix_ids`, `draft_ids`, timing, and policy metadata.
- Draft logprobs are not sent; greedy verification does not use them.
- Response sends `accepted_len` and `correction_token_id`.
- Accepted token IDs are not returned because the edge already has `draft_ids[:accepted_len]`.

- **Current cost:** ~30 ms/round on 14B AWQ (K=7, simple prompts)
- **Previous broken approach:** `generate(prefix, max_tokens=K+1)` — 196 ms/round (6.5×)

---

## Tree Async Baseline

```
Round n:
  draft K base tokens
  send base draft to verifier in a background thread
  start tree branch drafting in another background thread
  when verify returns:
      commit accepted base tokens + correction
      use a branch only if it is already ready and matches the verified path
```

Branch offsets are centered around `round(0.60 * K)`. A branch is reusable only
when it reconnects exactly to the target-verified path:

```text
branch.offset <= accepted_len
branch.draft_ids starts with base_draft[branch.offset:accepted_len] + correction_token
```

If a branch is still running when the verifier result arrives, the critical path
does not wait for it. The benchmark records whether branch drafting was actually
hidden with `async_detail.exposed_branch_ms`.

---

## Metrics

`quick_test.py` normalizes direct, sync speculative, and tree async rows into:

- `output`: generated tokens, total wall time, tok/s
- `timing`: local draft time, server model time, HTTP/RPC overhead, simulated UL/DL/network, RTT
- `speculative`: rounds, K, drafted tokens, accepted draft tokens, correction tokens, acceptance
- `async_detail`: launched/ready/reused branches, prefetched tokens, exposed branch time
- `verify_runtime`: prefix/draft/input lengths and vLLM runtime settings
- `raw`: method-specific direct timing or per-round details

Output files:

```text
quick_test_results.json   # normalized per-run rows
quick_test_rounds.jsonl   # raw direct/round/slot details
quick_test_summary.json   # grouped averages and speedups
```

---

## Transport Overhead — Current Architecture and Options

The current stack is: **edge → HTTP/JSON → FastAPI/Pydantic → vLLM subprocess → ZMQ**.

Each verify round pays:
- TCP round-trip (keep-alive session reused, no per-call handshake)
- JSON serialisation of prefix_ids + draft_ids (still grows with prefix length)
- Pydantic validation on the server
- ZMQ IPC between FastAPI worker and vLLM engine (~15–20 ms)
- vLLM request scheduling, `TokensPrompt` construction, prefix-cache lookup, and output-object construction

Already minimized in the current protocol:

- `prompt_logprobs=1`, not K, because greedy verification only needs the target top-1 token.
- `draft_logprobs` are not serialized to `/verify`.
- `accepted_token_ids` are not returned; the edge reconstructs them from the draft and `accepted_len`.
- Direct `/generate` uses a persistent `requests.Session`.

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

## Results

For recent diagnostic findings on tree-based speculative decoding, see:
`experiments/outputs_quick/DIAGNOSTIC_REPORT_2025-05-05.md`

---

## Known Bottlenecks and Next Steps

| Priority | Change | Expected gain |
|----------|--------|---------------|
| **High** | Improve tree offset prediction accuracy | Tree currently underperforms due to poor rejection point prediction; better prediction could enable true prefetch benefit |
| **High** | Upgrade draft to `Qwen2.5-7B-Instruct-AWQ` | Higher acceptance rate reduces rounds; speculative methods more competitive |
| **High** | Send token delta only (not full prefix_ids) per round | Cuts uplink payload sharply; requires server-side session state and recovery logic |
| Medium | Run vLLM in-process to eliminate ZMQ IPC | Saves ~15–20 ms/round; simplifies deployment |
| Medium | Binary token transport for `/verify` | Avoids JSON/Pydantic list overhead for token IDs |
| Low | Shorter max_tokens (64 instead of 128) | Fewer rounds, less accumulated prefix overhead |
| Low | Batch 3–5 verify rounds per HTTP call | Amortises transport cost; useful on very high-latency links |

### Tree-Based Decoding

The tree-based client (`tree_async_client.py`) attempts to hide branch drafting behind verify RTT by predicting the rejection point and prefetching continuation tokens. Current limitations:

- **Requires exact offset prediction**: Branch prefetch only helps when `branch.offset == actual accepted_len` and `branch.draft_ids[0] == correction_token`
- **Earlier offsets are allowed but must bridge**: If `branch.offset < accepted_len`, the branch must first reproduce `base_draft[offset:accepted_len] + correction_token`.
- **Prediction quality is the bottleneck**: Branch work only helps when it is ready before verify returns and reconnects to the target path.
- **Alternative strategies**: Consider hybrid approaches (sync → tree when confidence high) or improved prediction using smaller history windows

See `experiments/outputs_quick/DIAGNOSTIC_REPORT_2025-05-05.md` for detailed analysis.
