# Edge-Cloud Speculative Decoding

Research project studying speculative decoding across an edge-cloud network split.
A small draft model runs on the edge device; a large verify model runs on the cloud
server. Network latency and bandwidth are simulated to measure real-world impact.

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
├── server/
│   └── verify_server.py       FastAPI server exposing /verify and /generate
├── client/
│   ├── edge_client.py         Synchronous speculative decoding loop
│   ├── async_edge_client.py   Pipelined async variant (experimental)
│   └── http_cloud_client.py   HTTP transport to verify server
├── experiments/
│   ├── network_conditions.py  Network throttle simulation (good/medium/bursty)
│   ├── metrics_collector.py   Data model, export, summary tables
│   ├── run_network_experiment.py  Full sweep runner
│   └── outputs_network/
│       ├── results_latest.json
│       └── report.md          <- analysis report
├── draft_generator.py         vLLM-based draft token generator
├── model_manager.py           Model loader / GPU assignment
├── prompt_loader.py           Prompt dataset (simple + complex)
├── protocol.py                EdgeRequest / CloudResponse dataclasses
├── config.py                  Model paths, GPU mem fractions
└── quick_test.py              Fast sanity check: direct vs spec_k7
```

---

## Quick Start

### 1. Start verify server (GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 VERIFY_GPU_MEM=0.85 \
  .venv/bin/python -m server.verify_server --port 6006
```

### 2. Run quick sanity check (GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python quick_test.py
```

Runs direct vs spec_k7 over 4 prompts across good/medium/bursty conditions.
Prints a summary table with tok/s, speedup, and net useful tokens/round.

### 3. Run full network experiment

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m experiments.run_network_experiment \
    --prompts 5 --k-values 3 5 7
```

---

## Key Implementation Detail — verify() method

The verify server uses a **single prefill pass** per round:

```python
# Feed prefix + draft as one prompt, request logprobs only over draft window
SamplingParams(temperature=0.0, max_tokens=1,
               logprobs=1, prompt_logprobs=K)
llm.generate([TokensPrompt(prefix_ids + draft_ids)], ...)
```

vLLM runs one prefill (prefix KV cache hit — only K new positions computed)
then one decode for the correction token. Cost: **~30 ms/round** on 14B AWQ.

Previous broken approach: `generate(prefix, max_tokens=K+1)` did K+1 sequential
decode steps — **196 ms/round** (6.5x slower).

---

## Current Results (3B draft -> 14B verify, good network)

| Method | tok/s | vs direct | verify ms/round |
|--------|-------|-----------|-----------------|
| direct | 37 | 1.00x | — |
| spec_k7 (simple prompts) | 28–36 | 0.77–0.96x | ~30 ms |
| spec_k7 (complex prompts) | 12–18 | 0.32–0.49x | 200–400 ms |

Simple prompts nearly reach parity. Complex prompts suffer from O(n) attention
over long prefixes (500–900 tokens) inflating verify time beyond the 30 ms baseline.

See `experiments/outputs_network/report.md` for full analysis.

---

## Recommended Next Steps

| Priority | Change | Expected gain |
|----------|--------|---------------|
| **High** | Upgrade draft to `Qwen2.5-7B-Instruct-AWQ` | Acceptance ~57% -> ~72%, rounds ~27 -> ~20, spec wins by ~2x on simple prompts |
| Medium | Send token delta only (not full prefix) per round | Cuts uplink 20x, eliminates re-prefill for newly accepted tokens |
| Low | Shorter max_tokens (64 instead of 128) | Fewer rounds, less accumulated overhead |

See `experiments/README.md` for detailed break-even analysis and all options.
