# Edge-Cloud Speculative Decoding

Research system for edge-cloud speculative decoding when the draft and verify
models are physically separated.

A small draft model runs on the **edge device** (GPU 1); a large verify model
runs on the **cloud server** (GPU 0). The current quick experiment is fixed to
the `good` network profile and compares direct generation against the tree async
method.

---

## Hardware

```
GPU 0 (RTX 3090)  →  cloud / verify server   Qwen2.5-14B-Instruct-AWQ
GPU 1 (RTX 3090)  →  edge  / draft model     Qwen2.5-1.5B-Instruct-AWQ
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
│   │   ├── __init__.py
│   │   ├── edge_client.py         # Synchronous speculative decoding loop
│   │   └── http_cloud_client.py   # HTTP/keep-alive transport to verify server
│   ├── server/                    # Cloud verification server
│   │   ├── __init__.py
│   │   └── verify_server.py       # FastAPI server: /verify (speculative) + /generate (direct)
│   └── experiments/               # Experiment-specific utilities
│       ├── network_conditions.py  # Throttle simulation profiles; quick test uses good
│       ├── prompt_loader.py       # GSM8K, HumanEval, LongWriter prompt loader
│       ├── tree_async_client.py   # Tree-based async speculative client
│       ├── quick_test.py          # Direct vs tree async on good network
│       ├── analyze_quick_run.py   # Builds summaries / round details from quick results
│       └── outputs_quick/
│           ├── quick_test_results.json
│           ├── quick_test_rounds.jsonl
│           └── quick_test_summary.json
├── config.py                      # Model paths, GPU memory fractions
└── models/                        # Downloaded model weight directories
    ├── Qwen2.5-1.5B-Instruct-AWQ
    ├── Qwen2.5-3B-Instruct-AWQ
    ├── Qwen2.5-7B-Instruct-AWQ
    ├── Qwen2.5-14B-Instruct-AWQ
    ├── Qwen3-8B
    ├── qwen3_8b_eagle3
    ├── download.py                # Model download script
    └── hf.txt                     # Hugging Face token reference
```

---

## Quick Start

### 1. Start the verify server on GPU 0

```bash
bash start_verify.sh
```

`start_verify.sh` defaults to:

- verify model: `models/Qwen2.5-14B-Instruct-AWQ`
- GPU: `0`
- max model length: `32768`
- prefix caching: enabled
- eager mode: enabled by default via `VERIFY_ENFORCE_EAGER=1`

### 2. Run the quick test on GPU 1

```bash
./quick.sh
```

The wrapper runs **direct** vs **tree async** on the `good` network profile.
Current wrapper defaults are:

- prompt source: `longwriter_single_turn:input_10k`
- prompt count: `10`
- max generated tokens: `4096`
- tree base draft length: `K=12`
- tree branch width: `3`
- tree branch pre-draft length: `10`
- draft max model length: `32768`

`quick_test.py` writes normalized result rows to:

```text
scripts/experiments/outputs_quick/quick_test_results.json
```

### 3. Analyze the quick-test output

```bash
python3 scripts/experiments/analyze_quick_run.py
```

The analyzer reads `quick_test_results.json` and writes:

```text
quick_test_summary.json   # direct vs tree averages and speedup
quick_test_rounds.jsonl   # flattened direct / per-method detail rows
```

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
previous round, so only new draft positions are computed from scratch.

The `/verify` protocol is intentionally slim:

- Request sends only `request_id`, `prefix_ids`, and `draft_ids`.
- Draft logprobs are not sent; greedy verification does not use them.
- Response sends `request_id`, `accepted_len`, `correction_token_id`, and `server_verify_time_ms`.
- Accepted token IDs are not returned because the edge already has `draft_ids[:accepted_len]`.

- Default verify context length is `32768` for long-input LongWriter runs.

---

## Tree Async Method

```
Round n:
  if reusable prefetched tokens exist:
      send those tokens immediately as the verify draft
  else:
      draft K base tokens and send them as the verify draft

  while verification runs:
      pre-draft local branches from simple offsets

  when verify returns:
      commit accepted base tokens + correction
      reuse branch suffix tokens only if a branch reconnects to the verified path
```

The tree client is latency-oriented: if any reusable prefetched tokens are
available from the previous round, it sends them immediately. It does **not**
wait to top them up to the full base draft length `K`.

If no reusable tokens exist, the edge model drafts a fresh base draft of length
`K`, capped by the remaining generation budget.

The current pre-draft offset policy is deliberately simple. For a verify draft
of length `sent_draft_len`, branches are started at:

```text
1
sent_draft_len
round(0.6 * tree_branch_draft_length)
```

Offsets are clipped into `[1, sent_draft_len]`, duplicates are removed, and the
result is bounded by `tree_branch_width`.

A branch is reusable only when it reconnects exactly to the target-verified path:

```text
branch.offset <= accepted_len
branch.draft_ids starts with base_draft[branch.offset:accepted_len] + correction_token
```

If a branch is still running when the verifier result arrives, the critical path
does not wait for it. Only branch tokens already available at verifier return
time are considered.

---

## Prompt Datasets

The system supports three prompt families for evaluation:

**GSM8K (Grade School Math 8K)**
- Test split: 1,319 prompts
- Train split: 7,473 prompts
- Format: Math word problems
- Default: test split

**HumanEval**
- Test split: 164 prompts
- Format: Python function completion problems
- Source: OpenAI's HumanEval benchmark

**LongWriter / Long-Input Sources**
- Original LongWriter data: `data/longwriter_6k/train.jsonl`
- Derived single-turn partitions:
  - `longwriter_single_turn:input_4k`
  - `longwriter_single_turn:input_6k`
  - `longwriter_single_turn:input_8k`
  - `longwriter_single_turn:input_10k`
- Current quick wrapper default: `longwriter_single_turn:input_10k`

**Prompt Selection**
- Prompts are randomly sampled from each dataset using `random.sample()`
- GSM8K and HumanEval selection is filtered by character length, defaulting to 200-500 characters in the quick loader
- LongWriter sources are not length-filtered by the quick loader
- Each prompt is assigned a sequential ID (1, 2, 3, ...) within the selected batch
- The random selection ensures different prompts are used across test runs
- Use `--prompt-count` in quick_test.py or quick.sh to control how many prompts are loaded per type

**Configuration Example**
```bash
PROMPT_COUNT=5 ./quick.sh

PROMPT_TYPES="gsm8k" ./quick.sh

K=8 TREE_BRANCH_DRAFT_LENGTH=8 ./quick.sh
```

---

## Metrics

`quick_test.py` normalizes direct and tree async rows into:

- `output`: generated tokens, total wall time, tok/s
- `timing`: local draft time, server model time, HTTP/RPC overhead, simulated UL/DL/network, RTT
- `speculative`: rounds, K, drafted tokens, accepted draft tokens, correction tokens, acceptance length
- `async_detail`: branch reuse flag, average reused tokens, pre-draft window, reuse prep time
- `verify_runtime`: prefix/draft/input lengths and vLLM runtime settings
- `raw`: method-specific direct timing or tree network stats

Output files:

```text
quick_test_results.json   # written by quick_test.py
quick_test_summary.json   # written by analyze_quick_run.py
quick_test_rounds.jsonl   # written by analyze_quick_run.py
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
cloud API latency. Alternative architectural options:

- vLLM OpenAI-compatible API
- Batch multiple rounds per HTTP call
- Raw socket / gRPC
- Run vLLM in-process
- Accept HTTP as realistic cost (current approach)

---

### Current Scope

The synchronous speculative client still exists in `scripts/client/edge_client.py`,
but it is not part of the current quick-test path. The quick experiment is now
kept intentionally narrow: `good` network, direct baseline, tree async method,
and separate post-run analysis.
