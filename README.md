# PicoSpec — Speculative Decoding Research Toolkit

End-to-end implementation of **synchronous and asynchronous speculative decoding** with a small draft model (1.5 B) and a larger verify model (7 B), both running locally on the same machine across two GPUs.

---

## Repository Layout

```
.
├── config.py                 # ← single source of truth for all settings (env-overridable)
├── protocol.py               # shared data-structures (EdgeRequest, CloudResponse, …)
├── model_manager.py          # loads the draft LLM via vLLM
├── draft_generator.py        # generates K draft tokens + confidence stats
│
├── client/
│   ├── edge_client.py        # synchronous speculative decoding loop
│   ├── async_edge_client.py  # async / parallel-drafting loop (PicoSpec §3.2)
│   └── http_cloud_client.py  # HTTP client that talks to the verify server
│
├── server/
│   └── verify_server.py      # FastAPI verify server (hosts the 7B model)
│
├── policies/
│   └── policies.py           # K-selection policies (static_k, static_k_policy)
│
├── benchmarks/
│   ├── prompts.txt           # benchmark prompt set (easy + hard)
│   ├── metrics_logger.py     # structured metrics collection
│   ├── network_simulator.py  # optional network latency simulation
│   └── workloads.py          # prompt sampling helpers
│
├── experiments/
│   ├── experiment_async_pipeline.py       # Run 2 — async vs sync
│   └── experiment_async_pipeline_run3.py  # Run 3 — K=[2,4,8], 3 latency tiers
│
├── quick_test.py             # smoke test: Direct vs K=2,4 on 4 prompts
│
└── scripts/
    ├── start_verify.sh       # launch verify server in tmux (GPU 0)
    └── run_experiment.sh     # launch any experiment in tmux (GPU 1)
```

---

## Quick Start

### 1 — Start the verify server (GPU 0)

```bash
cd /root/code/draft
bash scripts/start_verify.sh
```

This launches `server/verify_server.py` in a tmux session called `verify-server`.  
The script polls `http://localhost:6006/health` and exits once the model is loaded.

```bash
# check health manually
curl http://localhost:6006/health
```

### 2 — Run the smoke test (GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/code/draft python3 quick_test.py
```

Runs Direct + Sync K=2 + Sync K=4 on 2 easy + 2 hard prompts and prints a speedup table.

### 3 — Run a full experiment (GPU 1, inside tmux)

```bash
bash scripts/run_experiment.sh
# or specify a script explicitly:
bash scripts/run_experiment.sh experiments/experiment_async_pipeline_run3.py
```

Results land in `experiments/outputs_run3_<date>/`.

---

## Configuration

All settings live in `config.py` and can be overridden with environment variables — **no file editing required**.

| Variable | Default | Meaning |
|---|---|---|
| `DRAFT_MODEL_PATH` | `/root/code/Qwen2.5-1.5B-Instruct-AWQ` | Draft model directory |
| `DRAFT_GPU_MEM` | `0.90` | vLLM GPU memory fraction for draft model |
| `DRAFT_MAX_LEN` | `4096` | Max sequence length for draft model |
| `VERIFY_MODEL_PATH` | `/root/code/Qwen2.5-7B-Instruct-AWQ` | Verify model directory |
| `VERIFY_GPU_MEM` | `0.90` | vLLM GPU memory fraction for verify model |
| `VERIFY_QUANTIZATION` | `awq` | Quantization for verify model |
| `VERIFY_SERVER_URL` | `http://localhost:6006` | URL the edge client connects to |
| `MAX_NEW_TOKENS` | `128` | Default generation length |
| `TEMPERATURE` | `0.0` | Sampling temperature (0 = greedy) |

Example override:
```bash
DRAFT_GPU_MEM=0.5 DRAFT_MAX_LEN=2048 python3 quick_test.py
```

---

## Reusing the Modules

### Load draft model and run speculative decoding

```python
from config import DRAFT_MODEL_PATH, DRAFT_GPU_MEM, DRAFT_MAX_LEN, VERIFY_SERVER_URL
from model_manager import VLLMModelManager
from draft_generator import VLLMDraftGenerator
from client.edge_client import EdgeClient
from client.http_cloud_client import create_http_cloud_client

model_manager = VLLMModelManager(DRAFT_MODEL_PATH, DRAFT_GPU_MEM, DRAFT_MAX_LEN)
llm, tokenizer = model_manager.load()
draft_gen = VLLMDraftGenerator(llm, tokenizer)
cloud_client = create_http_cloud_client(VERIFY_SERVER_URL)

client = EdgeClient(
    model_manager=model_manager,
    draft_generator=draft_gen,
    cloud_client=cloud_client,
    max_new_tokens=128,
)

metrics = client.generate(
    prompt="What is the capital of France?",
    policy=lambda round_id, tokens: 4,   # fixed K=4
    policy_name="StaticK4",
)
print(f"{metrics.generated_tokens} tokens  {metrics.acceptance_ratio:.1%} accept  {metrics.total_latency_ms:.0f} ms")
```

### Use the async (pipeline) client

```python
from client.async_edge_client import AsyncEdgeClient

async_client = AsyncEdgeClient(
    model_manager=model_manager,
    draft_generator=draft_gen,
    cloud_client=cloud_client,
    max_new_tokens=128,
    lookahead=2,          # 2 in-flight draft batches
)

am = async_client.generate(
    prompt="Explain quantum entanglement.",
    policy=lambda round_id, tokens: 8,
    policy_name="AsyncK8",
)
print(f"pipeline_efficiency={am.pipeline_efficiency:.1%}  bubble={am.avg_bubble_ms:.0f}ms")
```

### Use a policy from the registry

```python
from policies.policies import static_k_policy, get_policy

policy = static_k_policy(k=6)           # returns a policy function
policy = get_policy("static_k_4")       # look up by name
```

### Start the verify server programmatically

```python
# In a subprocess or separate process:
# python3 -m server.verify_server --host 0.0.0.0 --port 6006
```

---

## GPU Assignment

| Process | GPU | How |
|---|---|---|
| Verify server (`server/verify_server.py`) | GPU 0 | default CUDA device |
| Draft model / experiments | GPU 1 | `CUDA_VISIBLE_DEVICES=1` |

The `scripts/run_experiment.sh` script sets `CUDA_VISIBLE_DEVICES=1` automatically.

---

## Running the Verify Server Without tmux

```bash
PYTHONPATH=/root/code/draft python3 -m server.verify_server --host 0.0.0.0 --port 6006
```

Or override the model path:
```bash
VERIFY_MODEL_PATH=/path/to/other-model PYTHONPATH=/root/code/draft \
    python3 -m server.verify_server --port 6006
```

---

## Models

Both models are stored under `/root/code/`:

```
/root/code/Qwen2.5-1.5B-Instruct-AWQ/   ← draft
/root/code/Qwen2.5-7B-Instruct-AWQ/     ← verify
```

To use different models, set `DRAFT_MODEL_PATH` / `VERIFY_MODEL_PATH`.
