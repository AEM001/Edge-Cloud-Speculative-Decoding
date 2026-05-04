# Network-Constrained Speculative Decoding Experiments

Compares **direct generation**, **synchronous speculative decoding** (K = 3, 5, 7),
and **asynchronous (pipelined) speculative decoding** (K = 3, 5, 7) under a range
of simulated network conditions — from ideal local loopback to degraded 3G mobile.

---

## Files

| File | Role |
|------|------|
| `network_conditions.py` | Reusable network simulation layer — wraps any `cloud_client` and injects configurable latency, bandwidth throttle, jitter, and packet-loss delay |
| `metrics_collector.py` | Unified data model (`ExperimentResult`), converters from raw client metrics, per-(method × condition) aggregation, JSON/CSV export, and human-readable table printing |
| `run_network_experiment.py` | Main experiment runner — loads the draft model once, sweeps every network profile × method × prompt, streams results to `MetricsCollector`, saves outputs |

---

## Prerequisites

1. **Verify server running** on `localhost:6006` (or set `VERIFY_SERVER_URL`):
   ```bash
   python -m server.verify_server --port 6006
   ```
2. **Draft model** available at the path in `config.py` (`DRAFT_MODEL_PATH`).
3. Dependencies installed (`vllm`, `requests`, `fastapi`, `uvicorn`, etc.).

---

## Quick Start

```bash
# From the project root
python -m experiments.run_network_experiment \
    --prompts 2 \
    --max-tokens 128 \
    --k-values 3 5 7
```

This runs every method across **all 7 network profiles** using 2 simple + 2 complex
prompts and writes results to `experiments/outputs_network/`.

---

## CLI Reference

```
usage: run_network_experiment.py [-h]
    [--server URL]
    [--max-tokens N]
    [--prompts N]
    [--k-values K [K ...]]
    [--lookahead N]
    [--profiles {ideal,LAN,WiFi,WAN_Low,WAN_High,4G,3G} [...]]
    [--no-async]
    [--no-direct]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `http://localhost:6006` | Verify server base URL |
| `--max-tokens` | `128` | Max new tokens per generation call |
| `--prompts` | `2` | Prompts per type (simple + complex) |
| `--k-values` | `3 5 7` | Draft lengths K to test |
| `--lookahead` | `2` | Async pipeline lookahead depth |
| `--profiles` | all | Network profiles to include |
| `--no-async` | — | Skip async speculative runs |
| `--no-direct` | — | Skip direct generation runs |

### Example: sweep only WAN profiles

```bash
python -m experiments.run_network_experiment \
    --profiles WAN_Low WAN_High 4G 3G \
    --prompts 4 \
    --max-tokens 256
```

---

## Network Profiles

Profiles are defined as `NetworkCondition` dataclasses in `network_conditions.py`.
Each profile specifies:

| Profile | RTT (approx) | Bandwidth | Jitter | Packet loss |
|---------|--------------|-----------|--------|-------------|
| `good` | ~25 ms | 100 Mbps | ±3 ms | 0% |
| `medium` | ~50 ms | 50 Mbps | ±5 ms | 0% |
| `bursty` | 40-80 ms | 20 Mbps | ±20 ms | 1% |

The RTT seen by each round is approximately:

```
RTT ≈ (one_way_latency × 2) + payload_bytes / bandwidth + jitter + server_verify_time
```

### Adding a custom profile

```python
from experiments.network_conditions import NetworkCondition

my_link = NetworkCondition(
    name="Satellite",
    one_way_latency_ms=300.0,
    bandwidth_mbps=5.0,
    jitter_ms=50.0,
    packet_loss_prob=0.02,   # 2 % simulated loss → one retransmit overhead added
)
```

Pass it via `--profiles` (add to `DEFAULT_NETWORK_PROFILES` in the runner) or inject
directly into any `ThrottledCloudClient`:

```python
from experiments.network_conditions import ThrottledCloudClient
throttled = ThrottledCloudClient(base_client, my_link)
edge_client.cloud_client = throttled
```

---

## Metrics Tracked

### Core (all methods)

| Metric | Field | Description |
|--------|-------|-------------|
| Tokens/second | `tokens_per_second` | End-to-end throughput |
| Total latency | `total_latency_ms` | Wall-clock time for full generation |
| Tokens generated | `tokens_generated` | Output token count |

### Speculative decoding (sync + async)

| Metric | Field | Description |
|--------|-------|-------------|
| Acceptance ratio | `acceptance_ratio` | Fraction of drafted tokens accepted by verifier |
| Mean K chosen | `mean_k_chosen` | Average draft length per round |
| Total rounds | `total_rounds` | Number of draft–verify cycles |
| Draft time | `total_draft_time_ms` | Cumulative draft model compute time |
| Verify time | `total_verify_time_ms` | Cumulative verifier compute time |
| Network time | `total_network_time_ms` | Cumulative measured network overhead |
| Average RTT | `average_rtt_ms` | Mean round-trip time including simulated delay |
| Uplink bytes | `uplink_bytes` | Total serialised request payload |
| Downlink bytes | `downlink_bytes` | Total serialised response payload |

### Async pipeline (async only)

| Metric | Field | Description |
|--------|-------|-------------|
| Pipeline efficiency | `pipeline_efficiency` | Fraction of slots that were full hits (no rollback) |
| Avg bubble ms | `avg_bubble_ms` | Mean stall time per round when drafter waits for verifier |
| Prefetch waste ratio | `prefetch_waste_ratio` | Fraction of speculatively drafted tokens discarded on rollback |
| Async speedup vs sync | `async_speedup_vs_sync` | `(T_draft + T_verify) / max(T_draft, T_verify)` — theoretical pipeline benefit |
| Total rollbacks | `total_rollbacks` | Number of speculative prefix rollbacks |

### Simulated network overhead

| Metric | Field | Description |
|--------|-------|-------------|
| Simulated overhead | `simulated_overhead_ms` | Total injected delay (uplink + downlink) per request |
| Uplink delay | `simulated_uplink_delay_ms` | Latency + BW delay on the uplink direction |
| Downlink delay | `simulated_downlink_delay_ms` | Latency + BW delay on the downlink direction |

---

## Output Files

All outputs go to `experiments/outputs_network/`:

```
outputs_network/
├── results_YYYYMMDD_HHMMSS.json    # raw per-observation records
├── results_YYYYMMDD_HHMMSS.csv     # same data, spreadsheet-friendly
├── summary_YYYYMMDD_HHMMSS.json    # aggregated per-(method × condition)
├── results_latest.json             # symlink → most recent JSON
└── results_latest.csv              # symlink → most recent CSV
```

### Programmatic access

```python
from experiments.metrics_collector import MetricsCollector, ExperimentResult
import json

with open("experiments/outputs_network/results_latest.json") as f:
    records = [ExperimentResult(**r) for r in json.load(f)]

# Filter async results under WAN_High
async_wan = [r for r in records
             if r.mode == "async" and r.network_condition == "WAN_High"]
```

---

## Architecture

```
run_network_experiment.py
│
├── NetworkCondition          (network_conditions.py)
│   └── ThrottledCloudClient  wraps base_client, injects latency + BW delay
│
├── EdgeClient                (client/edge_client.py)
│   └── draft_generator → ThrottledCloudClient → verify server
│
├── AsyncEdgeClient           (client/async_edge_client.py)
│   └── draft_generator → ThrottledCloudClient (via background thread)
│
└── MetricsCollector          (metrics_collector.py)
    ├── add(ExperimentResult)
    ├── save_json / save_csv / save_summary_json
    └── print_table / print_metric_spotlight
```

The **draft model is loaded once** at startup; only the `cloud_client` reference on
`EdgeClient` / `AsyncEdgeClient` is swapped between conditions, so GPU memory and
model state are never reloaded mid-experiment.

---

## Reusing Individual Modules

### Simulate network in any existing script

```python
from experiments.network_conditions import NetworkCondition, ThrottledCloudClient
from client.http_cloud_client import create_http_cloud_client

base = create_http_cloud_client("http://localhost:6006")
wan  = ThrottledCloudClient(base, NetworkCondition.wan_high())

# Pass `wan` anywhere a cloud_client callable is expected
edge_client.cloud_client = wan
```

### Record results in any existing script

```python
from experiments.metrics_collector import MetricsCollector, from_sync_metrics

collector = MetricsCollector()

metrics = edge_client.generate(prompt, policy=lambda *_: 5)
collector.add(from_sync_metrics(metrics, k=5, network_condition="custom",
                                prompt_id=1, prompt_type="Simple"))

collector.save_json(Path("my_results.json"))
collector.print_table()
```
