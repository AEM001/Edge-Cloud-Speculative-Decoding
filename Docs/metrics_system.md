# Metrics System Documentation

This document describes the metrics system used throughout the PicoSpec speculative decoding research codebase, including what metrics are recorded and where they exist in the repository.

## Overview

The metrics system tracks performance characteristics of speculative decoding across multiple dimensions: throughput, latency, accepted draft length, network overhead, and tree-based pipeline efficiency. Metrics are collected at three levels:

1. **Per-request metrics** - High-level summary for a single generation request
2. **Per-round metrics** - Detailed breakdown for each verification round
3. **Network simulation metrics** - Simulated network delay and bandwidth statistics

## Metrics Data Structures

### 1. RequestMetrics

**Location:** `scripts/client/edge_client.py`

**Purpose:** Per-request metrics for synchronous speculative decoding (baseline edge client).

**Fields:**
- **request_id**: Unique identifier for the request
- **prompt`: Input prompt text
- **Timing Metrics:**
  - `total_latency_ms`: End-to-end generation time
  - `total_edge_draft_time_ms`: Cumulative time spent generating draft tokens on edge
  - `total_server_verify_time_ms`: Cumulative time spent verifying drafts on cloud
  - `total_network_time_ms`: Cumulative network RTT minus server verify time
  - `average_rtt_ms`: Average round-trip time across all rounds
- **Detailed Timing Breakdown:**
  - `total_model_time_ms`: Cumulative actual vLLM model processing time on server
  - `total_http_overhead_ms`: Cumulative FastAPI + serialization overhead on server
  - `total_network_tx_ms`: Cumulative time to send request (uplink transmission)
  - `total_network_rx_ms`: Cumulative time to receive response (downlink transmission)
- **Token Metrics:**
  - `generated_tokens`: Total tokens generated
  - `total_rounds`: Number of verification rounds
- **Acceptance Metrics:**
  - `acceptance_length`: Average accepted draft tokens per verification round
  - `total_drafted_tokens`: Total draft tokens generated
  - `total_accepted_drafted_tokens`: Total draft tokens accepted by verifier
- **Network Metrics:**
  - `uplink_bytes`: Total bytes sent to cloud
  - `downlink_bytes`: Total bytes received from cloud
- **round_details**: List of per-round details (see below)

**Per-Round Details (in `round_details` list):**
- `round_id`: Round number
- `K`: Draft length used
- `drafted`: Number of draft tokens
- `accepted`: Number of accepted tokens
- `draft_time_ms`: Time to generate draft
- `server_time_ms`: Server verification time
- `rtt_ms`: Round-trip time
- `correction`: Correction token ID (if any)
- `verify_prefix_len`: Prefix length at verification
- `verify_draft_len`: Draft length at verification
- `verify_input_len`: Total input length (prefix + draft)

---

### 2. AsyncRequestMetrics

**Location:** `scripts/experiments/tree_async_client.py`

**Purpose:** Per-request metrics for tree-based asynchronous speculative decoding.

**Fields:**
- **request_id**: Unique identifier for the request
- **prompt`: Input prompt text
- **Core Throughput:**
  - `total_latency_ms`: End-to-end generation time
  - `generated_tokens`: Total tokens generated
  - `tokens_per_second`: Throughput (tokens/second)
- **Round/Token Counts:**
  - `total_rounds`: Number of verification rounds
  - `total_accepted_tokens`: Total tokens accepted by verifier
  - `total_drafted_tokens`: Total draft tokens generated for verification
- **Timing Breakdown:**
  - `total_edge_draft_time_ms`: Cumulative edge draft time
  - `total_server_verify_time_ms`: Cumulative cloud verification time
  - `total_network_time_ms`: Cumulative network time
  - `average_rtt_ms`: Average round-trip time
  - `avg_bubble_ms`: Average time waiting for async operations per round (key async benefit metric)
- **Branch Reuse Metrics:**
  - `branch_reused`: Boolean - whether any prefetched branch was reused
  - `reused_tokens`: Number of tokens reused from prefetched branches (may be partial if stopped early)
- **Pre-Draft Timing:**
  - `predraft_window_ms`: Time between sending verification request and receiving response (window where pre-drafting happens)
  - `reuse_prep_time_ms`: Time between receiving verification and sending next request (time to prepare with reused tokens)
- **Network Metrics:**
  - `uplink_bytes`: Total bytes sent to cloud
  - `downlink_bytes`: Total bytes received from cloud

---

### 3. ExperimentResult (Unified Schema)

**Location:** `scripts/experiments/quick_test.py`

**Purpose:** Unified result schema that normalizes metrics across direct, sync speculative, and tree async methods for comparison.

**Fields:**
- `method`: Method name (e.g., "direct", "sync_k8", "tree_k8_b3")
- `method_family`: Method family ("direct", "sync_spec", "tree_async")
- `network`: Network condition name
- `prompt_type`: Dataset type (e.g., "gsm8k", "humaneval")
- `prompt_id`: Prompt identifier
- `output`: OutputMetrics (see below)
- `timing`: TimingMetrics (see below)
- `speculative`: SpeculativeMetrics (see below)
- `async_detail`: AsyncDetailMetrics (see below)
- `verify_runtime`: VerifyRuntimeMetrics (see below)
- `raw`: Dictionary of method-specific raw data

---

### 4. OutputMetrics

**Location:** `scripts/experiments/quick_test.py`

**Purpose:** Core output performance metrics.

**Fields:**
- `tokens_generated`: Total tokens generated
- `total_time_ms`: Total generation time
- `tokens_per_second`: Throughput (tokens/second)

---

### 5. TimingMetrics

**Location:** `scripts/experiments/quick_test.py`

**Purpose:** Detailed timing breakdown across the pipeline.

**Fields:**
- `client_wall_ms`: Total client wall-clock time
- `local_draft_ms`: Time spent on local draft generation
- `server_model_ms`: Time spent on server model computation
- `server_total_ms`: Total server time (including overhead)
- `http_rpc_ms`: HTTP RPC overhead time
- `simulated_ul_ms`: Simulated uplink delay (network simulation)
- `simulated_dl_ms`: Simulated downlink delay (network simulation)
- `simulated_network_ms`: Total simulated network delay
- `avg_rtt_ms`: Average round-trip time
- `critical_path_wait_ms`: Critical path wait time (bubble time for tree async)

---

### 6. SpeculativeMetrics

**Location:** `scripts/experiments/quick_test.py`

**Purpose:** Speculative decoding efficiency metrics.

**Fields:**
- `rounds`: Total number of verification rounds
- `k`: Draft length (K) parameter
- `drafted_tokens`: Total draft tokens generated
- `accepted_draft_tokens`: Total draft tokens accepted
- `correction_tokens`: Total correction tokens generated
- `generated_per_round`: Average tokens generated per round
- `acceptance_length`: Average accepted draft tokens per round
- `wasted_draft_tokens`: Draft tokens that were rejected

---

### 7. AsyncDetailMetrics

**Location:** `scripts/experiments/quick_test.py`

**Purpose:** Tree-specific async pipeline metrics (simplified).

**Fields:**
- `branch_reused`: Boolean - whether any prefetched branch was reused (across all rounds)
- `reused_tokens`: Average number of tokens reused from prefetched branches
- `predraft_window_ms`: Average time between sending verification request and receiving response
- `reuse_prep_time_ms`: Average time between receiving verification and sending next request

---

### 8. VerifyRuntimeMetrics

**Location:** `scripts/experiments/quick_test.py`

**Purpose:** Server-side verification runtime configuration and performance metrics.

**Fields:**
- `avg_prefix_len`: Average prefix length at verification
- `avg_draft_len`: Average draft length at verification
- `avg_input_len`: Average total input length (prefix + draft)
- `prefix_caching`: Whether prefix caching is enabled
- `enforce_eager`: Whether eager execution is enforced
- `attention_backend`: Attention backend used (e.g., "FLASH_ATTN")
- `vllm_version`: vLLM version string

---

### 9. NetworkCallStats

**Location:** `scripts/experiments/network_conditions.py`

**Purpose:** Network simulation statistics for throttled cloud client.

**Fields:**
- `num_calls`: Number of network calls made
- `total_simulated_uplink_delay_ms`: Cumulative simulated uplink delay
- `total_simulated_downlink_delay_ms`: Cumulative simulated downlink delay
- `total_simulated_overhead_ms`: Total simulated network overhead
- `total_uplink_bytes`: Total uplink bytes
- `total_downlink_bytes`: Total downlink bytes

---

## Protocol Data Structures (Communication Layer)

**Location:** `scripts/core/protocol.py`

These are not metrics per se, but data structures used for communication between edge and cloud:

### TokenInfo
- `token_id`: Token identifier
- `logprob`: Log probability

### DraftRequest
- `verified_prefix`: Verified prefix token IDs
- `num_draft_tokens`: Number of draft tokens to generate

### DraftResponse
- `draft_token_ids`: Generated draft token IDs
- `logprobs`: Log probabilities for draft tokens

### EdgeRequest
- `request_id`: Request identifier
- `prefix_ids`: Prefix token IDs
- `draft_ids`: Draft token IDs to verify

### CloudResponse
- `request_id`: Request identifier
- `accepted_len`: Number of accepted draft tokens
- `correction_token_id`: Correction token (if any)
- `server_verify_time_ms`: Server verification time
- `rtt_ms`: Round-trip time (network delay only)
- `end_to_end_ms`: End-to-end latency (server + network)

---

## Metrics Storage and Output

### Output Files

**Location:** `scripts/experiments/outputs_quick/`

#### 1. quick_test_results.json
- **Format:** JSON array of `ExperimentResult` objects
- **Purpose:** Primary results file containing normalized metrics for all methods and conditions
- **Structure:** Each entry is a complete `ExperimentResult` with all metric categories

#### 2. quick_test_rounds.jsonl
- **Format:** JSONL (one JSON object per line)
- **Purpose:** Per-round details for fine-grained analysis
- **Structure:** Each line contains:
  - Metadata: `network`, `method`, `method_family`, `prompt_type`, `prompt_id`, `detail_index`
  - Detail: Method-specific round/slot details (from `round_details` or `slot_details`)

#### 3. quick_test_summary.json
- **Format:** JSON object
- **Purpose:** Aggregated summary statistics by network condition and method
- **Structure:** Nested dictionary with averages across all prompts

### Analysis Outputs

**Location:** `scripts/experiments/outputs_quick/analysis/`

Generated by `scripts/experiments/analyze_quick_results.py`:

#### CSV Tables
- `table_run_summary.csv`: Summary statistics by network and method
- `table_prompt_type_summary.csv`: Summary statistics by network, prompt type, and method
- `table_acceptance_distribution.csv`: Distribution of accepted token counts per round
- `table_tree_offsets.csv`: Tree branch offset selection statistics
- `table_verify_rounds.csv`: Verification round details (input length, verify time, RTT)

#### PNG Charts
- `chart_throughput_by_method.png`: Bar chart of tokens/second by method and network
- `chart_speedup_by_prompt_type.png`: Speedup vs direct by prompt type
- `chart_acceptance_distribution.png`: Distribution of accepted draft tokens
- `chart_verify_scaling.png`: Verify time vs input length scatter plot
- `chart_tree_selected_offsets.png`: Tree branch offset selection histogram

---

## Metrics Collection Flow

### Synchronous Speculative Decoding

1. **EdgeClient.generate()** creates `RequestMetrics` instance
2. For each round:
   - Records draft time (`draft_time_ms`)
   - Sends request to cloud via cloud client
   - Records server verify time (`server_verify_time_ms`)
   - Records RTT (`rtt_ms`)
   - Computes network time as RTT - server time
   - Updates cumulative metrics
   - Appends round detail to `round_details`
3. After generation completes:
   - Computes derived metrics such as acceptance length
   - Returns `RequestMetrics`

### Tree-Based Async Speculative Decoding

1. **TreeAsyncEdgeClient.generate()** creates `AsyncRequestMetrics` instance
2. For each tree iteration:
   - Drafts base branch and speculative branches in parallel (pre-draft during verification window)
   - Records `predraft_window_ms` (time between sending verify request and receiving response)
   - Fires base verification in background thread
   - Records bubble time (wait for verification) → `avg_bubble_ms`
   - Records verification time and RTT
   - Tracks branch reuse: `branch_reused` (boolean) and `reused_tokens` (count)
   - Records `reuse_prep_time_ms` (time to prepare with reused tokens before next request)
   - Updates cumulative metrics
3. After generation completes:
   - Computes derived metrics such as throughput
   - Returns `AsyncRequestMetrics`

### Experiment Runner

**Location:** `scripts/experiments/quick_test.py`

1. Runs direct, sync speculative, and tree async methods for each prompt and network condition
2. Converts method-specific metrics to unified `ExperimentResult` schema
3. Saves results to `quick_test_results.json`
4. Extracts per-round details to `quick_test_rounds.jsonl`
5. Builds summary statistics to `quick_test_summary.json`

### Analysis Pipeline

**Location:** `scripts/experiments/analyze_quick_results.py`

1. Loads `quick_test_results.json` and `quick_test_rounds.jsonl`
2. Flattens to pandas DataFrames
3. Computes derived metrics (speedup vs direct, etc.)
4. Generates aggregated CSV tables
5. Generates visualization charts

---

## Key Metric Relationships

### Speedup Calculation
```
speedup_vs_direct = speculative_tps / direct_tps
```

### Acceptance Length
```
acceptance_length = total_accepted_tokens / total_rounds
```

### Network Time
```
network_time_ms = rtt_ms - server_verify_time_ms
```

### Bubble Time (Tree Async)
Time the critical path waits for async operations to complete. This is the key metric for measuring async benefit - it should be close to zero if pre-drafting successfully hides verification latency.

### Pre-Draft Window (Tree Async)
```
predraft_window_ms = time(send_verify_request) to time(receive_verify_response)
```
This is the window where pre-drafting happens - the edge drafts speculative branches while waiting for cloud verification.

### Reuse Prep Time (Tree Async)
```
reuse_prep_time_ms = time(receive_verify_response) to time(send_next_verify_request)
```
Time to prepare the next request using reused tokens from prefetched branches.

---

## Metrics by Component

### Edge Client (`scripts/client/`)
- **edge_client.py**: `RequestMetrics` - synchronous speculative decoding metrics

### Experiments (`scripts/experiments/`)
- **quick_test.py**: Unified schema (`ExperimentResult`, `OutputMetrics`, `TimingMetrics`, `SpeculativeMetrics`, `AsyncDetailMetrics`, `VerifyRuntimeMetrics`)
- **tree_async_client.py**: `AsyncRequestMetrics` - tree-based async metrics
- **network_conditions.py**: `NetworkCallStats` - network simulation metrics

### Server (`scripts/server/`)
- **verify_server.py**: Returns `server_verify_time_ms` in `CloudResponse`

### Protocol (`scripts/core/`)
- **protocol.py**: Communication data structures (not metrics, but carry timing info)

---

## Usage Examples

### Running Experiments
```bash
python3 scripts/experiments/quick_test.py
```
Outputs to `scripts/experiments/outputs_quick/`

### Analyzing Results
```bash
python3 scripts/experiments/analyze_quick_results.py
```
Generates tables and charts in `scripts/experiments/outputs_quick/analysis/`

### Loading Metrics in Python
```python
import json
from pathlib import Path

# Load results
results = json.loads(Path("scripts/experiments/outputs_quick/quick_test_results.json").read_text())

# Access metrics
for result in results:
    print(f"Method: {result['method']}")
    print(f"Tokens/sec: {result['output']['tokens_per_second']}")
    print(f"Acceptance length: {result['speculative']['acceptance_length']}")
    print(f"Server time: {result['timing']['server_model_ms']}")
```

---

## Notes

- All time metrics are in milliseconds unless otherwise specified
- Network metrics can be simulated (for testing) or real (for production)
- Tree async metrics include additional fields for branch reuse and pipeline parallelism
- The unified schema (`ExperimentResult`) allows direct comparison across different methods
- Per-round details (`round_details` / `slot_details`) are kept in the `raw` field to preserve method-specific information
