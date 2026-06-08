# Metrics System

The quick experiment records direct generation and real SpecExtend rows in a
single JSON payload.

## Output File

`scripts/experiments/outputs_quick/quick_test_results_*.json`

Top-level shape:

- `config`: run configuration, including prompt source, max tokens, tree nodes,
  max depth, and retrieval settings.
- `results`: list of normalized result rows.

## Result Row

- `method`: `direct` or `specextend_n<N>`.
- `method_family`: `direct` or `specextend`.
- `network`: simulated network profile name.
- `prompt_type` / `prompt_id`: prompt source metadata.
- `output`: generated token count, total wall time, and throughput.
- `timing`: client wall time, edge draft time, cloud verify time, HTTP time,
  simulated uplink/downlink/network time, KV load breakdown, and average
  round-trip time.
- `speculative`: SpecExtend rounds, node budget, accepted draft tokens,
  correction tokens, generated tokens per round, and acceptance length.
- `verify_runtime`: backend metadata such as `custom_qwen3`,
  tree-verification support, and attention-score availability.
- `raw`: method-specific details such as network counters, selected retrieval
  chunk IDs, and per-round records.

For SpecExtend KV-tier experiments, `raw.kv_tier_probe` includes:

- `retrieval_updates`: number of target-attention retrieval updates.
- `updates_with_cpu_kv` / `updates_with_ssd_kv`: retrieval-update rounds whose
  selected blocks included CPU/SSD-resident KV.
- `selected_gpu_chunks` / `selected_cpu_chunks` / `selected_ssd_chunks`: total
  selected block counts by tier across charged KV rebuilds.
- `gpu_load_ms` / `cpu_load_ms` / `ssd_load_ms` / `total_load_ms`: measured KV
  movement cost.

## SpecExtend Request Metrics

`scripts/client/specextend_edge_client.py` collects:

- `total_latency_ms`
- `generated_tokens`
- `total_rounds`
- `total_edge_draft_time_ms`
- `total_kv_load_time_ms`
- `total_kv_gpu_load_ms`
- `total_kv_cpu_load_ms`
- `total_kv_ssd_load_ms`
- `total_kv_gpu_chunks`
- `total_kv_cpu_chunks`
- `total_kv_ssd_chunks`
- `retrieval_updates_with_cpu_kv`
- `retrieval_updates_with_ssd_kv`
- `total_server_verify_time_ms`
- `total_network_time_ms`
- `total_accepted_tokens`
- `retrieval_updates`
- `selected_chunk_ids`
- `round_details`

Each round detail includes the round number, tree node count, accepted length,
whether retrieval was requested, and selected chunk IDs.
