# Metrics System

The quick experiment records direct generation and real SpecExtend rows in a
single JSON payload.

## Output Files

```text
outputs/quick_test_results_<timestamp>.json
outputs/quick_test_results_<timestamp>_summary.txt
```

Top-level shape of the JSON:

- `config`: run configuration, including prompt source, max tokens, tree nodes,
  max depth, and retrieval settings.
- `results`: list of normalized result rows.

## Result Row

- `method`: `direct` or `specextend`.
- `method_family`: `direct` or `specextend`.
- `network`: simulated network profile name.
- `prompt_type` / `prompt_id`: prompt source metadata.
- `output`: generated token count, total wall time, and throughput.
- `timing`: client wall time, edge draft time, cloud verify time, HTTP time,
  simulated uplink/downlink/network time, and average round-trip time.
- `speculative`: SpecExtend rounds, draft length, accepted draft tokens,
  correction tokens, generated tokens per round, and acceptance length.
- `verify_runtime`: backend metadata (`custom_qwen3`, tree-verify flag,
  attention-score availability).
- `raw`: network counters, selected retrieval chunk IDs, retrieval update
  count, and per-round records.

## SpecExtend Request Metrics

`scripts/client/specextend_edge_client.py` collects per-request:

- `total_latency_ms`
- `generated_tokens`
- `total_rounds`
- `total_edge_draft_time_ms`
- `total_server_verify_time_ms`
- `total_network_time_ms`
- `total_accepted_tokens`
- `retrieval_updates`
- `selected_chunk_ids`
- `round_details`

Each round detail includes: round index, draft token count, accepted length,
whether retrieval was requested, selected chunk IDs, and pipeline hit/built/wait
stats.
