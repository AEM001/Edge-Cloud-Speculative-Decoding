# Metrics System

The current experiment records SpecExtend observability rows. The main question is:
for each cloud guidance update, what changed, what was saved, and what did it cost?

## Output Files

```text
outputs/observability/<run_id>/raw/quick_test_results.json
outputs/observability/<run_id>/raw/round_observations.jsonl
outputs/observability/<run_id>/analysis/rounds.csv
outputs/observability/<run_id>/analysis/summary.csv
outputs/observability/<run_id>/report.md
```

`quick_test_results.json` keeps the normalized benchmark payload. `round_observations.jsonl`
and `analysis/rounds.csv` flatten the per-round records for analysis.

## Per-Round Signals

Edge-side fields:

- `accepted_len`, `accept_ratio`, `rejected_position`
- `round`, `generated_token_offset`, `generation_phase`
- `acceptance_trend_slope`
- `draft_entropy`, `draft_top1_confidence`, `draft_top1_top2_margin`
- `draft_tree_nodes`, `draft_tree_actual_depth`, `accepted_indices`
- `full_kv_tokens`, `working_kv_tokens`, `selected_full_ratio`
- `kv_append_ms`, `kv_select_ms`, `tree_construct_ms`
- `cuda_memory_allocated_mb`

Cloud-side fields:

- `cloud_target_verify_time_ms`
- `cloud_guidance_generation_time_ms`
- `top_attention_token_indices`
- `cloud_selected_count`, `cloud_selected_token_count`
- `attention_mass_covered`
- `guidance_jaccard`

System-cost fields:

- `edge_start_ts_ms`, `edge_finish_ts_ms`
- `request_payload_bytes`, `response_payload_bytes`
- `verify_elapsed_ms`, `network_time_ms`
- `tokens_since_guidance_update`

## Defaults

The fixed run uses:

```json
{
  "draft_tree_max_depth": 6,
  "retrieve_top_k": 16,
  "retrieve_every_n_steps": 16
}
```

Full raw attention scores are not saved by default. The cloud returns derived metrics,
including selected chunk IDs and attention mass covered by those chunks.
