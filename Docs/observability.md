# Observability Experiment

This document is the single documentation entry for the current SpecExtend
observability run. The focus is measurement, not protocol redesign or adaptive
policy.

## Fixed Setup

- `draft_mode=branching`
- `draft_length=8`
- `draft_tree_nodes=32`
- `draft_tree_max_depth=6`
- `retrieval_chunk_size=64`
- `retrieve_top_k=16`
- `retrieve_every_n_steps=16`
- `retrieve_on_first_round=true`

## Run

Start the verify server:

```bash
bash start_verify.sh
```

Prepare a run directory and config:

```bash
python3 scripts/experiments/prepare_observability_config.py \
  --prompt-count 3 \
  --max-tokens 256 \
  --prompt-input-tokens 2048 \
  --dataset-split pg19_2K
```

Run the experiment:

```bash
python3 scripts/experiments/run_observability_experiment.py \
  outputs/observability/<run_id>/configs/observability.json
```

Analyze an existing raw result:

```bash
python3 scripts/experiments/analyze_observability.py \
  outputs/observability/<run_id>/raw/quick_test_results.json
```

## Outputs

Each run writes its complete archive under:

```text
outputs/observability/<run_id>/
  configs/observability.json
  raw/quick_test_results.json
  raw/round_observations.jsonl
  analysis/rounds.csv
  analysis/summary.csv
  report.md
  manifest.json
```

The latest analyzed run is also copied to:

```text
Analysis/LATEST_RUN.txt
Analysis/observability_latest_rounds.csv
Analysis/observability_latest_summary.csv
Analysis/observability_latest_report.md
```

## Logged Signals

Edge-side signals include acceptance length and ratio, rejected position,
generation phase, local acceptance trend, draft entropy, top-1 confidence,
top-1/top-2 margin, tree size/depth, KV working-set size, sparse/full KV ratio,
KV append/select latency, tree construction latency, CUDA memory, accepted
indices, and correction token.

Cloud-side signals include target verify time, guidance generation time,
top-attention token indices, selected KV count, selected token count, attention
mass covered by selected KV, guidance Jaccard overlap, and cloud receive/finish
timestamps.

System-cost signals include edge start/finish timestamps, payload bytes,
verification elapsed time, estimated network time, pipeline wait/reuse fields,
and tokens supported since the last guidance update.
