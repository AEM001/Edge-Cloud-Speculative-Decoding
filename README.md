# SpecExtend Observability Experiment

This workspace is currently prepared for the next observability run, not a policy sweep.
The goal is to log what each cloud guidance update changes, what it saves, and what it costs.

Fixed parameters for this run:

- `draft_tree_max_depth=6`
- `retrieve_top_k=16`
- `retrieve_every_n_steps=16`

Old guidance-reuse outputs and scripts were archived under `Achieve/`.

## Run

Detailed run and metric documentation lives under `Docs/`:

- `Docs/observability.md`
- `Docs/run.md`
- `Docs/metrics_system.md`

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

Each run writes:

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

`round_observations.jsonl` and `analysis/rounds.csv` are the main observability artifacts.
They contain edge draft uncertainty, KV working-set stats, tree stats, cloud verification results,
attention mass covered by selected KV, guidance overlap, timestamps, and payload costs.

The latest analyzed result is copied to `Analysis/` for cross-run reading:

```text
Analysis/LATEST_RUN.txt
Analysis/observability_latest_rounds.csv
Analysis/observability_latest_summary.csv
Analysis/observability_latest_report.md
```
