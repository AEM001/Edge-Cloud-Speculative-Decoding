# Observability Experiment

This experiment records why a SpecExtend round helped or hurt. It intentionally does not add
adaptive update policy logic.

## Fixed Setup

- Branching draft tree
- `draft_tree_max_depth=6`
- `retrieve_top_k=16`
- `retrieve_every_n_steps=16`
- Full raw attention scores are not persisted by default; only derived metrics are saved.

## Commands

```bash
python3 scripts/experiments/prepare_observability_config.py \
  --prompt-count 3 \
  --max-tokens 256 \
  --prompt-input-tokens 2048 \
  --dataset-split pg19_2K
```

```bash
bash start_verify.sh
```

```bash
python3 scripts/experiments/run_observability_experiment.py \
  outputs/observability/<run_id>/configs/observability.json
```

## Important Files

```text
outputs/observability/<run_id>/raw/round_observations.jsonl
outputs/observability/<run_id>/analysis/rounds.csv
outputs/observability/<run_id>/analysis/summary.csv
outputs/observability/<run_id>/report.md
```

Use `rounds.csv` for per-round modeling and correlation analysis. Use `summary.csv`
for a quick run-level check.
