# Experiments

The current experiment focus is SpecExtend observability logging. Use:

```bash
python3 scripts/experiments/prepare_observability_config.py
bash start_verify.sh
python3 scripts/experiments/run_observability_experiment.py \
  outputs/observability/<run_id>/configs/observability.json
```

`quick_test.py` remains the model-running entrypoint. The observability runner wraps it,
archives the raw result into the run directory, and writes per-round JSONL/CSV outputs.

Old guidance-reuse sweep scripts were archived under `Achieve/scripts/experiments/`.
