Start the verify server in one terminal:

```bash
bash start_verify.sh
```

If you need a different port or GPU:

```bash
PORT=6008 VERIFY_GPU_ID=0 bash start_verify.sh
```

For the current observability experiment, prepare a run config:

```bash
python3 scripts/experiments/prepare_observability_config.py \
  --prompt-count 3 \
  --max-tokens 256 \
  --prompt-input-tokens 2048 \
  --dataset-split pg19_2K
```

The fixed observability parameters are:

```json
{
  "draft_mode": "branching",
  "draft_length": 8,
  "draft_tree_nodes": 32,
  "draft_tree_max_depth": 6,
  "retrieval_chunk_size": 64,
  "retrieve_top_k": 16,
  "retrieve_every_n_steps": 16
}
```

Run the experiment in another terminal:

```bash
python3 scripts/experiments/run_observability_experiment.py \
  outputs/observability/<run_id>/configs/observability.json
```

Common one-command overrides:

```bash
DRAFT_MODE=linear bash quick.sh
DRAFT_TREE_NODES=32 DRAFT_TREE_MAX_DEPTH=6 bash quick.sh
MAX_TOKENS=256 PROMPT_COUNT=1 bash quick.sh
DATASET_SPLIT=pg19_8K PROMPT_INPUT_TOKENS=8192 bash quick.sh
METHODS="direct observability_fixed_R16_k16" bash quick.sh
VERIFY_SERVER_URL=http://localhost:6008 bash quick.sh
```

Method names:

```text
direct       target model direct generation baseline
observability_fixed_R16_k16   SpecExtend with fixed R=16 guidance updates and observability logging
```

Do not set `retrieve_every_n_steps` to `0`. That disables target-attention retrieval updates.

Results are written to:

```text
outputs/observability/<run_id>/raw/quick_test_results.json
outputs/observability/<run_id>/raw/round_observations.jsonl
outputs/observability/<run_id>/analysis/rounds.csv
outputs/observability/<run_id>/analysis/summary.csv
outputs/observability/<run_id>/report.md
```
