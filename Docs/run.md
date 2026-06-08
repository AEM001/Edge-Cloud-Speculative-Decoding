Start the verify server in one terminal:

```bash
bash start_verify.sh
```

If you need a different port or GPU:

```bash
PORT=6008 VERIFY_GPU_ID=0 bash start_verify.sh
```

Configure the benchmark by editing:

```text
quick_benchmark_config.json
```

The main parameters to change are:

```json
{
  "methods": ["direct", "specextend"],
  "max_tokens": 512,
  "prompt_count": 3,
  "dataset_split": "pg19_4K",
  "prompt_input_tokens": 2048,
  "draft_mode": "branching",
  "draft_length": 8,
  "draft_tree_nodes": 32,
  "draft_tree_max_depth": 8,
  "retrieval_chunk_size": 64,
  "retrieve_top_k": 16,
  "retrieve_every_n_steps": 16
}
```

Run the benchmark in another terminal:

```bash
bash quick.sh
```

Common one-command overrides:

```bash
DRAFT_MODE=linear bash quick.sh
DRAFT_TREE_NODES=32 DRAFT_TREE_MAX_DEPTH=8 bash quick.sh
MAX_TOKENS=256 PROMPT_COUNT=1 bash quick.sh
DATASET_SPLIT=pg19_8K PROMPT_INPUT_TOKENS=8192 bash quick.sh
METHODS="direct specextend" bash quick.sh
VERIFY_SERVER_URL=http://localhost:6008 bash quick.sh
```

Method names:

```text
direct       target model direct generation baseline
specextend   SpecExtend with sparse draft KV and target-attention retrieval
```

Do not set `retrieve_every_n_steps` to `0`. That disables target-attention retrieval updates.

Results are written to:

```text
outputs/quick_test_results_<timestamp>.json
outputs/quick_test_results_<timestamp>_summary.txt
```
