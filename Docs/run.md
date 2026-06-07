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
  "methods": ["direct", "specextend_gpu", "specextend_kvload_cpu"],
  "max_tokens": 512,
  "prompt_count": 3,
  "dataset_split": "pg19_4K",
  "prompt_input_tokens": 2048,
  "draft_length": 8,
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
DRAFT_LENGTH=16 bash quick.sh
MAX_TOKENS=256 PROMPT_COUNT=1 bash quick.sh
DATASET_SPLIT=pg19_8K PROMPT_INPUT_TOKENS=8192 bash quick.sh
METHODS="direct specextend_gpu" bash quick.sh
VERIFY_SERVER_URL=http://localhost:6008 bash quick.sh
```

Method names:

```text
direct                 target model direct generation baseline
specextend_gpu         linear SpecExtend with sparse draft KV, selected KV on GPU
specextend_kvload_cpu  same sparse KV selection, plus CPU KV loading cost on retrieval updates
```

Do not set `retrieve_every_n_steps` to `0` for sparse-KV experiments. That disables target-attention retrieval updates.

Results are written to:

```text
scripts/experiments/outputs_quick/quick_test_results_*.json
scripts/experiments/outputs_quick/quick_test_summary.json
scripts/experiments/outputs_quick/quick_test_rounds.jsonl
```
