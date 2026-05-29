# Server Runbook

Requires Linux + CUDA + Python 3.12 (miniconda3 base). Models must be present
under `models/`.

## 1. Install Dependencies

```bash
source /etc/network_turbo   # optional: academic network acceleration
pip install -r requirements.txt
```

## 2. Start The Verify Server

```bash
# defaults: Qwen3-14B-AWQ @ GPU 0, port 6008, eager attention
bash start_verify.sh

# health check
curl http://localhost:6008/health
```

Override any default via env:

```bash
VERIFY_GPU_ID=0 VERIFY_MODEL_PATH=$PWD/models/Qwen3-14B-AWQ \
VERIFY_ATTN_IMPLEMENTATION=eager PORT=6008 bash start_verify.sh
```

Expected health response fields: `backend: custom_qwen3`,
`specextend_tree_verify: true`, `attention_scores: true`.

## 3. Run The Experiment

```bash
# defaults: Qwen3-1.7B draft @ GPU 1, pg19, 2048 input, 256 output
bash run_quick.sh
```

Common overrides:

```bash
MAX_TOKENS=512 NODES=32 PROMPT_TYPES=pg19 bash run_quick.sh
```

Or launch server + experiment together (server killed on exit):

```bash
bash run_2x3090.sh
```

## 4. Inspect Results

```bash
ls -lh scripts/experiments/outputs_quick/
python3 scripts/experiments/analyze_quick_run.py
```

Key fields: `speculative.acceptance_length`, `timing.local_draft_ms`,
`timing.server_model_ms`, `raw.round_details`.

## 5. Common Failures

### Port already in use

```bash
ss -tlnp | grep 6008
kill <PID>
```

Or use a different port: `PORT=6009 bash start_verify.sh` and set
`VERIFY_SERVER_URL=http://localhost:6009` for `run_quick.sh`.

### CUDA Out Of Memory

```bash
MAX_TOKENS=32 NODES=8 MAX_DEPTH=3 bash run_quick.sh
```

### Health Check Not Ready

Check the `start_verify.sh` terminal for traceback. Common cause: missing
`autoawq` for AWQ models (`pip install autoawq`).

### No Retrieval Updates

Set `RETRIEVE_EVERY_N_STEPS=1` for short runs with few rounds.

## 6. Minimal Validation Before Long Runs

```bash
python3 -m unittest discover -s tests
curl http://localhost:6008/health
MAX_TOKENS=32 NODES=8 MAX_DEPTH=3 bash run_quick.sh
```
