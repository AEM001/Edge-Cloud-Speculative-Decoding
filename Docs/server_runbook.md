# Server Runbook

Use this on the CUDA server, not on the Mac. The Mac workspace can edit and run
unit tests, but the real Qwen3 backend needs Linux, CUDA, PyTorch, and the model
weights.

## 1. Sync The Branch

```bash
cd /root/code/research/Infra/draft
git status --short
git pull
```

Use the server path that matches where you clone the repo. The rest of this doc
assumes the working directory is the `draft/` repo.

## 2. Create The Python Environment

The project requires Python 3.12.

```bash
uv python install 3.12
uv sync
```

If you do not use `uv`, create a Python 3.12 venv and install:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

After dependency changes, refresh the lockfile on the server:

```bash
uv lock
```

## 3. Download Models

Default one-model setup, using Qwen3-8B for both edge draft and cloud verify:

```bash
.venv/bin/python models/download.py --preset edge-cloud
```

Smaller smoke-test setup:

```bash
.venv/bin/python models/download.py --preset smoke
```

Separate draft and verify setup:

```bash
.venv/bin/python models/download.py --preset separate
```

That downloads:

- `models/Qwen3-1.7B`
- `models/Qwen3-8B`

You can also download explicit aliases:

```bash
.venv/bin/python models/download.py --model qwen3-1.7b --model qwen3-8b
```

If Hugging Face auth is required, either export `HF_TOKEN`/login with the HF CLI
or put the token in `models/hf.txt`.

## 4. Start The Verify Server

On GPU 0:

```bash
VERIFY_GPU_ID=0 \
VERIFY_MODEL_PATH=$PWD/models/Qwen3-8B \
VERIFY_DTYPE=fp16 \
VERIFY_MAX_LEN=32768 \
bash start_verify.sh --port 6007
```

Health check from another shell:

```bash
curl http://localhost:6007/health
```

Expected runtime fields:

- `backend`: `custom_qwen3`
- `specextend_tree_verify`: `true`
- `attention_scores`: `true`

## 5. Run A Small SpecExtend Smoke Test

On GPU 1, start small first:

```bash
DRAFT_GPU_ID=1 \
DRAFT_MODEL_PATH=$PWD/models/Qwen3-8B \
DRAFT_DTYPE=fp16 \
VERIFY_SERVER_URL=http://localhost:6007 \
MAX_TOKENS=32 \
PROMPT_COUNT=1 \
PROMPT_TYPES=prompts_2048 \
NODES=8 \
MAX_DEPTH=3 \
RETRIEVE_EVERY_N_STEPS=2 \
./quick.sh
```

For separate draft/verify models:

```bash
DRAFT_GPU_ID=1 \
DRAFT_MODEL_PATH=$PWD/models/Qwen3-1.7B \
VERIFY_SERVER_URL=http://localhost:6007 \
MAX_TOKENS=32 \
NODES=8 \
MAX_DEPTH=3 \
./quick.sh
```

If the smoke run works, increase:

- `MAX_TOKENS=512`
- `NODES=32`
- `MAX_DEPTH=8`
- longer prompt types

## 6. Inspect Results

Quick results are timestamped:

```bash
ls -lh scripts/experiments/outputs_quick/
```

Summarize the newest result:

```bash
.venv/bin/python scripts/experiments/analyze_quick_run.py
```

Important fields:

- `method_family`: should include `direct` and `specextend`
- `speculative.rounds`
- `speculative.acceptance_length`
- `speculative.accepted_draft_tokens`
- `raw.selected_chunk_ids`
- `raw.round_details`

## 7. Common Failures

### CUDA Out Of Memory

Use smaller settings first:

```bash
MAX_TOKENS=32 NODES=8 MAX_DEPTH=3 ./quick.sh
```

Or use smaller models:

```bash
.venv/bin/python models/download.py --preset separate
VERIFY_MODEL_PATH=$PWD/models/Qwen3-8B
DRAFT_MODEL_PATH=$PWD/models/Qwen3-1.7B
```

### Python Version Error

The repo requires Python 3.12. Recreate `.venv` with Python 3.12 and rerun
`uv sync`.

### Health Check Says Not Ready

The server is still loading the model or failed during startup. Check the
`start_verify.sh` terminal for the model path, CUDA device, dtype, and traceback.

### No Retrieval Updates

`SpecExtendEdgeClient` requests target attention scores after
`RETRIEVE_EVERY_N_STEPS`. For a tiny run with too few rounds, set:

```bash
RETRIEVE_EVERY_N_STEPS=1
```

## 8. Minimal Validation Before Long Runs

Run these first:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall scripts tests
curl http://localhost:6007/health
MAX_TOKENS=32 NODES=8 MAX_DEPTH=3 ./quick.sh
```

Only start long-context experiments after those pass.
