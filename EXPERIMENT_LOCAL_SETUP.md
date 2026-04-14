# Local Experiment Setup: k=2,4,6,8 with Network Simulation

## Overview
This setup runs both the **draft model (1.5B)** and **verification model (32B)** on the same machine (AutoDL), using network simulation to simulate different network conditions (good, medium, bad, bursty).

## Model Locations
- **Draft Model (1.5B)**: `/root/autodl-tmp/ubuntu-verify/models/` (2.9GB)
- **Verify Model (32B)**: `/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-32B-Instruct-GPTQ-Int4/` (19GB)

## Files Created

### 1. `experiment_local_k2k8.py`
Main experiment script with:
- K values: 2, 4, 6, 8
- Network regimes: good, medium, bad, bursty
- 10 easy + 10 hard prompts
- Tests: Direct + K=2 + K=4 + K=6 + K=8 = 5 methods per prompt
- Total: 20 prompts × 5 methods × 4 regimes = 400 tests

### 2. `run_local_experiment.sh`
Starts the verification server and runs the experiment:
```bash
./run_local_experiment.sh
```

### 3. `experiments/run_local_experiment_tmux.sh`
Runs experiment in tmux (continues even if SSH disconnects):
```bash
cd experiments && bash run_local_experiment_tmux.sh
```

### 4. `config_local.py`
Local configuration using `localhost` instead of remote server.

### 5. `check_model_status.py`
Check if the 1.5B model is ready:
```bash
python check_model_status.py          # Check once
python check_model_status.py --wait   # Wait until ready
```

## How to Run

### Option 1: Direct Run (Foreground)
```bash
cd /root/autodl-tmp/draft
bash run_local_experiment.sh
```

### Option 2: Background Run with Tmux (Recommended)
```bash
cd /root/autodl-tmp/draft/experiments
bash run_local_experiment_tmux.sh

# View progress:
tmux attach-session -t local-experiment-k2k8

# Detach (Ctrl+B, then D)
```

### Option 3: Manual Steps
```bash
# 1. Start verification server (in one terminal)
cd /root/autodl-tmp/ubuntu-verify
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/api_server.py --host 0.0.0.0 --port 6006

# 2. Run experiment (in another terminal)
cd /root/autodl-tmp/draft
export AINFRA_DRAFT_MODEL_PATH=/root/autodl-tmp/ubuntu-verify/models
export AINFRA_TEMPERATURE=0.0
python experiments/experiment_local_k2k8.py
```

## Network Simulation Parameters

| Regime | RTT (ms) | Bandwidth (Mbps) | Description |
|--------|----------|------------------|-------------|
| good   | 10       | 100              | Excellent network |
| medium | 40       | 20               | Average network |
| bad    | 100      | 5                | Poor network |
| bursty | 20-120   | 5-50             | Alternates every 20s |

## Output Files

Results are saved in `experiments/outputs_local_k2k8/`:
- `comprehensive_results.json` - All results and summary
- `results_{regime}.json` - Results per network regime
- `experiment_log.txt` - Detailed log

## Monitoring Progress

```bash
# View live log
tail -f /root/autodl-tmp/draft/experiments/outputs_local_k2k8/experiment_log.txt

# View tmux session
tmux attach-session -t local-experiment-k2k8

# Check if running
pgrep -f experiment_local_k2k8.py
```

## Expected Runtime
- Direct: ~20-30 minutes total
- Estimated: 2-3 hours for full experiment (400 tests)

## Troubleshooting

### Model Not Found
Check if models exist:
```bash
ls -lh /root/autodl-tmp/ubuntu-verify/models/model.safetensors  # Should be ~2.9GB
ls -lh /root/autodl-tmp/ubuntu-verify/models/Qwen2.5-32B-Instruct-GPTQ-Int4/
```

### Out of Memory
Reduce GPU memory utilization in `config_local.py`:
```python
GPU_MEMORY_UTILIZATION = 0.6  # Default is 0.7
```

### Port Already in Use
Change verify port in scripts or kill existing process:
```bash
lsof -ti:6006 | xargs kill -9 2>/dev/null
```
