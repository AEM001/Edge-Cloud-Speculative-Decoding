# Edge-Cloud Speculative Decoding Experiment Setup

Complete setup guide for running the static K baseline experiment across Mac (edge) and Ubuntu (cloud).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mac Edge Client                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Draft Model: Qwen2.5-0.5B-Instruct-4bit (MLX)          │   │
│  │  - Generates K draft tokens                              │   │
│  │  - Sends to cloud for verification                       │   │
│  │  - Receives acceptance results                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP over FRP
                              │ http://49.234.57.210:8005
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Ubuntu Cloud Server                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Target Model: Qwen2.5-3B-Instruct-4bit (vLLM)          │   │
│  │  - Verifies draft tokens                                 │   │
│  │  - Returns accepted length + correction                  │   │
│  │  - Runs on RTX 3060 12GB                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Mac (Edge Side)
- MacBook Air M1 with 8GB RAM
- Python 3.10+
- MLX installed
- Internet connection to reach cloud server

### Ubuntu (Cloud Side)
- RTX 3060 12GB GPU
- Python 3.10+
- CUDA-compatible PyTorch
- vLLM installed
- FRP client configured

## Setup Instructions

### DONE! Part 1: Ubuntu Cloud Server Setup

#### 1.1 Install Dependencies

```bash
cd /home/albert/code/AInfra/ubuntu-verify

# Install dependencies with uv
uv sync

# Activate environment
source .venv/bin/activate
```

#### 1.2 Download Target Model

The server will auto-download the model on first run, or pre-download:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='~/.cache/huggingface/hub/Qwen2.5-3B-Instruct')
"
```

#### 1.3 Configure FRP for External Access

Add cloud verification server to FRP configuration:

```bash
# Edit FRP config
sudo nano /etc/frp/frpc.toml

# Add this proxy:
[[proxies]]
name = "cloud-verify"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8000
remotePort = 8005

# Restart FRP
sudo systemctl restart frpc

# Verify
sudo systemctl status frpc
```

See `ubuntu-verify/FRP_SETUP.md` for detailed FRP setup.

#### 1.4 Start Cloud Server

```bash
cd /home/albert/code/AInfra/ubuntu-verify

# Start server (will load model and listen on port 8000)
./start_server.sh

# Or manually:
source .venv/bin/activate
cd src
python api_server.py --host 0.0.0.0 --port 8000
```

The server will be accessible at:
- Locally: `http://localhost:8000`
- Externally (via FRP): `http://49.234.57.210:8005`

#### 1.5 Verify Server is Running

```bash
# From Ubuntu (local)
curl http://localhost:8000/health

# From Mac or anywhere (external)
curl http://49.234.57.210:8005/health

# Expected response:
# {"status":"healthy","model_info":{"model_name":"Qwen/Qwen2.5-3B-Instruct",...}}
```

### DONE! Part 2: Mac Edge Client Setup

#### 2.1 Install Dependencies

```bash
cd /path/to/mac-draft

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2.2 Download Draft Model

```bash
# The model will auto-download on first run
# Or pre-download:
python -c "
from mlx_lm import load
load('mlx-community/Qwen2.5-0.5B-Instruct-4bit')
"
```

#### 2.4 Test Connection to Cloud Server

```bash
python run_experiment.py \
  --server-url http://49.234.57.210:8005 \
  --test-connection

# Expected output:
# ✓ Connection successful!
```

### Part 3: Run Experiments

#### 3.1 Run Full Experiment Matrix

```bash
cd /path/to/mac-draft

# Run all conditions (3 K × 2 workloads × 4 networks = 24 conditions)
python run_experiment.py \
  --server-url http://49.234.57.210:8005 \
  --output-dir results/full_experiment \
  --max-tokens 128

# This will take several hours depending on prompt count
```

#### 3.2 Run Single Condition (for testing)

```bash
# Test with one condition first
python run_experiment.py \
  --server-url http://49.234.57.210:8005 \
  --output-dir results/test \
  --max-tokens 128
```

#### 3.3 Disable Network Simulation (use real network only)

```bash
python run_experiment.py \
  --server-url http://49.234.57.210:8005 \
  --no-network-simulation \
  --output-dir results/real_network
```

## Experiment Matrix

The experiment runs:

| Dimension | Values |
|-----------|--------|
| **K (draft tokens)** | 2, 4, 6 |
| **Workload** | Easy (30 prompts), Hard (30 prompts) |
| **Network** | Good (10ms RTT, 100Mbps), Medium (40ms, 20Mbps), Bad (100ms, 5Mbps), Bursty (alternating) |

**Total conditions**: 3 × 2 × 4 = 24

**Total requests**: 24 × 30 = 720 (if using 30 prompts per workload)

## Results

Results are saved to the output directory with the following structure:

```
results/
├── experiment_metadata.json       # Experiment configuration
├── all_results.json               # Aggregate results for all conditions
└── k{K}_{workload}_{network}/     # Per-condition results
    ├── request_metrics.json       # Per-request metrics
    └── aggregate_metrics.json     # Condition-level aggregates
```

### Key Metrics

Per-request:
- Total latency
- Tokens per second
- Acceptance ratio
- Wasted drafted tokens
- Network RTT

Per-condition:
- Mean/p50/p95 latency
- Mean acceptance ratio
- Variance in acceptance ratio

## Monitoring

### Monitor Cloud Server

```bash
# On Ubuntu, watch server logs
cd /home/albert/code/AInfra/ubuntu-verify/src
python api_server.py --host 0.0.0.0 --port 8000

# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi
```

### Monitor Edge Client

```bash
# On Mac, watch experiment progress
tail -f results/full_experiment/experiment.log
```

## Troubleshooting

### Connection Issues

```bash
# Test FRP tunnel
curl http://49.234.57.210:8005/health

# Check FRP status on Ubuntu
sudo systemctl status frpc
sudo journalctl -u frpc -f
```

### Model Loading Issues

**Ubuntu (vLLM)**:
- Check GPU memory: `nvidia-smi`
- Reduce `gpu_memory_utilization` in `src/config/settings.py`
- Use smaller model if 3B is too large

**Mac (MLX)**:
- Check available RAM
- Use 4-bit quantized model (already configured)

### Network Simulation Not Working

```bash
# Disable simulation and use real network
python run_experiment.py \
  --server-url http://49.234.57.210:8005 \
  --no-network-simulation
```

### Slow Generation

- Reduce `max_tokens` from 128 to 64
- Reduce number of prompts per workload
- Check network latency: `ping 49.234.57.210`

## File Locations

### Ubuntu
- Server code: `/home/albert/code/AInfra/ubuntu-verify/src/`
- API server: `/home/albert/code/AInfra/ubuntu-verify/src/api_server.py`
- Cloud verifier: `/home/albert/code/AInfra/ubuntu-verify/src/cloud_server.py`
- Prompts: `/home/albert/code/AInfra/ubuntu-verify/prompts_{easy,hard}.json`
- FRP config: `/etc/frp/frpc.toml`

### Mac
- Client code: `/path/to/mac-draft/client/`
- Edge client: `/path/to/mac-draft/client/edge_client.py`
- HTTP client: `/path/to/mac-draft/client/http_cloud_client.py`
- Network wrapper: `/path/to/mac-draft/client/network_wrapper.py`
- Experiment runner: `/path/to/mac-draft/run_experiment.py`

## Next Steps

1. **Run pilot test**: Test with 1-2 prompts per condition
2. **Verify metrics**: Check that acceptance ratios are reasonable
3. **Run full experiment**: Execute complete 24-condition matrix
4. **Analyze results**: Use provided metrics to generate tables and plots
5. **Iterate**: Adjust parameters based on initial results

## References

- Experiment spec: `/home/albert/code/AInfra/ubuntu-verify/experiment_spec.md`
- FRP setup: `/home/albert/code/AInfra/ubuntu-verify/frp_info.md`
- Mac README: `/path/to/mac-draft/README.md`
- Ubuntu README: `/home/albert/code/AInfra/ubuntu-verify/README.md`
