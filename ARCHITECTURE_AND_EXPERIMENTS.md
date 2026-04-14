# Speculative Decoding System: Architecture & Setup Guide

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Components](#3-components)
4. [Data Flow & Protocol](#4-data-flow--protocol)
5. [Experiment Setup](#5-experiment-setup)
6. [Benchmarks & Metrics](#6-benchmarks--metrics)
7. [Setup & Deployment](#7-setup--deployment)
8. [Troubleshooting](#8-troubleshooting)
- [Appendix A: File Index](#appendix-a-file-index)

---

## 1. System Overview

### 1.1 What is This System?

This is an **edge-cloud speculative decoding research platform** designed to:
- Study the performance characteristics of speculative decoding in real-world edge-cloud scenarios
- Compare different speculative strategies (static K values)
- Analyze the impact of network conditions, prompt types, and model configurations

### 1.2 Core Concept: Speculative Decoding

```
Traditional:  Target Model generates tokens one-by-one (slow, high quality)
Speculative:  Draft Model generates K candidate tokens (fast, lower quality)
                ↓
                Target Model verifies all K tokens in parallel
                ↓
                Accept valid tokens, reject invalid ones, continue
```

### 1.3 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPECULATIVE DECODING SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐          ┌─────────────────────────┐          │
│  │    DRAFT SIDE           │          │    CLOUD SIDE           │          │
│  │    (Edge/Client)        │          │    (Server/Verify)      │          │
│  │                         │          │                         │          │
│  │  • Draft Model          │◄────────►│  • Target Model         │          │
│  │    (Qwen2.5-3B)         │   HTTP   │    (Qwen2.5-32B-GPTQ)   │          │
│  │  • Token Generation     │          │  • Token Verification   │          │
│  │  • Confidence Stats     │          │  • Parallel Verification│          │
│  │  • Policy Selection     │          │  • Correction Token     │          │
│  │                         │          │                         │          │
│  │  Location: Local CUDA   │          │  Location: RTX 5090     │          │
│  │  Port: 8001 (client)    │          │  Port: 6006 (server)    │          │
│  └─────────────────────────┘          └─────────────────────────┘          │
│                                                                             │
│  Repository: draft/                      Repository: remote-verify/          │
│  Role: Generate drafts                   Role: Verify & correct             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Repository Structure

```
AInfra/
├── draft/                          # Edge/client side (draft generation)
│   ├── client/
│   │   ├── edge_client.py         # Main edge client orchestration
│   │   ├── draft_generator.py     # Draft model inference
│   │   ├── http_cloud_client.py   # HTTP client to cloud
│   │   └── model_manager.py       # Model loading & management
│   ├── experiments/               # Experiment scripts and results
│   │   ├── comprehensive_benchmark.py    # Main benchmark script
│   │   ├── run_experiment.sh      # Tmux wrapper for experiments
│   │   └── EXPERIMENT_STATUS.md   # Current experiment status
│   ├── start_ssh_tunnel.sh        # SSH tunnel with auto-reconnect
│   ├── protocol.py                # Data structures for communication
│   ├── config.py                  # Configuration settings
│   └── requirements.txt             # Python dependencies
│
└── remote-verify/                  # Cloud/server side (verification)
    ├── src/
    │   ├── api_server.py          # FastAPI HTTP server
    │   ├── cloud_server.py        # Verification logic with vLLM
    │   └── config/                # Server configuration
    ├── start.sh                   # Server startup script
    ├── start_background.sh        # Background server starter
    └── stop_background.sh         # Background server stopper
```

---

## 2. Architecture

### 2.1 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Separation of concerns** | Draft and verify are separate repos with clear interfaces |
| **Protocol-based communication** | HTTP API with well-defined request/response structures |
| **Policy-driven K selection** | Pluggable policy system for experimenting with different strategies |
| **Comprehensive metrics** | Detailed timing, token, and network metrics for analysis |
| **Reproducible benchmarks** | Fixed prompt sets with controlled length and difficulty |

### 2.2 Communication Flow

```
┌──────────┐     ┌──────────────┐     ┌──────────┐
│  User    │────►│ Edge Client  │────►│  Cloud   │
│  Prompt  │     │              │     │  Server  │
└──────────┘     └──────────────┘     └──────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  1. Generate K    │
              │     draft tokens  │
              │     (draft model) │
              └───────────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  2. Send to cloud │
              │     via HTTP POST │
              │     /verify       │
              └───────────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  3. Verify drafts │
              │     (target model)│
              │     in parallel   │
              └───────────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  4. Return results│
              │     - accepted_len│
              │     - correction  │
              └───────────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  5. Update prefix │
              │     & continue    │
              │     if more tokens│
              └───────────────────┘
```

### 2.3 Round-Based Processing

Each generation request consists of multiple **rounds**:

```
Round 1:  prefix="Hello" → draft K=4 tokens → verify → accept 2, reject 2
Round 2:  prefix="Hello world" → draft K=4 tokens → verify → accept 4
Round 3:  prefix="Hello world how are" → draft K=4 tokens → verify → accept 1, EOS
Done.
```

**Metrics per round**:
- Draft time (edge)
- Network RTT
- Verify time (cloud)
- Acceptance count

---

## 3. Components

### 3.1 Draft Side (draft/)

#### 3.1.1 EdgeClient (`client/edge_client.py`)

**Responsibilities**:
- Orchestrate the entire speculative decoding loop
- Maintain verified prefix state across rounds
- Interface with draft generator and cloud client
- Collect and aggregate metrics per request

**Key methods**:
```python
generate(prompt, policy, policy_name) -> RequestMetrics
# Main entry point: generates text using speculative decoding

_draft_chunk(prefix_ids, k) -> DraftResult
# Generate K draft tokens from current prefix

_verify_with_cloud(draft_ids, prefix_ids) -> CloudResponse
# Send draft to cloud for verification
```

**RequestMetrics dataclass**:
```python
@dataclass
class RequestMetrics:
    # Timing
    total_latency_ms: float          # End-to-end time
    total_edge_draft_time_ms: float  # Cumulative draft time
    total_server_verify_time_ms: float  # Cumulative verify time
    total_network_time_ms: float     # Cumulative network time
    
    # Tokens
    generated_tokens: int            # Final output tokens
    total_rounds: int                # Number of verification rounds
    mean_K_chosen: float             # Average K per round
    
    # Acceptance
    acceptance_ratio: float            # accepted / drafted
    total_drafted_tokens: int
    total_accepted_drafted_tokens: int
    wasted_drafted_tokens: int
```

#### 3.1.2 DraftGenerator (`draft_generator.py`)

**Responsibilities**:
- Generate K draft tokens autoregressively
- Collect confidence statistics for each token
- Handle temperature and sampling parameters

**Confidence statistics per token**:
- `probability`: Token probability
- `max_prob`: Max probability in distribution
- `entropy`: Distribution entropy
- `top_margin`: Top-1 minus top-2 probability

#### 3.1.3 ModelManager (`model_manager.py`)

**Responsibilities**:
- Load draft model from local path or HuggingFace
- Handle vLLM model initialization
- Provide model and tokenizer to other components

**Current draft model**: `Qwen/Qwen2.5-3B-Instruct` (local CUDA)

#### 3.1.4 HTTPCloudClient (`client/http_cloud_client.py`)

**Responsibilities**:
- Send EdgeRequest to cloud server
- Handle network timeouts and retries
- Measure RTT

### 3.2 Cloud Side (remote-verify/)

#### 3.2.1 CloudVerifier (`src/cloud_server.py`)

**Responsibilities**:
- Initialize and manage vLLM model
- Verify draft tokens in parallel
- Return acceptance count and correction token

**Verification algorithm**:
```python
def verify_draft(self, prefix_ids, draft_ids):
    # 1. Run target model on prefix + draft_tokens
    # 2. Get logits for each position
    # 3. Compare target's predicted token vs draft token
    # 4. Count consecutive matches from the start
    # 5. Return accepted_len and correction token
```

**Current target model**: `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` on RTX 5090

#### 3.2.2 API Server (`src/api_server.py`)

**FastAPI endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health with model info |
| `/verify` | POST | Verify draft tokens (main endpoint) |
| `/generate` | POST | Direct generation (for comparison) |

**Request/Response models**:
```python
class EdgeRequest(BaseModel):
    request_id: str
    round_id: int
    prefix_ids: List[int]
    draft_ids: List[int]
    draft_logprobs: List[float]
    edge_draft_time_ms: float

class CloudResponse(BaseModel):
    accepted_len: int
    accepted_token_ids: List[int]
    correction_token_id: Optional[int]
    server_verify_time_ms: float
```

### 3.3 Protocol Layer

**Data structures** (`protocol.py`):

```python
# Edge → Cloud
EdgeRequest:
    - request_id: str              # UUID for tracking
    - round_id: int                # Round number in request
    - prefix_ids: List[int]       # Verified context tokens
    - draft_ids: List[int]        # Draft tokens to verify
    - draft_logprobs: List[float] # Draft confidence scores
    - edge_draft_time_ms: float   # Time spent drafting

# Cloud → Edge  
CloudResponse:
    - accepted_len: int            # Number of accepted tokens
    - accepted_token_ids: List[int]
    - correction_token_id: Optional[int]  # If first token rejected
    - server_verify_time_ms: float
    - rtt_ms: Optional[float]
```

---

## 4. Data Flow & Protocol

### 4.1 Request Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    REQUEST LIFECYCLE                             │
└─────────────────────────────────────────────────────────────────┘

1. INITIATION
   User calls: edge_client.generate(prompt="Hello", policy=static_k_2)
   
2. TOKENIZATION
   prompt → token_ids (e.g., [101, 7592, ...])
   
3. ROUND LOOP (until max_tokens or EOS)
   
   ┌──────────────────────────────────────────────────────┐
   │ ROUND N                                              │
   │                                                      │
   │ 3a. DRAFT (edge)                                    │
   │     prefix_ids = current_verified_prefix            │
   │     K = policy(round_id, draft_tokens)  # e.g., K=2 │
   │     draft_ids = draft_model.generate(prefix_ids, K) │
   │     Record: draft_time_ms                           │
   │                                                      │
   │ 3b. VERIFY (cloud)                                  │
   │     POST /verify with EdgeRequest                   │
   │     {prefix_ids, draft_ids, draft_time_ms}          │
   │                                                      │
   │ 3c. PROCESS RESPONSE                                │
   │     accepted = response.accepted_len                │
   │     prefix_ids += accepted tokens                   │
   │     if accepted < K:                                │
   │         prefix_ids += [correction_token]            │
   │                                                      │
   │ 3d. RECORD METRICS                                  │
   │     round_metrics: {draft_time, verify_time, rtt,   │
   │                     accepted, K}                    │
   └──────────────────────────────────────────────────────┘
   
4. COMPLETION
   Return: generated_text, RequestMetrics
```

### 4.2 Network Communication

**Current setup**:

| Method | URL | Notes |
|--------|-----|-------|
| SSH Tunnel | `http://localhost:6006` | Recommended - secure and stable |
| SSH Port | `20514` | Updated from 12272 |

**SSH Tunnel setup** (recommended):

```bash
# Manual tunnel
ssh -p 20514 -L 6006:localhost:6006 root@connect.westd.seetacloud.com -N

# Or use the auto-reconnect script
cd /home/albert/code/AInfra/draft
bash start_ssh_tunnel.sh

# This creates a tmux session 'ssh-tunnel' that auto-reconnects on disconnect
```

**SSH Config** (optional, for convenience):
```bash
# ~/.ssh/config
Host 5090-new
    HostName connect.westd.seetacloud.com
    Port 20514
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

---

## 5. Experiment Setup

### 5.1 Experiment Types

| Experiment | Description | Script | Status |
|------------|-------------|--------|--------|
| **Comprehensive Benchmark** | Easy vs Hard prompts with K=2,4,6 and Direct | `experiments/comprehensive_benchmark.py` | Active |
| Static K Baseline | Compare K=2,4,6 systematically | Future work | Planned |
| Network Impact | Test under different RTT conditions | Future work | Planned |

### 5.2 Policy System

**Current policies**:

```python
def static_k_2(round_id, draft_tokens) -> int:
    """Conservative: draft 2 tokens per round"""
    return 2

def static_k_4(round_id, draft_tokens) -> int:
    """Balanced: draft 4 tokens per round"""
    return 4

def static_k_6(round_id, draft_tokens) -> int:
    """Aggressive: draft 6 tokens per round"""
    return 6
```

**Policy signature**:
```python
Policy = Callable[[round_id: int, draft_tokens: List[TokenInfo]], int]
# Returns: K (number of tokens to draft next round)
```

---

## 6. Benchmarks & Metrics

### 6.1 Benchmark Prompts

**Source datasets**:
- **Easy**: `ConversationChronicles` (785K conversation prompts)
- **Hard**: `Bespoke-Stratos-17k` (16.6K reasoning/math/code prompts)

**Selection criteria**:
- Target length: ~512 tokens (prompt)
- Target completion: ~128 tokens
- Diverse topics and difficulty levels

**Current benchmark sets**:
- `prompts_easy.json`: 30 conversation prompts
- `prompts_hard.json`: 30 reasoning prompts
- Selected prompts have similar lengths for fair comparison

### 6.2 Key Metrics

#### 6.2.1 Performance Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| **Tokens per Second** | tok/s | Throughput: generated_tokens / total_time |
| **Time per Output Token** | ms | TPOT: total_time / generated_tokens |
| **Total Latency** | ms | End-to-end generation time |

#### 6.2.2 Speculative Decoding Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| **Acceptance Rate** | % | accepted_tokens / drafted_tokens |
| **Rounds** | count | Number of verification rounds |
| **Mean K** | tokens | Average draft length per round |
| **Wasted Tokens** | tokens | Drafted but rejected tokens |

#### 6.2.3 Timing Breakdown

| Metric | Unit | Description |
|--------|------|-------------|
| **Draft Time** | ms/round | Time to generate K draft tokens (edge) |
| **Verify Time** | ms/round | Time to verify K tokens (cloud) |
| **Network RTT** | ms | Round-trip time for HTTP request |
| **Network Bytes** | bytes | Uplink + downlink traffic |

### 6.3 Metrics Collection

**Per-request metrics** (collected in `RequestMetrics`):
```python
metrics = {
    # Timing
    'total_latency_ms': 15000,           # 15s total
    'total_edge_draft_time_ms': 8000,  # 8s drafting
    'total_server_verify_time_ms': 3000,  # 3s verifying
    'total_network_time_ms': 4000,     # 4s network
    
    # Tokens
    'generated_tokens': 128,
    'total_drafted_tokens': 200,
    'total_accepted_drafted_tokens': 80,
    'wasted_drafted_tokens': 120,
    
    # Ratios
    'acceptance_ratio': 0.40,           # 40%
    'tokens_per_second': 8.5,
    'total_rounds': 45,
    'mean_K_chosen': 4.4,
}
```

---

## 7. Setup & Deployment

### 7.1 Prerequisites

**Hardware requirements**:
- **Draft side**: CUDA-capable GPU (tested on local CUDA with 8GB+ VRAM)
- **Cloud side**: RTX 5090 or equivalent with 24GB+ VRAM (for 32B model)
- **Network**: SSH access to cloud server

**Software**:
- Python 3.9+
- CUDA 12.1+ (for PyTorch)
- SSH client (for tunnel)
- tmux (for background processes)

### 7.2 Draft Side Setup (draft/)

```bash
# 1. Clone and navigate
cd /home/albert/code/AInfra/draft

# 2. Use existing virtual environment (already set up)
source /home/albert/learn/l-vllm/.venv/bin/activate

# 3. Dependencies already installed:
# - vllm
# - torch
# - transformers
# - accelerate
# - numpy
# - requests

# 4. Verify model path (config.py)
# MODEL_PATH = Path.home() / "models" / "Qwen--Qwen2.5-3B-Instruct"

# 5. Test connection to cloud (after setting up SSH tunnel)
python3 -c "
import requests
print(requests.get('http://localhost:6006/health').json())
"
```

### 7.3 Cloud Side Setup (remote-verify/)

```bash
# 1. SSH to cloud server
ssh -p 20514 root@connect.westd.seetacloud.com

# 2. Navigate to project
cd /root/autodl-tmp/ubuntu-verify

# 3. Pull latest code
git pull

# 4. Use background scripts (from latest pull)
./start_background.sh

# 5. Verify it's running
curl http://localhost:6006/health
```

### 7.4 SSH Tunnel Setup (Recommended)

On draft machine:

```bash
cd /home/albert/code/AInfra/draft

# Option 1: Use the auto-reconnect script (RECOMMENDED)
bash start_ssh_tunnel.sh

# This creates tmux session 'ssh-tunnel' with:
# - Auto-reconnect on disconnect
# - ServerAliveInterval 30
# - TCPKeepAlive enabled

# Option 2: Manual tunnel
ssh -p 20514 -L 6006:localhost:6006 root@connect.westd.seetacloud.com -N

# Verify tunnel
curl http://localhost:6006/health
```

### 7.5 Running Experiments

**Quick test** (single prompt):
```bash
cd /home/albert/code/AInfra/draft
source /home/albert/learn/l-vllm/.venv/bin/activate
python3 simple_test.py
```

**Comprehensive benchmark** (with tmux - won't stop on SSH disconnect):
```bash
cd /home/albert/code/AInfra/draft/experiments
bash run_experiment.sh

# This creates tmux session 'benchmark-experiment'
# The experiment continues even if you disconnect
```

**View experiment progress**:
```bash
# View live log
tail -f /home/albert/code/AInfra/draft/experiments/comprehensive_log.txt

# Enter tmux session
tmux attach-session -t benchmark-experiment
# Press Ctrl+B, then D to detach

# Check if still running
pgrep -f comprehensive_benchmark.py
```

### 7.6 Configuration Reference

**Draft config** (`draft/config.py`):
```python
# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_PATH = Path.home() / "models" / "Qwen--Qwen2.5-3B-Instruct"

# vLLM settings
GPU_MEMORY_UTILIZATION = 0.7
MAX_MODEL_LEN = 8192  # Reduced from 32768 for 3B model

# Server configuration
CLOUD_SERVER = "connect.westd.seetacloud.com"
CLOUD_SSH_PORT = 20514  # Updated from 12272
CLOUD_PORT = 6006

# Generation settings
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 128
```

**Cloud config** (environment variables in remote-verify/.env):
```bash
VERIFY_MODEL_NAME=Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4
VERIFY_GPU_MEMORY_UTILIZATION=0.9
VERIFY_MAX_MODEL_LEN=8192
VERIFY_QUANTIZATION=gptq
```

---

## 8. Troubleshooting

### 8.1 SSH Connection Issues

**Problem**: Cannot connect to SSH port 20514
```bash
# Check if SSH config is correct
cat ~/.ssh/config

# Should contain:
Host 5090-new
    HostName connect.westd.seetacloud.com
    Port 20514
    User root

# Test connection
ssh 5090-new
```

**Problem**: SSH tunnel disconnects frequently
```bash
# Use the auto-reconnect script
cd /home/albert/code/AInfra/draft
bash start_ssh_tunnel.sh

# Check tunnel status
tmux attach-session -t ssh-tunnel
```

### 8.2 Model Loading Issues

**Problem**: "vLLM not installed" error
```bash
# Activate correct virtual environment
source /home/albert/learn/l-vllm/.venv/bin/activate

# Verify vllm is available
python3 -c "import vllm; print(vllm.__version__)"
```

**Problem**: Out of memory when loading 3B model
```bash
# Solution already applied in config.py:
# MAX_MODEL_LEN = 8192  (reduced from 32768)
# GPU_MEMORY_UTILIZATION = 0.7  (increased from 0.6)

# If still OOM, further reduce MAX_MODEL_LEN to 4096
```

### 8.3 JSON Serialization Error

**Problem**: `Out of range float values are not JSON compliant: -inf`
```bash
# Fixed in protocol.py - logprobs with -inf are replaced with -1e10
# No action needed if using updated code
```

### 8.4 Low Acceptance Rate

**Problem**: Acceptance rate < 20%
```bash
# This was observed with 1.5B draft model paired with 32B target model
# Solution: Use 3B draft model (already configured in config.py)

# Expected acceptance rates with 3B:
# K=2: ~44%
# K=4: ~67%
```

### 8.5 Experiment Stops on SSH Disconnect

**Problem**: Experiment stops when closing terminal
```bash
# Use tmux wrapper script
cd /home/albert/code/AInfra/draft/experiments
bash run_experiment.sh

# Or manually run in tmux
tmux new-session -s experiment
source /home/albert/learn/l-vllm/.venv/bin/activate
python3 comprehensive_benchmark.py 2>&1 | tee comprehensive_log.txt
# Press Ctrl+B, then D to detach
```

### 8.6 Check Server Status

```bash
cd /home/albert/code/AInfra/draft
bash check_remote_server.sh

# Or manually:
ssh -p 20514 root@connect.westd.seetacloud.com "curl -s http://localhost:6006/health"
```

---

## Appendix A: File Index

### Draft Side (draft/)

| File | Purpose | Lines |
|------|---------|-------|
| `client/edge_client.py` | Main speculative decoding orchestration | 270 |
| `draft_generator.py` | Draft model inference | 200 |
| `client/http_cloud_client.py` | HTTP client to cloud | 150 |
| `protocol.py` | Communication data structures | 128 |
| `model_manager.py` | Model loading & management | 100 |
| `config.py` | Configuration | 34 |
| `start_ssh_tunnel.sh` | SSH tunnel with auto-reconnect | 80 |
| `experiments/comprehensive_benchmark.py` | Main benchmark script | 313 |
| `experiments/run_experiment.sh` | Tmux wrapper for experiments | 60 |

### Cloud Side (remote-verify/)

| File | Purpose | Lines |
|------|---------|-------|
| `src/api_server.py` | FastAPI HTTP server | 249 |
| `src/cloud_server.py` | Verification logic | 220 |
| `start.sh` | Server startup script | 100 |
| `start_background.sh` | Background server starter | 66 |
| `stop_background.sh` | Background server stopper | 46 |

### Experiment Artifacts

**Location**: `draft/experiments/`

| File | Description |
|------|-------------|
| `comprehensive_benchmark.py` | Main experiment script |
| `comprehensive_results.json` | Raw experimental data |
| `comprehensive_log.txt` | Detailed experiment log |
| `EXPERIMENT_STATUS.md` | Current experiment status |

---

*Document version: 2.0*  
*Last updated: 2026-04-14*  
*Changes: Updated SSH port to 20514, changed draft model to 3B, target model to 32B-GPTQ, added troubleshooting section, removed analysis content*
