# MLX Speculative Decoding Edge Node

A Mac-based edge node for speculative decoding experiments using MLX-accelerated draft models.

## Overview

This system implements an edge-side draft model for speculative decoding, running locally on Apple Silicon using MLX. The edge node receives verified prefixes from a server, generates draft tokens with confidence statistics, and returns them for verification.

## Architecture

```
Edge (Mac)                    Server
    │                              │
    │─── verified prefix ─────────>│
    │                              │
    │<───── draft tokens ──────────│
    │  + confidence stats          │
    │                              │
    │                              │─── verify drafts
    │                              │
    │<───── final output ──────────│
```

## Experiment Status

### Current Status: Performance Bottleneck Identified

**Issue**: Draft model (Mac) is **4.4x slower** than server verification (Ubuntu)

| Component | Time | Expected |
|-----------|------|----------|
| Draft generation (0.5B 4-bit) | 200ms | <20ms |
| Server verify (3B 4-bit) | 45ms | - |
| **Ratio** | **4.4:1** | **1:10+** |

**Result**: Speculative decoding is **5x slower** than direct server calls on current hardware.

### Root Cause

1. **Hardware limitation**: MacBook Air M1 8GB GPU has insufficient compute
2. **Model already minimal**: 0.5B 4-bit is the smallest available Qwen2.5 model
3. **No optimization space**: 2-bit quantization tested but slower (350ms vs 215ms)

### Hardware Suitability

**MacBook Air M1 8GB is NOT suitable for edge-cloud speculative decoding research.**

Speculative decoding requires draft model to be 10-100x faster than target model. Current hardware achieves the opposite (draft is 4.4x slower).

For research purposes, consider:
- **MacBook Pro M3 18GB**: Marginal (~50ms draft)
- **Mac Studio M2 Ultra 64GB**: Suitable (~10ms draft)
- **NVIDIA GPU (Jetson Orin)**: Ideal for edge deployment

### Documentation

See `docs/EXPERIMENT_STATUS.md` for detailed analysis and next steps.

## Project Structure

```
mac-draft/
├── config.py              # Configuration settings (model path, params)
├── model_manager.py       # Model loading with custom storage
├── draft_generator.py     # Draft token generation with confidence stats
├── protocol.py            # Data structures for communication
├── metrics.py             # Metrics collection and aggregation
├── benchmarks.py          # Benchmark prompt selection
├── main.py               # Entry point with demo
├── fetch_data.py         # Dataset download script
├── requirements.txt      # Python dependencies
├── benchmarks/           # Selected benchmark prompts
│   ├── prompts.json
│   ├── prompts.txt
│   └── summary.json
├── data/                 # Raw datasets (gitignored)
│   ├── bespoke_stratos.json
│   └── conversation_chronicles.json
└── results/              # Experiment results (gitignored)
```

## Installation

```bash
# Install dependencies with uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Dependencies

- `mlx-lm` - MLX language model library
- `mlx` - MLX framework for Apple Silicon
- `numpy` - Numerical computing
- `datasets` - HuggingFace datasets library

## Quick Start

### 1. Download Datasets

```bash
# Download both datasets (requires HF_TOKEN)
python fetch_data.py
```

This downloads:
- **Bespoke-Stratos-17k**: 16,610 hard reasoning prompts (34 MB)
- **ConversationChronicles**: 785,129 conversation prompts (687 MB)

### 2. Generate Benchmarks

```bash
# Select 20 diverse prompts with fixed lengths
python benchmarks.py
```

This creates:
- Target prompt length: 512 tokens
- Target completion length: 128 tokens
- 10 prompts from each dataset

### 3. Run Draft Model

```bash
# Load model and generate draft tokens
python main.py
```

Example output:
```
Generated 5 draft tokens:
  [0] token=29871, text=' How'
       prob=0.1775, logprob=-1.7279
       max_prob=0.1775, entropy=nan, top_margin=nan
  ...
Draft text: ' How can I assist you'
```

## Model

**Draft Model**: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`

- **Size**: 0.5B parameters, 4-bit quantized (~278 MB)
- **Hardware**: Apple Silicon (Metal acceleration)
- **Storage**: `~/.models/mlx-draft-model` (configurable in `config.py`)

## Protocol

### Draft Request

```python
{
    "verified_prefix": [token_ids...],  # Verified context
    "num_draft_tokens": 5               # K value
}
```

### Draft Response

```python
{
    "draft_token_ids": [token_ids...],
    "logprobs": [logprobs...],
    "probabilities": [probs...],
    "confidence_stats": {
        "max_probs": [...],
        "entropies": [...],
        "top_margins": [...]
    }
}
```

### Confidence Statistics

Per drafted token:
- **max_prob**: Maximum probability in distribution
- **entropy**: Distribution entropy (uncertainty measure)
- **top_margin**: Top-1 minus top-2 probability gap

## Metrics

The `metrics.py` module collects comprehensive performance metrics.

### Per-Request Metrics

- `total_latency`: Total time from request to completion
- `generated_tokens`: Total tokens generated by server
- `tokens_per_sec`: Generated tokens / total latency
- `tpot`: Time per output token
- `total_rounds`: Number of verification rounds
- `mean_k_chosen`: Average K value across requests
- `acceptance_ratio`: Accepted / drafted tokens
- `total_drafted_tokens`: Total tokens drafted by edge
- `total_accepted_drafted_tokens`: Tokens accepted by server
- `wasted_drafted_tokens`: Drafted - accepted
- `uplink_bytes`: Bytes sent to server
- `downlink_bytes`: Bytes received from server
- `average_rtt`: Round-trip time
- `average_server_verify_time`: Server verification time
- `average_edge_draft_time`: Edge drafting time

### Per-Condition Metrics

- `mean_latency`: Average latency
- `p50_latency`: 50th percentile latency
- `p95_latency`: 95th percentile latency
- `mean_tpot`: Average time per output token
- `mean_tokens_per_sec`: Average tokens per second
- `mean_acceptance_ratio`: Average acceptance ratio
- `mean_wasted_drafted_tokens`: Average wasted tokens
- `variance_acceptance_ratio`: Variance of acceptance ratio

### Usage Example

```python
from metrics import MetricsCollector

collector = MetricsCollector(output_dir="results")

# Start request
collector.start_request("req_001")

# Draft phase
collector.start_draft()
# ... generate drafts ...
collector.end_draft()

# Network phase
collector.start_network()
# ... send to server ...
collector.end_network()

# Verify phase
collector.start_verify()
# ... server verifies ...
collector.end_verify()

# Record results
collector.record_request(
    generated_tokens=128,
    drafted_tokens=25,
    accepted_tokens=20,
    k_chosen=5,
    rounds=3,
    uplink_bytes=1024,
    downlink_bytes=2048,
)

# Save metrics
collector.save_metrics(condition_name="baseline")
```

## Configuration

Edit `config.py` to customize:

```python
MODEL_NAME = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MODEL_PATH = Path.home() / ".models" / "mlx-draft-model"
MAX_DRAFT_TOKENS = 5
TEMPERATURE = 0.8
TOP_P = 0.95
```

## Datasets

### Bespoke-Stratos-17k
- **Source**: HuggingFaceH4/Bespoke-Stratos-17k
- **Size**: 16,610 samples
- **Content**: Hard reasoning tasks with thinking traces
- **Format**: System + user messages

### ConversationChronicles
- **Source**: Dans-DiscountModels/ConversationChronicles-sharegpt
- **Size**: 785,129 samples
- **Content**: Multi-turn conversations
- **Format**: ShareGPT (system/human/assistant)

## Benchmarks

Generated benchmarks are stored in `benchmarks/`:

- `prompts.json`: JSON format with formatted prompts
- `prompts.txt`: Human-readable version
- `summary.json`: Statistics about selected prompts

## Development

### Adding New Metrics

Extend `RequestMetrics` dataclass in `metrics.py`:

```python
@dataclass
class RequestMetrics:
    # ... existing fields ...
    new_metric: float  # Add new field
```

### Custom Benchmark Selection

Modify `benchmarks.py`:

```python
NUM_BENCHMARKS = 50  # Change number
TARGET_PROMPT_LENGTH = 1024  # Change length
```

## Troubleshooting

### Model Download Issues

If download fails, check HF_TOKEN:
```bash
export HF_TOKEN=your_token
python fetch_data.py
```

### Memory Issues

For large datasets, use streaming mode in `fetch_data.py`.

### MLX Not Found

Ensure MLX is installed for Apple Silicon:
```bash
uv pip install mlx mlx-lm
```

## License

MIT License

## References

- [MLX](https://github.com/ml-explore/mlx)
- [Speculative Decoding](https://arxiv.org/abs/2305.09181)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
