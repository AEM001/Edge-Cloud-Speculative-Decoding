# Static K Baseline Experiment

## Goal

**Question:** How does static K perform under changing network conditions and prompt difficulty in a real edge-cloud setup?

**Deliverable:** Baseline results showing performance patterns for K=2, K=4, K=6.

---

## Current System Status

**Implementation is complete.** See existing documentation:
- Architecture, models, metrics: `README.md`
- Performance bottleneck analysis: `docs/PERFORMANCE_ISSUE.md`
- Code: `run_experiment.py`, `client/edge_client.py`

**Known Issue:** Draft model (Mac M1 8GB) is **4.4x slower** than server verification (RTX 3060). See `docs/PERFORMANCE_ISSUE.md` for details. This experiment proceeds to establish baseline behavior despite the hardware limitation.

---

## Experiment Configuration

### Hardware
- **Edge:** MacBook Air M1 8GB (draft model)
- **Cloud:** RTX 3060 12GB (verification server)

### Models
- **Draft:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- **Target:** `Qwen2.5-3B-Instruct-4bit`

### Policies (Static K)
- K = 2 (conservative)
- K = 4 (balanced)
- K = 6 (aggressive)

### Workloads
- **Easy:** Natural language prompts (`prompts_easy.json`, 30 prompts)
- **Hard:** Structured reasoning prompts (`prompts_hard.json`, 30 prompts)

### Network Regimes
| Regime | RTT | Bandwidth |
|--------|-----|-----------|
| Good | 10 ms | 100 Mbps |
| Medium | 40 ms | 20 Mbps |
| Bad | 100 ms | 5 Mbps |
| Bursty | 20/120 ms alternating | 50/5 Mbps alternating |

---

## Experiment Matrix

- 3 policies × 2 workloads × 4 networks = **24 conditions**
- 30 prompts per workload = **720 total request runs**
- Reduce to 20 prompts if needed: **480 runs**

---

## Execution

### Prerequisites
```bash
# Verify server is accessible
python run_experiment.py --server-url http://<server>:8000 --test-connection
```

### Run Full Experiment
```bash
python run_experiment.py \
  --server-url http://<server>:8000 \
  --output-dir experiments/results_$(date +%Y%m%d_%H%M%S) \
  --max-tokens 128 \
  --num-prompts 30
```

### Run Quick Test (5 prompts)
```bash
python run_experiment.py \
  --server-url http://<server>:8000 \
  --output-dir experiments/quick_test \
  --num-prompts 5
```

---

## Key Metrics

Per-request (collected in `benchmarks/metrics_logger.py`):
- Total latency, tokens/sec, TPOT
- Acceptance ratio, wasted drafted tokens
- Average RTT, server verify time, edge draft time

Per-condition (aggregated automatically):
- Mean/p50/p95 latency
- Mean acceptance ratio and variance
- Mean tokens/sec

---

## Hypotheses

| ID | Statement |
|----|-----------|
| H1 | No single static K is optimal across all conditions |
| H2 | K=6 wins in good networks; K=2 is safer in bad networks |

---

## Expected Results

- **K=6:** Best in good networks, worst in bad/bursty
- **K=2:** Safest in bad networks, under-performs in good
- **K=4:** Middle ground but suboptimal at extremes
- **Conclusion:** Static K fails under changing conditions → adaptive policies needed

---

## Deliverables

1. **Table 1:** Mean latency and TPOT by policy × network × workload
2. **Table 2:** Acceptance ratio and wasted drafted tokens per policy
3. **Figure 1:** Latency vs K value under each network regime
4. **Figure 2:** Acceptance ratio distribution across prompts for each K
5. **Figure 3:** p95 latency comparison across K values

Results are auto-saved to the output directory specified in `--output-dir`.

