# Guidance Reuse Experiment Scripts

These scripts prepare the next research experiments from `spec/experiment.md`:
cloud guidance reuse degradation, fixed update tradeoffs, observable update
signals, and a first threshold-based adaptive policy.

## Output Layout

All experiment artifacts are grouped by run:

```text
outputs/guidance_reuse/<run_id>/
  configs/guidance_reuse_sweep.json
  manifest.json
  raw/quick_test_results_<timestamp>.json
  analysis/guidance_reuse_summary.csv
  analysis/guidance_reuse_rounds.csv
  analysis/guidance_reuse_report.md
  analysis/update_signal_correlations.csv
  analysis/adaptive_policy_simulation.csv
```

The raw quick result is also kept in the existing `outputs/` location because
`quick_test.py` already writes there.

## Step 1: Prepare Sweep Config

```bash
python3 scripts/experiments/prepare_guidance_reuse_configs.py \
  --reuse-windows 1,2,4,8,16,32 \
  --prompt-count 3 \
  --max-tokens 256 \
  --prompt-input-tokens 2048
```

Use the printed `CONFIG_PATH` in the next step. The generated profiles are:

- `reuse_R1`: every-round cloud attention refresh, used as the oracle proxy.
- `reuse_R{N}`: fixed interval refresh every `N` speculative rounds.
- `no_update`: no target-attention refresh, lower-bound sparse KV baseline.

## Step 2: Run Sweep

Start the verify server first:

```bash
bash start_verify.sh
```

Then run:

```bash
python3 scripts/experiments/run_guidance_reuse_sweep.py \
  outputs/guidance_reuse/<run_id>/configs/guidance_reuse_sweep.json
```

The script copies the raw result into `<run_id>/raw/` and updates
`manifest.json`.

## Step 3: Analyze R Reuse Degradation

```bash
python3 scripts/experiments/analyze_guidance_reuse.py \
  outputs/guidance_reuse/<run_id>/raw/quick_test_results_<timestamp>.json
```

This writes summary and per-round CSVs. If `reuse_R1` exists, its selected KV
sets are used as the per-round oracle proxy for `oracle_overlap`.

## Step 4: Correlate Update Signals

```bash
python3 scripts/experiments/analyze_update_signals.py \
  outputs/guidance_reuse/<run_id>/analysis/guidance_reuse_rounds.csv
```

This checks whether edge-visible signals such as recent accepted length and
staleness correlate with next-round degradation. `oracle_overlap` is diagnostic
only; it should not be used as an edge-side trigger.

## Step 5: Simulate A Simple Adaptive Policy

```bash
python3 scripts/experiments/simulate_adaptive_update_policy.py \
  outputs/guidance_reuse/<run_id>/analysis/guidance_reuse_rounds.csv \
  --thresholds 1,2,3,4,5,6 \
  --window 4 \
  --update-cost-ms 0
```

This estimates how often a threshold policy would request cloud guidance before
implementing it in the live client.
