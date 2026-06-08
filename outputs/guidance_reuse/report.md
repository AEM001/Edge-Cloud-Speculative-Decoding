All 6 steps complete. Here's a summary of results:

---

# Experiment Results Summary

**Run:** `guidance_reuse_20260608_170806`  
**Setup:** Qwen3-14B-AWQ (cloud, GPU0) + Qwen3-1.7B draft (edge, GPU1), 3×pg19 prompts, 256 output tokens, branching tree (32 nodes, depth 4)

## Step 4: R Reuse Degradation (`guidance_reuse_report.md`)

| Method | R | Avg acc_len | Avg acc_ratio | Avg latency/accepted (ms) | oracle_overlap |
|---|---|---|---|---|---|
| `reuse_R1` | 1 | 1.97 | 0.165 | 234 | **1.000** |
| `reuse_R2` | 2 | 2.06 | 0.172 | 207 | 0.763 |
| `reuse_R4` | 4 | 2.11 | 0.176 | 186 | 0.775 |
| `reuse_R8` | 8 | 2.18 | 0.182 | 174 | 0.740 |
| `reuse_R16` | 16 | 2.25 | 0.188 | 163 | 0.725 |
| `reuse_R32` | 32 | **2.42** | **0.202** | **148** | 0.666 |
| `no_update` | 0 | 1.32 | 0.110 | 278 | 0.000 |

Key finding: **larger R → fewer cloud updates → lower latency/token and surprisingly higher acceptance length** (likely due to less retrieval overhead). `no_update` is clearly worst. `oracle_overlap` degrades gradually with R.

## Step 5: Update Signal Correlations

| Feature | Target | Pearson r |
|---|---|---|
| `staleness_rounds` | next_accept_len_drop | ~0 (0.008) |
| `recent_acceptance_len` | next_accept_len_drop | **0.284** |
| `recent_acceptance_len` | stale_oracle_label | **-0.251** |
| `accepted_len_drop` | next_accept_len_drop | -0.186 |

Key finding: **`recent_acceptance_len` is the most predictive edge-visible signal** (r≈0.28). Raw staleness is nearly uncorrelated (r≈0.01), meaning round count alone isn't a good trigger.

## Step 6: Adaptive Policy Simulation

| Threshold | Updates triggered | Accepted/update |
|---|---|---|
| 1 | 290 / 1810 rounds (16%) | 12.5 |
| 2 | 874 / 1810 rounds (48%) | 4.1 |
| 3 | 1441 / 1810 rounds (80%) | 2.5 |
| ≥5 | 1810 / 1810 (100%) | 2.0 |

Key finding: **threshold=1 achieves sparse updates (16% of rounds) with highest efficiency (12.5 accepted tokens/update)**.

---

All output artifacts are in `@/root/code/Edge-Cloud-Speculative-Decoding/outputs/guidance_reuse/guidance_reuse_20260608_170806/`.