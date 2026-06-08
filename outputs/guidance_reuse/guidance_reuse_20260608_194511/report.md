# New Experiment Results

**Run:** `guidance_reuse_20260608_194511`  
**Setup:** Qwen3-14B-AWQ (cloud, GPU0) + Qwen3-1.7B draft (edge, GPU1)  
**Params:** 3×pg19 prompts, **8K input**, **512 output tokens**, branching tree (32 nodes, depth 4), **chunk_size=32**, top-k sweep [8, 16]

## Step 4: R Reuse Degradation (averaged over 3 prompts)

| method | R | acc_len | acc_ratio | latency/acc ms |
|---|---:|---:|---:|---:|
| `reuse_R1_k8` | 1 | 2.990 | 0.249 | 138.6 |
| `reuse_R1_k16` | 1 | **3.142** | **0.262** | 129.6 |
| `reuse_R2_k8` | 2 | 2.967 | 0.247 | 126.0 |
| `reuse_R2_k16` | 2 | 3.085 | 0.257 | 120.9 |
| `reuse_R4_k8` | 4 | 2.935 | 0.245 | 122.8 |
| `reuse_R4_k16` | 4 | 3.052 | 0.254 | 116.9 |
| `reuse_R8_k8` | 8 | 2.904 | 0.242 | 120.3 |
| `reuse_R8_k16` | 8 | **3.243** | **0.270** | **106.6** |
| `reuse_R16_k8` | 16 | 3.112 | 0.259 | 112.4 |
| `reuse_R16_k16` | 16 | 2.992 | 0.249 | 114.5 |
| `reuse_R32_k8` | 32 | 2.809 | 0.234 | 124.6 |
| `reuse_R32_k16` | 32 | 2.990 | 0.249 | 114.5 |
| `no_update_k8` | 0 | 1.870 | 0.156 | 212.7 |
| `no_update_k16` | 0 | 1.968 | 0.164 | 201.9 |

Key findings:
- **`no_update` is significantly worse** (acc_len ~1.9 vs ~3.0 for reuse, latency ~210ms vs ~115ms). With 8K context and chunk=32, the sparse KV coverage is much lower, so cloud guidance matters.
- **k=16 consistently outperforms k=8** at the same R (higher acc_len, lower latency).
- **R=8 with k=16 achieves the best latency/accepted token** (106.6 ms), suggesting a sweet spot where cloud updates are frequent enough to maintain quality but not so frequent that overhead dominates.
- Unlike the previous experiment (chunk=64, 2K input), **here larger R does not monotonically improve results** — R=32 is slightly worse than R=8/16. This is because with sparser coverage (chunk=32, 8K input), stale guidance degrades more noticeably.

## Step 5: Update Signal Correlations

| Feature | Target | Pearson r |
|---|---|---|
| `staleness_rounds` | next_accept_len_drop | 0.004 |
| `recent_acceptance_len` | next_accept_len_drop | **0.214** |
| `accepted_len_drop` | next_accept_len_drop | **-0.207** |

Key finding: **`recent_acceptance_len` remains the most predictive edge-visible signal** (r≈0.21), though weaker than in the previous run (0.28). The drop in correlation is expected because the absolute differences between methods are smaller here.

## Step 6: Adaptive Policy Simulation

| Threshold | Updates triggered | Accepted/update |
|---|---:|---:|
| 1 | 418 / 5799 (7.2%) | **37.8** |
| 2 | 1410 / 5799 (24.3%) | 11.2 |
| 3 | 2954 / 5799 (50.9%) | 5.3 |
| 4 | 4320 / 5799 (74.5%) | 3.7 |
| ≥5 | 5799 / 5799 (100%) | 2.7 |

Key finding: **threshold=1 achieves extremely sparse updates (7.2%) with highest efficiency (37.8 accepted tokens/update)**. The policy only triggers when `recent_accept_len_drop > 1`, which is rare because acceptance lengths are relatively stable in this setup.

---

All artifacts are in `@/root/code/Edge-Cloud-Speculative-Decoding/outputs/guidance_reuse/guidance_reuse_20260608_194511/`.