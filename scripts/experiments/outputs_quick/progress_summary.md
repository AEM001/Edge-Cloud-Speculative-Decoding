# Research Progress Summary

## Brief Overview

1. **Strengthened baseline** - Improved implementation details and measurement accuracy
2. **Asynchronous pipeline** - Parallel draft generation during verification RTT to hide network latency
3. **Multi-offset pre-draft** - Predict acceptance length at multiple offsets to maximize branch reuse probability
4. **Streaming generation with early termination** - Token-by-token branch drafting that stops when verification arrives
5. **Bimodal acceptance distribution** - Discovered acceptance length is bimodal, not smooth; average acceptance rate is misleading
6. **K adjustment trap** - Directly increasing K can increase RTT and mask draft time savings
7. **Draft-side heuristic** - Dynamic pre-draft K adjustment based on acceptance distribution, with significant performance gains

---

## Detailed Progress

### 1. Strengthened Baseline

- Improved measurement accuracy and instrumentation
- Enhanced baseline comparison methodology
- Fixed bugs in synchronous speculative decoding implementation
- Established reliable performance comparison framework

### 2. Asynchronous Pipeline Implementation

**Core Innovation:** Hide verification RTT by parallelizing draft generation

- Draft generation runs in background thread while cloud verification is in flight
- Critical path only waits for verification, not for draft completion
- Verification thread: sends base draft to cloud server and waits for response
- Draft thread: generates speculative branches in parallel
- Achieves async speedup by overlapping local compute with remote verification

**Key Insight:** The bottleneck is network RTT, not local draft time. By making draft work asynchronous, we hide this latency.

### 3. Multi-Offset Pre-Draft Strategy

**Core Innovation:** Predict rejection point and generate multiple parallel branches

- Instead of single prediction, generate branches at multiple predicted offsets
- Offsets centered around `BASE_ACCEPTANCE_RATIO = 0.60` based on historical acceptance rate
- For K=8, typical offsets: [8, 4, 5] (full acceptance, lower-mid, center)
- Each branch drafts from a different position in the base draft sequence
- When verification returns, select branch that matches verified path

**Reuse Condition:** Branch is reusable only if:
```python
branch.offset <= accepted_len
branch.draft_ids starts with base_draft[branch.offset:accepted_len] + correction_token
```

**Benefit:** Multiple offsets increase probability of matching actual acceptance point, enabling branch reuse in more rounds.

### 4. Streaming Generation with Early Termination

**Core Innovation:** Token-by-token branch drafting with immediate stop on verification

- Branch drafts are generated incrementally, not all-at-once
- Each branch thread generates one token at a time in a loop
- When verification result arrives, `branch_stop.set()` signals all branch threads to stop
- Even partial branches can be useful if they bridge to the verified path
- Exposed branch time is near-zero (0.052ms good, 0.009ms medium network)

**Implementation Details:**
- `_run_branch_draft()` function streams tokens one-by-one
- Checks `branch_stop.is_set()` after each token generation
- Partial branches captured via `_snapshot_streamed_branches()`
- Early termination ensures branch work doesn't block critical path

### 5. Discovery: Bimodal Acceptance Distribution

**Key Finding:** Acceptance length distribution is bimodal, not smooth Gaussian

For sync K=8 across network profiles:

| Accepted len | Count | Percentage |
| ---: | ---: | ---: |
| 0 | 60 | ~14% |
| 1 | 43 | ~10% |
| 2 | 34 | ~8% |
| 3 | 23 | ~5% |
| 4 | 24 | ~6% |
| 5 | 20 | ~5% |
| 6 | 10 | ~2% |
| 7 | 12 | ~3% |
| 8 | 209 | ~48% |

**Implications:**
- Full acceptance (K=8) is the dominant mode at 48%
- Very short acceptance (0-2) is secondary mode at 31%
- Mean acceptance (~5.04) is misleading - distribution has two peaks
- Optimization should target the full-acceptance case, not the mean
- Smooth offset ladders are suboptimal; prioritize full-acceptance offset

### 6. Discovery: K Adjustment Trap

**Key Finding:** Directly increasing K can backfire due to RTT increase

- Larger K means more tokens to send per verification round
- Increased payload size → higher RTT → longer verification time
- Draft time savings can be masked by increased network latency
- This is especially problematic on high-latency networks

**Example:**
- Increasing K from 7 to 8 might save 10ms draft time
- But adds 5ms RTT due to larger payload
- Net gain is only 5ms, not the expected 10ms

**Implication:** K optimization must consider the full pipeline, not just draft time.

### 7. Draft-Side Heuristic for Adaptive K

**Core Innovation:** Dynamic pre-draft K adjustment based on acceptance distribution

**Problem Solved:** Previous implementation had a pipeline bug:
- Reused partial prefetch could shorten the base draft sent to verification
- This caused extra verification rounds
- Erased the benefit of branch pre-drafting

**Solution:**
1. **Top-up prefetched drafts:** When reusing a prefetched branch, top it up to the policy K before verification
   ```python
   remaining_k = k - len(base_draft)
   if remaining_k > 0:
       topup_resp = self.draft_generator.generate_draft_tokens(...)
   ```

2. **Prioritize full-acceptance offset:** Tree branch offsets now place a branch at K (full acceptance) first, then lower offsets
   - Matches the bimodal distribution where full acceptance is dominant
   - Full-offset branch selected in ~45% of rounds (matches 48% full-acceptance rate)

**Results:**
- Tree beats direct by 1.40x (good network) and 1.15x (medium network)
- Tree beats sync by 1.07x-1.09x consistently
- Tree beats sync on 40/40 paired prompt/network cases
- Prefetch reuse is bursty (median 0 tokens, 90th percentile 7 tokens) - large wins on full-acceptance rounds

**Key Insight:** The heuristic works because it aligns with the actual distribution. Full-acceptance is the single largest mode, so prioritizing that offset captures most of the value.

---

## Performance Impact

### Throughput Comparison

| Network | Direct | Sync K=8 | Tree K=8 B=3 | Tree vs Direct | Tree vs Sync |
| --- | ---: | ---: | ---: | ---: | ---: |
| good | 36.83 tok/s | 47.96 tok/s | 51.59 tok/s | 1.40x | 1.07x |
| medium | 37.34 tok/s | 39.33 tok/s | 42.90 tok/s | 1.15x | 1.09x |

### Branch Selection Distribution

| Network | Selected base offset 0 | Selected full offset 8 | Avg prefetched tokens |
| --- | ---: | ---: | ---: |
| good | 240 | 194 | 2.61 |
| medium | 240 | 194 | 3.03 |

Full-offset branch carries most of the useful gain, matching the 48% full-acceptance rate.

---

## Technical Implementation Locations

- **Asynchronous pipeline:** `tree_async_client.py` lines 307-316 (verify thread), 346-427 (branch thread)
- **Multi-offset strategy:** `_tree_offsets()` method (lines 164-188)
- **Streaming generation:** `_run_branch_draft()` function (lines 346-427)
- **Early termination:** `branch_stop` Event (line 331, 467)
- **Adaptive K top-up:** lines 256-274
- **Branch reuse logic:** `_prefetch_from_branch()` method (lines 190-217)

---

## Future Directions

From the report, the next optimization should not be another mean-centered offset adjustment. The acceptance distribution is bimodal, so branch policy should:
- Continue to prioritize the full-acceptance case
- Treat lower offsets as fallback anchors
- Consider hybrid approaches (sync → tree when confidence high)
- Explore improved prediction using smaller history windows
