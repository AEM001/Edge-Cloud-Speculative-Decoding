# Tree-Based Speculative Decoding Diagnostic Report

**Date**: 2026-05-05  
**Purpose**: Diagnose why tree-based speculative decoding underperforms direct generation

---

## Summary

Direct generation timing was found to be **server-side slow** (3.0s per 128 tokens) under CUDAGraphs configuration, not a measurement bug. After restoring `enforce_eager=True`, direct speed returned to normal (~43 tok/s). 

Tree-based speculative decoding was fundamentally misimplemented: it verified speculative branches in the same round as the base branch, adding a full extra RTT per round. After fixing this to prefetch-and-reuse semantics, tree improved significantly but still underperforms direct due to poor prediction quality of the rejection point.

---

## Hardware Configuration

```
GPU 0 (RTX 4090)  →  cloud / verify server   Qwen2.5-14B-Instruct-AWQ
GPU 1 (RTX 4090)  →  edge  / draft model     Qwen2.5-3B-Instruct-AWQ
```

Server config: `VLLM_ATTENTION_BACKEND=sdpa`, `enforce_eager=True` (restored for fairness)

---

## Direct Timing Analysis

### Issue Found

Initial quicktest showed direct generation at **9 tok/s**, which was incorrectly suspected to be a timing measurement bug.

### Instrumentation Added

Split direct timing into components:
- `server_ms`: Actual server-reported generation time
- `http_ms`: Wall-clock HTTP call duration (includes server)
- `sim_ul_ms`: Simulated uplink delay
- `sim_dl_ms`: Simulated downlink delay

### Root Cause

The slow direct was caused by the verify server configuration:
- **With CUDAGraphs** (`enforce_eager=False`): `/generate` took ~13.8s, direct ≈ 9 tok/s
- **With eager mode** (`enforce_eager=True`): `/generate` took ~2.9–3.0s, direct ≈ 43 tok/s

The CUDAGraphs configuration optimized verification but severely degraded direct generation.

### Fix Applied

Restored `enforce_eager=True` in `scripts/server/verify_server.py` for fair comparison.

### Current Direct Performance

| Network | tok/s | server_ms | http_ms | sim_ul_ms | sim_dl_ms |
|---------|-------|-----------|---------|-----------|-----------|
| good    | 43.0  | ~3000     | ~3005   | 27        | 27        |
| medium  | 43.0  | ~2900     | ~2905   | 27        | 27        |

---

## Tree-Based Decoding Diagnosis

### Original Misimplementation

The tree client was verifying **all branches in the same round**:

```
Round timeline (WRONG):
  t=0       Draft base (K tokens)
  t=base_ms Draft branches (K × N_branches)
  t=total_draft_ms → send ALL branches to verify (batch)
  t=total_draft_ms + RTT → results back
```

This added **branch_draft_ms + second RTT** per round, making tree slower than sync.

### Diagnostic Instrumentation Added

Per-round tree metrics:
- `base_accept`: Actual accepted length from base branch
- `base_draft_ms`: Time to draft base
- `branch_draft_ms`: Time to draft speculative branches
- `base_wait_ms`: Time waiting for base verify
- `total_wait_ms`: Total wait time
- `spec_verify_wall_ms`: Wall time for speculative branch verify
- `stale_branches`: Number of unused branches
- `prefetched_tokens`: Number of tokens carried to next round

### Diagnostic Findings

Before fix:
```
spec_verify_wall ≈ 382–503ms  ← extra RTT killing performance
base_draft ≈ 35ms
branch_draft ≈ 38ms
```

After fix (removed speculative branch verify):
```
spec_verify_wall = 0ms  ← eliminated
base_draft ≈ 15ms
branch_draft ≈ 35–39ms
base_wait ≈ 75–192ms
```

### Fixed Algorithm

Now implements correct prefetch semantics:

```
Round timeline (CORRECT):
  t=0       Draft base (K tokens)
  t=base_ms FIRE base verify in background thread
  t=base_ms Draft branches from predicted offsets (overlaps with verify RTT)
  t=base_ms + branch_draft_ms → wait for base verify
  t=base_ms + RTT → base verify returns with accepted_len + correction
  → Find branch where offset == accepted_len and first token == correction
  → If found: reuse remaining branch tokens as next round's draft (prefetch)
  → If not found: discard all branch work
```

This eliminates the extra RTT and actually hides branch drafting behind the verify RTT.

---

## Current Performance (After Fixes)

Bursty network temporarily disabled for diagnostic clarity.

| Network | direct | sync_k7       | tree_k7_b3     |
|---------|--------|---------------|----------------|
| good    | 43.0   | 37.7 (0.876x) | 40.6 (0.944x)  |
| medium  | 43.0   | 31.1 (0.723x) | 32.0 (0.744x)  |

### Tree Diagnostics

```
good:
  base_accept ≈ 4.91
  selected_offset ≈ 0.6
  prefetched_tokens: low (offset miss)

medium:
  base_accept ≈ 4.12
  selected_offset ≈ 3.4
  prefetched_tokens: low (offset miss)
```

---

## Remaining Bottleneck: Prefetch Hit Quality

### Problem

The tree can only benefit when:
1. `branch.offset == actual base_accept`
2. `branch.draft_ids[0] == correction_token`

Diagnostics show the offset prediction is inaccurate:
- Actual `base_accept`: 4.1–4.9
- Predicted `selected_offset`: 0.6–3.4

Most branch draft work is wasted because the predicted rejection point doesn't match the actual one.

### Offset Prediction Strategy

Current implementation uses:
- **Without history**: Fixed at 0.6 × K (≈ 4.2 for K=7)
- **With history**: Mode of past accepted lengths, ± 1

The mode-based prediction is not yet effective because:
- History window (50) may be too large
- Acceptance patterns change per prompt type
- The offset must be **exact**, not close

### Why Tree Still Underperforms

Even with correct prefetch semantics:
- **Branch drafting cost**: ~35–39ms per round
- **Prefetch hit rate**: Very low (offset miss)
- **Effective cost**: `base_draft + branch_draft + RTT` ≈ 15ms + 38ms + 75–192ms = 128–245ms

Sync cost: `draft + RTT` ≈ 35ms + 75–192ms = 110–227ms

Tree is essentially paying extra branch draft time without enough prefetch hits to amortize it.

---

## Files Modified

### Server Configuration
- `scripts/server/verify_server.py`
  - Restored `enforce_eager=True` (line 111)
  - Restored for fair direct speed comparison

### Quick Test
- `scripts/experiments/quick_test.py`
  - Disabled bursty network temporarily (line 334)
  - Added `DirectTiming` dataclass for detailed timing breakdown (lines 72–80)
  - Modified `_direct_with_throttle()` to return `DirectTiming` (lines 93–127)
  - Updated `run_direct_case()` to log detailed timing (lines 174–184)
  - Added tree diagnostic logging (lines 275–285)

### Tree Client
- `scripts/experiments/tree_async_client.py`
  - Added `prefetched_ids` and `prefetched_logprobs` for token reuse (lines 136–137)
  - Modified to use prefetched tokens when available (lines 161–177)
  - Removed speculative branch verification in same round (lines 259–284)
  - Added prefetch reuse logic (lines 301–313)
  - Improved offset prediction using mode ± 1 (lines 94–100)
  - Added detailed per-round diagnostics (lines 340–341)
  - Fixed `total_drafted_tokens` to count only base draft (line 266)
  - Fixed `stale_branches` to count from all branches (line 334)

---

## Recommendations

### Short-term
1. **Improve offset prediction accuracy**:
   - Use smaller history window (20 instead of 50)
   - Consider recent trend (last 5–10) more than global mode
   - Add fallback to offset=0 when confidence is low

2. **Reduce branch count when confidence is low**:
   - Dynamic branch width based on prediction confidence
   - Only draft branches when mode frequency is high (> 30%)

### Medium-term
3. **Alternative tree strategy**:
   - Instead of exact offset matching, use **soft matching**
   - Accept branch if offset is within ± 1 of actual accepted
   - Requires recomputing prefix for slight misalignment

4. **Hybrid approach**:
   - Start with sync, switch to tree when prediction confidence stabilizes
   - Use tree only on later rounds when history is sufficient

### Long-term
5. **Consider alternative architectures**:
   - The fundamental issue: tree requires **exact** prediction of rejection point
   - Speculative decoding may benefit more from **better draft model** than complex tree
   - Async with conservative prefix selection (CQT) may be more robust

---

## Conclusion

The tree-based approach is now correctly implemented with proper prefetch semantics and no extra RTT overhead. However, the requirement for **exact offset prediction** makes it fragile. Without high prediction accuracy, the branch drafting cost outweighs the benefit. The remaining work is primarily an ML prediction problem (accurate rejection point prediction) rather than a systems engineering problem.
