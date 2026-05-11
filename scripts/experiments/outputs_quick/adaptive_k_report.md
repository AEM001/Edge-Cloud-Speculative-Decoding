# Adaptive Draft-Side K Result Report

Date: 2026-05-10

## Scope

This report analyzes the latest `outputs_quick` run after the latest meaningful tree client change:

- Commit: `cbcc925 fix streaming and adaptive k for pre-draft`
- Code path: `scripts/experiments/tree_async_client.py`
- Result files:
  - `quick_test_summary.json`
  - `quick_test_results.json`
  - `quick_test_rounds.jsonl`

The later commit on top of this only changes ignore metadata, so the adaptive draft-side K / pre-draft reuse change is the relevant code change for this result set.

## Executive Summary

The latest run is the first strong validation that the adaptive draft-side pre-draft fix is doing useful work. Tree speculative decoding now beats both direct generation and sync speculative decoding under both tested network profiles.

Average throughput:

| Network | Direct | Sync K=8 | Tree K=8 B=3 | Tree vs Direct | Tree vs Sync |
| --- | ---: | ---: | ---: | ---: | ---: |
| good | 36.83 tok/s | 47.96 tok/s | 51.59 tok/s | 1.40x | 1.07x |
| medium | 37.34 tok/s | 39.33 tok/s | 42.90 tok/s | 1.15x | 1.09x |

The important signal is not only the average. In paired prompt comparisons, tree beats sync on every prompt:

| Network | Tree wins vs Sync | Mean Tree/Sync | Approx. 95% CI |
| --- | ---: | ---: | ---: |
| good | 20 / 20 | 1.073x | +/- 0.014x |
| medium | 20 / 20 | 1.085x | +/- 0.015x |

That is a clean result: the tree path is consistently adding value over sync speculative decoding.

## What Changed Technically

The meaningful change is in the tree async client:

1. Prefetched base drafts are now topped up back to the policy `K` before verification.
   - Before this, a reused partial prefetch could shorten the next verification draft.
   - That caused extra rounds and erased the benefit of branch pre-drafting.

2. Tree branch offsets now prioritize the full-acceptance offset first, then a lower/mid acceptance anchor.
   - For `K=8`, this makes the tree place a branch at the high-value full-acceptance position.
   - This matches the observed acceptance distribution, where full acceptance is the single largest mode.

## Acceptance Shape

The accepted length distribution is still not smooth. It remains bimodal:

For sync `K=8`, across each network profile:

| Accepted len | Count |
| ---: | ---: |
| 0 | 60 |
| 1 | 43 |
| 2 | 34 |
| 3 | 23 |
| 4 | 24 |
| 5 | 20 |
| 6 | 10 |
| 7 | 12 |
| 8 | 209 |

Key interpretation:

- Full acceptance (`8`) happens in about 48% of rounds.
- Very short acceptance (`0-2`) happens in about 31% of rounds.
- The mean accepted draft tokens per round is about 5.04.

This is exactly the shape where a full-offset branch is valuable. The tree does not need a smooth offset ladder; it needs to hit the dominant full-acceptance case while not breaking the lower-acceptance cases.

## Tree Reuse Behavior

Tree selection confirms that the full-offset branch is carrying most of the useful gain:

| Network | Selected base offset 0 | Selected full offset 8 | Avg prefetched tokens |
| --- | ---: | ---: | ---: |
| good | 240 | 194 | 2.61 |
| medium | 240 | 194 | 3.03 |

The full-offset branch is selected in about 45% of tree rounds, very close to the 48% full-acceptance rate. This alignment is the central reason the tree path now beats sync.

Prefetch reuse is bursty rather than smooth:

- Median prefetched tokens is `0`.
- 90th percentile prefetched tokens is `7`.

So the tree speedup comes from large wins on full-acceptance rounds, not small uniform wins every round.

## Latency Breakdown

Good network:

| Method | Total ms | Local draft ms | Server model ms | Simulated network ms | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 3380.6 | 0.0 | 3352.3 | 25.1 | n/a |
| sync K=8 | 2903.0 | 840.9 | 1455.1 | 549.4 | 21.75 |
| tree K=8 B=3 | 2718.2 | 2387.7 | 1447.4 | 548.1 | 21.70 |

Medium network:

| Method | Total ms | Local draft ms | Server model ms | Simulated network ms | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 3331.9 | 0.0 | 3273.7 | 55.3 | n/a |
| sync K=8 | 3523.7 | 821.9 | 1443.7 | 1205.2 | 21.75 |
| tree K=8 B=3 | 3269.5 | 2516.7 | 1439.5 | 1202.4 | 21.70 |

Tree has much higher local draft work because it drafts speculative branches. The reason it still wins is that this work is mostly overlapped with remote verification. The exposed branch time is effectively zero:

- good: `0.052 ms`
- medium: `0.009 ms`

This means branch drafting is not usually on the critical path.

## Why This Run Is Good

The previous failure mode was not low model agreement alone. It was a pipeline accounting problem: reused partial drafts could reduce the base draft length sent to verification, increasing verification rounds. After topping up reused prefetches back to `K`, tree and sync have almost identical accepted-token behavior:

| Network | Method | Acceptance rate | Accepted / round | Generated / round |
| --- | --- | ---: | ---: | ---: |
| good | sync K=8 | 0.663 | 5.290 | 6.290 |
| good | tree K=8 B=3 | 0.664 | 5.304 | 6.304 |
| medium | sync K=8 | 0.663 | 5.290 | 6.290 |
| medium | tree K=8 B=3 | 0.664 | 5.304 | 6.304 |

That is the expected behavior. Tree should not improve target-model acceptance directly; it should keep the same verification quality while hiding future draft latency. The new results match that model.

## Conclusion

The adaptive draft-side pre-draft change is validated by the latest run.

The strongest evidence is:

- Tree beats direct by `1.40x` on good network and `1.15x` on medium network.
- Tree beats sync by `1.07x-1.09x`.
- Tree beats sync on `40 / 40` paired prompt/network cases.
- The selected branch distribution matches the bimodal acceptance distribution.
- The previous partial-prefetch failure mode is gone: tree no longer pays extra rounds from shortened base verification drafts.

The next optimization should not be another mean-centered offset adjustment. The acceptance distribution is bimodal, so branch policy should continue to prioritize the full-acceptance case and treat lower offsets as fallback anchors.
