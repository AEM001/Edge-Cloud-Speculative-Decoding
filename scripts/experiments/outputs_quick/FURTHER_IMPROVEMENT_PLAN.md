# Further Improvement Plan: Bounded Adaptive K

## Result From Latest Run

The first adaptive policy reduced wasted draft tokens, but it hurt throughput.
It raised average rounds from about `23.9` to `31.6`, so verifier and RTT cost
grew more than draft cost shrank.

The policy also had a state-trap bug: thresholds were calibrated for fixed
`K=7`, but after switching to `K=3` or `K=4`, the policy could not reach
`net>=6`, so it stayed in small-K mode.

## Corrected Policy

- First round: use `K=7`.
- Previous `K < 7`: return to `K=7` as a probe.
- Previous net output `>= 6`: use `K=7`.
- Previous net output `3-5`: use one `K=4` recovery round.
- Previous net output `1-2`: use one `K=3` recovery round.
- Clamp K to the remaining generation budget.

## Why This Is Safer

Reduced K is now a bounded recovery action, not a new steady state. This keeps
the main benefit of avoiding the worst waste immediately after rejection while
preventing extra verify/RTT rounds from dominating end-to-end latency.

## Next Measurement

Re-run `quick_test.py` and compare `sync_adaptive_k3_4_7` against `sync_k7` on:

- tokens/sec
- rounds per prompt
- mean K chosen
- wasted draft tokens
- server model time
- simulated network time

The corrected policy should only be kept if it reduces waste without materially
increasing rounds.
