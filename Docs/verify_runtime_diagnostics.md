# Verify Runtime Diagnostics

This note explains the instrumentation added to expose why verification can
become slow on long prefixes.

## Problem

The verifier currently checks draft tokens by sending the full token sequence
to vLLM:

```text
prefix_ids + draft_ids
```

It requests:

```python
SamplingParams(
    temperature=0.0,
    max_tokens=1,
    logprobs=1,
    prompt_logprobs=len(draft_ids),
)
```

In the ideal case, vLLM prefix caching should reuse the KV cache for the
already-seen `prefix_ids`, so each verify round mostly computes only the new
draft tokens plus one correction token.

If prefix caching does not actually help this `prompt_logprobs` path, each
verify round may recompute most or all of the long prefix. Then verify time
grows with `prefix_len`, and speculative decoding becomes slow on long prompts.

## Runtime Fields

The `/health` endpoint now reports the verifier runtime configuration:

- `vllm_version`
- `enable_prefix_caching`
- `enforce_eager`
- `attention_backend`
- `max_model_len`
- `tensor_parallel_size`
- `quantization`
- `model_path`

Each `/verify` and `/verify_batch` response now also reports:

- `prefix_len`: number of prefix tokens sent to verifier
- `draft_len`: number of draft tokens being verified
- `input_len`: `prefix_len + draft_len`
- `prompt_logprobs_requested`: value passed to `SamplingParams(prompt_logprobs=...)`
- `max_tokens_requested`: value passed to `SamplingParams(max_tokens=...)`
- `verify_batch_size`: number of verify requests in the vLLM call
- `enable_prefix_caching`
- `enforce_eager`
- `attention_backend`
- `vllm_version`

The edge clients preserve these fields in per-round metrics, and `quick_test.py`
stores aggregate values in the JSON output:

- `avg_verify_prefix_len`
- `avg_verify_input_len`
- `verify_prefix_caching`
- `verify_enforce_eager`
- `verify_attention_backend`
- `verify_vllm_version`

## How To Read The Result

Look at these values together:

```text
avg_verify_prefix_len
avg_verify_input_len
verify_time_ms / num_rounds
verify_prefix_caching
verify_enforce_eager
verify_attention_backend
verify_vllm_version
```

If `verify_prefix_caching=True`, but per-round verify time still grows strongly
as `prefix_len` grows, then prefix caching is probably not effective for the
current verification path.

That means the server is likely paying something close to:

```text
prefill(prefix_len + draft_len) + 1 decode
```

instead of the desired:

```text
prefill(new draft tokens only) + 1 decode
```

## Why This Matters

Direct generation pays the long-prefix prefill cost once, then decodes many
tokens in the same request.

Speculative verification sends many verify requests. If each request recomputes
the long prefix, the system repeatedly pays the long-prefix prefill cost.

In that case, tree prefetch can hide only local branch draft time. It cannot fix
the dominant server-side cost from repeated long-prefix verification.

## Practical Diagnosis

A healthy prefix-cache pattern should look roughly like this:

```text
prefix_len increases over rounds
server_verify_time_ms stays mostly flat or grows slowly
```

An unhealthy pattern looks like this:

```text
prefix_len increases over rounds
server_verify_time_ms increases with prefix_len
```

The second pattern is the runtime problem this instrumentation is meant to
expose clearly.
