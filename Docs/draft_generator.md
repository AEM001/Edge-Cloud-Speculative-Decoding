# key takeaways
- orginal version:
- the things we send 
    - draft_token_ids.append(42)
    - draft_logprobs.append(-0.1)
    - draft_probs.append(0.9048)
    - max_probs.append(0.708)
    - entropies.append(0.976)
    - top_margins.append(0.533)
- The logprobs are the result after being softmaxed
- for minial speculative decoding, we only need the draft_token_ids and the log_probs
- here is a batched version, sending multiple requests to the vllm(maybe for tree based parallelism)


# compute_confidence_stats
This function computes confidence metrics from the model's output probability distribution, not hidden states or raw logits.

What logprobs_dict contains:

A dictionary mapping each token ID to its log-probability
These are the probabilities the model assigned to every possible next token at a specific position
vLLM returns these when you request logprobs=1 in sampling parameters
What probs is:

Line 43: probs[token_id] = np.exp(logprob) converts log-probabilities to regular probabilities
Log-probabilities are used for numerical stability (avoiding underflow with tiny values like 1e-50)
After conversion, probs is a normalized probability distribution over the vocabulary

## Example

Suppose at a certain position, the model assigns these log-probabilities to 4 tokens:
the logprobs are the result after being softmaxed

```python
logprobs_dict = {
    42: -0.1,   # token 42: logprob = -0.1
    99: -2.3,   # token 99: logprob = -2.3
    156: -1.5,  # token 156: logprob = -1.5
    204: -3.0,  # token 204: logprob = -3.0
}
```

**Step 1: Convert logprobs to probs**
```python
probs = {
    42: np.exp(-0.1) = 0.9048,
    99: np.exp(-2.3) = 0.1003,
    156: np.exp(-1.5) = 0.2231,
    204: np.exp(-3.0) = 0.0498,
}
# Total = 1.278, so normalize:
probs = {
    42: 0.9048 / 1.278 = 0.708,
    99: 0.1003 / 1.278 = 0.078,
    156: 0.2231 / 1.278 = 0.175,
    204: 0.0498 / 1.278 = 0.039,
}
```

**Step 2: Compute metrics**
```python
probs_array = [0.708, 0.078, 0.175, 0.039]

# max_prob = highest probability
max_prob = 0.708  # token 42 is most likely

# entropy = -sum(p * log(p))
entropy = -[0.708*log(0.708) + 0.078*log(0.078) + 0.175*log(0.175) + 0.039*log(0.039)]
entropy = -[-0.345 + -0.205 + -0.293 + -0.133]
entropy = 0.976  # moderate uncertainty

# top_margin = top-1 minus top-2
sorted_probs = [0.708, 0.175, 0.078, 0.039]
top_margin = 0.708 - 0.175 = 0.533  # clear winner
```

**Interpretation:**
- max_prob=0.708: Model is 70.8% confident in token 42
- entropy=0.976: Some uncertainty (max would be log(4)=1.39 for uniform distribution)
- top_margin=0.533: Token 42 is clearly favored over the next best (156)

## What Entropy Means

Entropy measures **uncertainty** in the probability distribution:

- **Low entropy (near 0)**: Model is very confident, probability mass is concentrated on one token
- **High entropy**: Model is uncertain, probability is spread across many tokens
- **Maximum entropy**: For n tokens, max entropy = log(n) (uniform distribution where all tokens equally likely)

Formula: `H = -Σ p_i * log(p_i)`

In the example: entropy=0.976 is moderate - not completely certain but has a clear favorite.

## What Top Margin Means

Top margin is the **gap between the top-1 and top-2 probabilities**:

- **High margin**: Clear winner, less ambiguity in the model's choice
- **Low margin**: Top tokens are close, model is unsure between them
- **Zero margin**: Top two tokens have equal probability

In the example: margin=0.533 means token 42 (70.8%) is much more likely than token 156 (17.5%).

## Connection to Softmax

Yes, this is exactly related to softmax! The pipeline is:

```
Logits (raw scores) → Softmax → Probabilities → Log-probabilities
```

- **Logits**: Raw unnormalized scores from the final layer (e.g., [5.2, 1.8, 3.1, 0.5])
- **Softmax**: Converts logits to probabilities that sum to 1
- **Log-probabilities**: log(prob) for numerical stability

The `logprobs_dict` you receive from vLLM contains the log-probabilities after softmax has been applied. The function converts them back to regular probabilities using `np.exp()`.

**Why use log-probabilities instead of probabilities?**
- Numerical stability: probabilities can be extremely small (1e-50), leading to underflow
- Log space turns multiplication into addition: `log(a*b) = log(a) + log(b)`
- Many operations (like cross-entropy loss) work naturally in log space

# generate_draft_tokens

## What is the request object?

`DraftRequest` contains:
- `verified_prefix`: List of token IDs representing the context that has already been verified (accepted by the verification model)
- `num_draft_tokens`: Number K of draft tokens to generate (default 5)

This is used in speculative decoding: the small draft model generates K candidate tokens based on the verified prefix, then a larger verification model checks which ones to accept.

## What are the things being generated?

The function generates:
1. **Draft tokens**: K token IDs that the draft model predicts will follow the prefix
2. **Per-token statistics**: For each draft token, its logprob, probability, and confidence stats (max_prob, entropy, top_margin)

## The loop explained (lines 90-115)

```python
for i, token_id in enumerate(generated_ids):
```
Iterates through each generated draft token.

```python
if i < len(out_logprobs) and out_logprobs[i] is not None:
    logprobs_dict = out_logprobs[i]
```
- `out_logprobs` is a list where each element is a `logprobs_dict` for one position
- `logprobs_dict[i]` contains log-probabilities for ALL vocabulary tokens at position i
- This is the full distribution, not just the chosen token

## Example with actual numbers

Suppose K=2 and the model generates tokens [42, 156]:

**out_logprobs structure:**
```python
out_logprobs = [
    {  # Position 0 (first generated token)
        42: LogProb(-0.1),   # the token that was chosen
        99: LogProb(-2.3),
        156: LogProb(-1.5),
        204: LogProb(-3.0),
        # ... thousands more tokens
    },
    {  # Position 1 (second generated token)
        42: LogProb(-1.8),
        99: LogProb(-0.5),   # different distribution at position 1
        156: LogProb(-0.2),  # the token that was chosen
        204: LogProb(-2.1),
        # ... thousands more tokens
    }
]
```

**Iteration 0 (i=0, token_id=42):**
```python
logprobs_dict = out_logprobs[0]  # Get distribution for position 0
# logprobs_dict = {42: -0.1, 99: -2.3, 156: -1.5, 204: -3.0, ...}

token_logprob_obj = logprobs_dict.get(42)  # Get logprob for chosen token
# token_logprob_obj = LogProb(-0.1)

token_logprob_val = -0.1
token_prob = np.exp(-0.1) = 0.9048

# Compute confidence stats using FULL distribution
max_prob, entropy, top_margin = compute_confidence_stats(logprobs_dict)
# max_prob = 0.708, entropy = 0.976, top_margin = 0.533

# Append to lists
draft_token_ids.append(42)
draft_logprobs.append(-0.1)
draft_probs.append(0.9048)
max_probs.append(0.708)
entropies.append(0.976)
top_margins.append(0.533)
```

**Iteration 1 (i=1, token_id=156):**
```python
logprobs_dict = out_logprobs[1]  # Get distribution for position 1
# logprobs_dict = {42: -1.8, 99: -0.5, 156: -0.2, 204: -2.1, ...}

token_logprob_obj = logprobs_dict.get(156)  # Get logprob for chosen token
# token_logprob_obj = LogProb(-0.2)

token_logprob_val = -0.2
token_prob = np.exp(-0.2) = 0.8187

# Compute confidence stats using FULL distribution (different from position 0)
max_prob, entropy, top_margin = compute_confidence_stats(logprobs_dict)
# Suppose: max_prob = 0.650, entropy = 1.1, top_margin = 0.400

# Append to lists
draft_token_ids.append(156)
draft_logprobs.append(-0.2)
draft_probs.append(0.8187)
max_probs.append(0.650)
entropies.append(1.1)
top_margins.append(0.400)
```

**Final result:**
```python
draft_token_ids = [42, 156]
draft_logprobs = [-0.1, -0.2]
draft_probs = [0.9048, 0.8187]
max_probs = [0.708, 0.650]
entropies = [0.976, 1.1]
top_margins = [0.533, 0.400]
```

Each position has its own probability distribution, so the confidence stats can differ even if the same token is chosen.

# generate_draft_tokens_batch

## What's the difference between batch and single?

**generate_draft_tokens (single):**
- Input: One `DraftRequest`
- Output: One `DraftResponse`
- Use case: Generate draft tokens for a single prefix

**generate_draft_tokens_batch:**
- Input: List of `DraftRequest` (multiple requests)
- Output: List of `DraftResponse` (one per request)
- Use case: Generate draft tokens for multiple prefixes efficiently

## Key difference: Single vLLM call vs multiple

The batch version processes multiple requests in **one vLLM call**:

```python
# Single version (one request)
outputs = self.llm.generate(
    prompts=[TokensPrompt(prompt_token_ids=list(prefix))],
    sampling_params=sampling_params,
)

# Batch version (multiple requests)
outputs = self.llm.generate(
    prompts=[TokensPrompt(prompt_token_ids=list(req.verified_prefix)) for req in requests],
    sampling_params=sampling_params,
)
```

vLLM can process multiple prompts in parallel efficiently using batching, which is faster than making separate calls for each request.

## Processing logic

After vLLM returns, the batch version iterates through each output:

```python
for output, req in zip(outputs, requests):
    # Same processing as single version:
    # - Extract generated_ids
    # - Extract out_logprobs
    # - Loop through tokens and compute confidence stats
    # - Build DraftResponse
    responses.append(DraftResponse(...))
```

The per-token processing (lines 165-188) is identical to the single version - it's just wrapped in an outer loop to handle multiple requests.

## Example

If you have 3 requests with different prefixes:
- Single version: Call `generate_draft_tokens` 3 times (3 vLLM calls)
- Batch version: Call `generate_draft_tokens_batch` once with list of 3 requests (1 vLLM call)

The batch version is more efficient when processing multiple independent requests.