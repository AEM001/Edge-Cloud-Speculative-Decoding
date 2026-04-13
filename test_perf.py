"""Quick performance test to diagnose slowness."""
import time
import requests
from mlx_lm import load
from client.draft_generator import DraftGenerator
from protocol import DraftRequest

print("=== Performance Diagnosis ===\n")

# 1. Network RTT
print("1. Testing network RTT...")
start = time.time()
resp = requests.get('http://49.234.57.210:8005/health', timeout=10)
rtt = (time.time() - start) * 1000
print(f"   Health check RTT: {rtt:.0f}ms\n")

# 2. Load models
print("2. Loading draft model...")
model, tokenizer = load('/Users/Mac/.models/mlx-community/Qwen2.5-0.5B-Instruct-4bit')
draft_gen = DraftGenerator(model, tokenizer)
print("   Draft model loaded\n")

# 3. Test prompt
test_prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n"
prompt_ids = tokenizer.encode(test_prompt)
print(f"3. Test prompt: {len(prompt_ids)} tokens\n")

# 4. Draft generation speed (K=2)
print("4. Testing draft generation (K=2)...")
draft_req = DraftRequest(verified_prefix=prompt_ids, num_draft_tokens=2)
times = []
for i in range(3):
    start = time.time()
    draft_resp = draft_gen.generate_draft_tokens(draft_req, temperature=0.8)
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.0f}ms")
avg_draft = sum(times) / len(times)
print(f"   Average: {avg_draft:.0f}ms\n")

# 5. Server verify speed
print("5. Testing server verify (K=2)...")
payload = {
    'request_id': 'test',
    'round_id': 0,
    'prefix_ids': prompt_ids,
    'draft_ids': draft_resp.draft_token_ids,
    'draft_logprobs': draft_resp.logprobs,
    'edge_draft_time_ms': avg_draft,
    'policy_metadata': {'K': 2}
}

times = []
for i in range(3):
    start = time.time()
    resp = requests.post('http://49.234.57.210:8005/verify', json=payload, timeout=60)
    elapsed = (time.time() - start) * 1000
    data = resp.json()
    times.append(elapsed)
    print(f"   Run {i+1}: total={elapsed:.0f}ms, server={data['server_verify_time_ms']:.0f}ms, accepted={data['accepted_len']}")
avg_verify = sum(times) / len(times)
print(f"   Average total: {avg_verify:.0f}ms\n")

# 6. Estimate time for 128 tokens
print("6. Estimating time for 128 tokens...")
# With K=2, acceptance rate ~10%, we need ~128 / (2*0.1 + 1) = ~107 rounds
# Each round: draft + verify = avg_draft + avg_verify
rounds_needed = 128 / (2 * 0.1 + 1)  # Assume 10% acceptance
time_per_round = avg_draft + avg_verify
estimated_time = rounds_needed * time_per_round / 1000
print(f"   Rounds needed (10% acceptance): {rounds_needed:.0f}")
print(f"   Time per round: {time_per_round:.0f}ms")
print(f"   Estimated total: {estimated_time:.0f}s ({estimated_time/60:.1f}min)\n")

# 7. Compare to direct generation baseline
print("7. Baseline: What if we just called server 128 times?")
baseline_time = 128 * avg_verify / 1000
print(f"   128 sequential verify calls: {baseline_time:.0f}s ({baseline_time/60:.1f}min)")
print(f"   Speedup: {baseline_time / estimated_time:.2f}x\n")

print("=== Summary ===")
print(f"Draft generation: {avg_draft:.0f}ms")
print(f"Server verify: {avg_verify:.0f}ms (server={data['server_verify_time_ms']:.0f}ms, network={avg_verify - data['server_verify_time_ms']:.0f}ms)")
print(f"Per-round total: {time_per_round:.0f}ms")
print(f"Expected time for 128 tokens: {estimated_time:.0f}s")

if avg_verify > 500:
    print("\n⚠️  WARNING: Server verify time is very high (>500ms)")
    print("   Possible causes:")
    print("   - Network latency/proxy overhead")
    print("   - Server is slow (vLLM inference time)")
    print("   - Server is overloaded")
