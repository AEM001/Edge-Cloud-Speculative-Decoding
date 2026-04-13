"""Test 2-bit model speed vs 4-bit."""
import time
from mlx_lm import load, generate

print("=== Comparing 4-bit vs 2-bit ===\n")

# Test prompt
test_prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

# Test 4-bit model
print("1. Loading 4-bit model...")
model_4bit, tokenizer_4bit = load('/Users/Mac/.models/mlx-community/Qwen2.5-0.5B-Instruct-4bit')
print("   Testing 4-bit (K=2)...")
times_4bit = []
for i in range(3):
    start = time.time()
    result = generate(model_4bit, tokenizer_4bit, prompt=test_prompt, max_tokens=2, verbose=False)
    elapsed = (time.time() - start) * 1000
    times_4bit.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.0f}ms")
avg_4bit = sum(times_4bit) / len(times_4bit)
print(f"   Average: {avg_4bit:.0f}ms\n")

# Test 2-bit model
print("2. Loading 2-bit model...")
model_2bit, tokenizer_2bit = load('/Users/Mac/.models/mlx-community/Qwen2.5-0.5B-Instruct-2bit')
print("   Testing 2-bit (K=2)...")
times_2bit = []
for i in range(3):
    start = time.time()
    result = generate(model_2bit, tokenizer_2bit, prompt=test_prompt, max_tokens=2, verbose=False)
    elapsed = (time.time() - start) * 1000
    times_2bit.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.0f}ms")
avg_2bit = sum(times_2bit) / len(times_2bit)
print(f"   Average: {avg_2bit:.0f}ms\n")

# Summary
print("=== Summary ===")
print(f"4-bit: {avg_4bit:.0f}ms")
print(f"2-bit: {avg_2bit:.0f}ms")
speedup = avg_4bit / avg_2bit
print(f"Speedup: {speedup:.2f}x")

if avg_2bit < 100:
    print(f"\n✅ 2-bit is fast enough! ({avg_2bit:.0f}ms < 100ms)")
    print("   Estimated time for 128 tokens: ~15s")
    print("   This could work for speculative decoding!")
else:
    print(f"\n⚠️  2-bit still too slow ({avg_2bit:.0f}ms)")
    print(f"   Estimated time for 128 tokens: ~{avg_2bit * 107 / 1000:.0f}s")
