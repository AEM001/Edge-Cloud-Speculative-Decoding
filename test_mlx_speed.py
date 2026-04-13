"""Test MLX-LM generation speed directly."""
import time
from mlx_lm import load, generate

print("Loading model...")
model, tokenizer = load('/Users/Mac/.models/mlx-community/Qwen2.5-0.5B-Instruct-4bit')

test_prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

print(f"Prompt: {test_prompt!r}\n")

# Test 1: Generate 2 tokens
print("Test 1: Generate 2 tokens")
for i in range(3):
    start = time.time()
    result = generate(model, tokenizer, prompt=test_prompt, max_tokens=2, verbose=False)
    elapsed = (time.time() - start) * 1000
    print(f"  Run {i+1}: {elapsed:.0f}ms - {result[len(test_prompt):]!r}")

# Test 2: Generate 6 tokens
print("\nTest 2: Generate 6 tokens")
for i in range(3):
    start = time.time()
    result = generate(model, tokenizer, prompt=test_prompt, max_tokens=6, verbose=False)
    elapsed = (time.time() - start) * 1000
    print(f"  Run {i+1}: {elapsed:.0f}ms - {result[len(test_prompt):]!r}")

# Test 3: Longer prompt
long_prompt = test_prompt + "I am doing well, thank you for asking. " * 20
print(f"\nTest 3: Long prompt ({len(tokenizer.encode(long_prompt))} tokens), generate 2 tokens")
for i in range(3):
    start = time.time()
    result = generate(model, tokenizer, prompt=long_prompt, max_tokens=2, verbose=False)
    elapsed = (time.time() - start) * 1000
    print(f"  Run {i+1}: {elapsed:.0f}ms")
