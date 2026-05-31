#!/usr/bin/env python3
"""Comprehensive test for verify server with 2048 input and 512 output tokens."""

import requests
import json
import time
import sys
from typing import Dict, List

SERVER_URL = "http://127.0.0.1:6007"

def flush_print(msg: str, end: str = '\n'):
    """Print and immediately flush to stdout."""
    print(msg, end=end, flush=True)

import threading
import itertools

def show_progress_indicator(duration_seconds: int = 120):
    """Show a spinning progress indicator in a separate thread."""
    spinner = itertools.cycle(['-', '/', '|', '\\'])
    start_time = time.time()
    
    def _spin():
        while time.time() - start_time < duration_seconds:
            flush_print(f"\r  Working... {next(spinner)}  ", end='')
            time.sleep(0.1)
        flush_print("\r" + " " * 20 + "\r", end='')
    
    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    return thread

def generate_long_prompt(num_tokens: int = 2048) -> str:
    """Generate a long prompt by repeating text."""
    base_text = "The quick brown fox jumps over the lazy dog. "
    repeats = (num_tokens // 10) + 1
    return (base_text * repeats)[:num_tokens * 10]

def test_health() -> bool:
    """Test the health endpoint."""
    flush_print("=" * 70)
    flush_print("HEALTH CHECK")
    flush_print("=" * 70)
    flush_print("Checking server health...")
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            flush_print(f"✓ Status: {data.get('status')}")
            flush_print(f"  Model: {data.get('model_info')}")
            flush_print(f"  Runtime: {data.get('runtime')}")
            return True
        else:
            flush_print(f"✗ Failed with status {response.status_code}")
            return False
    except Exception as e:
        flush_print(f"✗ Error: {e}")
        return False

def test_generate_long_context(input_tokens: int = 2048, output_tokens: int = 512) -> Dict:
    """Test generation with long context."""
    flush_print("\n" + "=" * 70)
    flush_print(f"GENERATION TEST: {input_tokens} input tokens -> {output_tokens} output tokens")
    flush_print("=" * 70)
    
    flush_print(f"Generating prompt with ~{input_tokens} tokens...")
    prompt = generate_long_prompt(input_tokens)
    flush_print(f"Prompt ready, sending request...")
    
    try:
        start_time = time.time()
        request_data = {
            "prompt": prompt,
            "max_tokens": output_tokens,
            "temperature": 0.0
        }
        flush_print(f"Generating {output_tokens} tokens (this may take a while)...")
        progress_thread = show_progress_indicator(120)
        response = requests.post(f"{SERVER_URL}/generate", json=request_data, timeout=120)
        end_time = time.time()
        flush_print("\r" + " " * 30 + "\r", end='')  # Clear spinner
        
        if response.status_code == 200:
            data = response.json()
            total_time_ms = (end_time - start_time) * 1000
            
            flush_print(f"✓ Generation successful")
            flush_print(f"  Generated text length: {len(data.get('text', ''))} chars")
            flush_print(f"  Tokens generated: {data.get('tokens_generated')}")
            flush_print(f"  Server generation time: {data.get('generation_time_ms'):.2f}ms")
            flush_print(f"  End-to-end time: {total_time_ms:.2f}ms")
            flush_print(f"  Network overhead: {total_time_ms - data.get('generation_time_ms', 0):.2f}ms")
            
            if data.get('tokens_generated') > 0:
                throughput = data.get('tokens_generated') / (data.get('generation_time_ms', 1) / 1000)
                flush_print(f"  Throughput: {throughput:.2f} tokens/second")
            
            return {
                "success": True,
                "tokens_generated": data.get('tokens_generated'),
                "generation_time_ms": data.get('generation_time_ms'),
                "total_time_ms": total_time_ms,
                "network_overhead_ms": total_time_ms - data.get('generation_time_ms', 0)
            }
        else:
            flush_print(f"✗ Failed with status {response.status_code}")
            flush_print(f"  Response: {response.text}")
            return {"success": False}
    except Exception as e:
        flush_print(f"✗ Error: {e}")
        return {"success": False}

def test_verify_large_draft(prefix_tokens: int = 2048, draft_tokens: int = 512) -> Dict:
    """Test verification with large draft."""
    flush_print("\n" + "=" * 70)
    flush_print(f"VERIFICATION TEST: {prefix_tokens} prefix + {draft_tokens} draft tokens")
    flush_print("=" * 70)
    
    flush_print(f"Preparing {prefix_tokens} prefix tokens and {draft_tokens} draft tokens...")
    # Create dummy token IDs (in real scenario, these would be actual tokens)
    prefix_ids = list(range(1000, 1000 + prefix_tokens))
    draft_ids = list(range(2000, 2000 + draft_tokens))
    flush_print(f"Tokens ready, sending verification request...")
    
    try:
        start_time = time.time()
        request_data = {
            "request_id": f"test_large_{int(time.time())}",
            "prefix_ids": prefix_ids,
            "draft_ids": draft_ids
        }
        flush_print(f"Verifying {draft_tokens} draft tokens (this may take a while)...")
        progress_thread = show_progress_indicator(120)
        response = requests.post(f"{SERVER_URL}/verify", json=request_data, timeout=120)
        end_time = time.time()
        flush_print("\r" + " " * 30 + "\r", end='')  # Clear spinner
        
        if response.status_code == 200:
            data = response.json()
            total_time_ms = (end_time - start_time) * 1000
            
            flush_print(f"✓ Verification successful")
            flush_print(f"  Request ID: {data.get('request_id')}")
            flush_print(f"  Accepted length: {data.get('accepted_len')}")
            flush_print(f"  Correction token ID: {data.get('correction_token_id')}")
            flush_print(f"  Server verify time: {data.get('server_verify_time_ms'):.2f}ms")
            flush_print(f"  Model time: {data.get('model_time_ms'):.2f}ms")
            flush_print(f"  HTTP overhead: {data.get('http_overhead_ms'):.2f}ms")
            flush_print(f"  End-to-end time: {total_time_ms:.2f}ms")
            flush_print(f"  Network overhead: {total_time_ms - data.get('server_verify_time_ms', 0):.2f}ms")
            
            if data.get('server_verify_time_ms') > 0:
                throughput = draft_tokens / (data.get('server_verify_time_ms', 1) / 1000)
                flush_print(f"  Verification throughput: {throughput:.2f} tokens/second")
            
            acceptance_rate = data.get('accepted_len', 0) / draft_tokens * 100 if draft_tokens > 0 else 0
            flush_print(f"  Acceptance rate: {acceptance_rate:.2f}%")
            
            return {
                "success": True,
                "accepted_len": data.get('accepted_len'),
                "correction_token_id": data.get('correction_token_id'),
                "server_verify_time_ms": data.get('server_verify_time_ms'),
                "model_time_ms": data.get('model_time_ms'),
                "http_overhead_ms": data.get('http_overhead_ms'),
                "total_time_ms": total_time_ms,
                "network_overhead_ms": total_time_ms - data.get('server_verify_time_ms', 0),
                "acceptance_rate": acceptance_rate
            }
        else:
            flush_print(f"✗ Failed with status {response.status_code}")
            flush_print(f"  Response: {response.text}")
            return {"success": False}
    except Exception as e:
        flush_print(f"✗ Error: {e}")
        return {"success": False}

def test_specextend_verify_large_tree(prefix_tokens: int = 2048, tree_nodes: int = 512) -> Dict:
    """Test SpecExtend tree verification with large tree."""
    flush_print("\n" + "=" * 70)
    flush_print(f"SPECEXTEND TREE VERIFICATION: {prefix_tokens} prefix + {tree_nodes} tree nodes")
    flush_print("=" * 70)
    
    flush_print(f"Preparing tree structure with {tree_nodes} nodes...")
    # Create a simple tree structure
    prefix_ids = list(range(1000, 1000 + prefix_tokens))
    tree_input_ids = list(range(2000, 2000 + tree_nodes))
    tree_position_ids = list(range(prefix_tokens, prefix_tokens + tree_nodes))
    
    # Simple parent indices (linear chain for simplicity)
    parent_indices = [i - 1 if i > 0 else -1 for i in range(tree_nodes)]
    
    # Simple attention mask (lower triangular)
    tree_attention_mask = [
        [1 if col <= row else 0 for col in range(tree_nodes)]
        for row in range(tree_nodes)
    ]
    flush_print(f"Tree structure ready, sending verification request...")
    
    try:
        start_time = time.time()
        request_data = {
            "request_id": f"test_tree_{int(time.time())}",
            "prefix_ids": prefix_ids,
            "tree_input_ids": tree_input_ids,
            "tree_position_ids": tree_position_ids,
            "parent_indices": parent_indices,
            "tree_attention_mask": tree_attention_mask,
            "retrieve_attn_scores": False
        }
        flush_print(f"Verifying tree with {tree_nodes} nodes (this may take a while)...")
        progress_thread = show_progress_indicator(120)
        response = requests.post(f"{SERVER_URL}/specextend/verify", json=request_data, timeout=120)
        end_time = time.time()
        flush_print("\r" + " " * 30 + "\r", end='')  # Clear spinner
        
        if response.status_code == 200:
            data = response.json()
            total_time_ms = (end_time - start_time) * 1000
            
            flush_print(f"✓ Tree verification successful")
            flush_print(f"  Request ID: {data.get('request_id')}")
            flush_print(f"  Accepted length: {data.get('accepted_len')}")
            flush_print(f"  Correction token ID: {data.get('correction_token_id')}")
            flush_print(f"  Accepted tree indices: {len(data.get('accepted_tree_indices', []))} nodes")
            flush_print(f"  Server verify time: {data.get('server_verify_time_ms'):.2f}ms")
            flush_print(f"  Model time: {data.get('model_time_ms'):.2f}ms")
            flush_print(f"  HTTP overhead: {data.get('http_overhead_ms'):.2f}ms")
            flush_print(f"  End-to-end time: {total_time_ms:.2f}ms")
            flush_print(f"  Network overhead: {total_time_ms - data.get('server_verify_time_ms', 0):.2f}ms")
            
            if data.get('server_verify_time_ms') > 0:
                throughput = tree_nodes / (data.get('server_verify_time_ms', 1) / 1000)
                flush_print(f"  Tree verification throughput: {throughput:.2f} nodes/second")
            
            acceptance_rate = data.get('accepted_len', 0) / tree_nodes * 100 if tree_nodes > 0 else 0
            flush_print(f"  Acceptance rate: {acceptance_rate:.2f}%")
            
            return {
                "success": True,
                "accepted_len": data.get('accepted_len'),
                "correction_token_id": data.get('correction_token_id'),
                "accepted_tree_indices_count": len(data.get('accepted_tree_indices', [])),
                "server_verify_time_ms": data.get('server_verify_time_ms'),
                "model_time_ms": data.get('model_time_ms'),
                "http_overhead_ms": data.get('http_overhead_ms'),
                "total_time_ms": total_time_ms,
                "network_overhead_ms": total_time_ms - data.get('server_verify_time_ms', 0),
                "acceptance_rate": acceptance_rate
            }
        else:
            flush_print(f"✗ Failed with status {response.status_code}")
            flush_print(f"  Response: {response.text}")
            return {"success": False}
    except Exception as e:
        flush_print(f"✗ Error: {e}")
        return {"success": False}

def print_summary(results: Dict):
    """Print summary of all tests."""
    flush_print("\n" + "=" * 70)
    flush_print("PERFORMANCE SUMMARY")
    flush_print("=" * 70)
    
    if results.get('health'):
        flush_print("✓ Health check: PASSED")
    else:
        flush_print("✗ Health check: FAILED")
    
    gen = results.get('generation', {})
    if gen.get('success'):
        flush_print(f"✓ Generation ({gen.get('tokens_generated')} tokens): {gen.get('generation_time_ms', 0):.2f}ms")
        flush_print(f"  - Network overhead: {gen.get('network_overhead_ms', 0):.2f}ms")
    else:
        flush_print("✗ Generation: FAILED")
    
    verify = results.get('verify', {})
    if verify.get('success'):
        flush_print(f"✓ Verification: {verify.get('server_verify_time_ms', 0):.2f}ms")
        flush_print(f"  - Model time: {verify.get('model_time_ms', 0):.2f}ms")
        flush_print(f"  - HTTP overhead: {verify.get('http_overhead_ms', 0):.2f}ms")
        flush_print(f"  - Network overhead: {verify.get('network_overhead_ms', 0):.2f}ms")
        flush_print(f"  - Acceptance rate: {verify.get('acceptance_rate', 0):.2f}%")
    else:
        flush_print("✗ Verification: FAILED")
    
    tree = results.get('tree_verify', {})
    if tree.get('success'):
        flush_print(f"✓ Tree verification: {tree.get('server_verify_time_ms', 0):.2f}ms")
        flush_print(f"  - Model time: {tree.get('model_time_ms', 0):.2f}ms")
        flush_print(f"  - HTTP overhead: {tree.get('http_overhead_ms', 0):.2f}ms")
        flush_print(f"  - Network overhead: {tree.get('network_overhead_ms', 0):.2f}ms")
        flush_print(f"  - Acceptance rate: {tree.get('acceptance_rate', 0):.2f}%")
    else:
        flush_print("✗ Tree verification: FAILED")
    
    flush_print("=" * 70)

def main():
    flush_print(f"Comprehensive verify server test at {SERVER_URL}")
    flush_print(f"Testing with 2048 input tokens and 512 output tokens\n")
    
    results = {}
    
    # Run all tests
    results['health'] = test_health()
    results['generation'] = test_generate_long_context(2048, 512)
    results['verify'] = test_verify_large_draft(2048, 8)
    results['tree_verify'] = test_specextend_verify_large_tree(2048, 8)
    
    # Print summary
    print_summary(results)
    
    # Return exit code
    all_passed = all([
        results.get('health', False),
        results.get('generation', {}).get('success', False),
        results.get('verify', {}).get('success', False),
        results.get('tree_verify', {}).get('success', False)
    ])
    
    if all_passed:
        flush_print("\n✓ All comprehensive tests passed!")
        return 0
    else:
        flush_print("\n✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
