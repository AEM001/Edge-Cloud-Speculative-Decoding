"""
Comprehensive Benchmark: Easy vs Hard Prompts with K=2, K=4, K=6, and Direct
Tests: 50 easy prompts + 50 hard prompts × 4 methods = 400 tests total
Output: comprehensive_results.json, comprehensive_log.txt
"""
import logging
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import requests
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_NAME, MODEL_PATH, TEMPERATURE, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN
from model_manager import VLLMModelManager
from draft_generator import VLLMDraftGenerator
from client.edge_client import EdgeClient
from client.http_cloud_client import create_http_cloud_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    method: str  # 'direct', 'spec_k2', 'spec_k4', 'spec_k6'
    prompt_type: str  # 'easy' or 'hard'
    prompt_id: int
    prompt_length: int
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float
    # Speculative specific
    num_rounds: int = 0
    acceptance_rate: float = 0.0
    avg_draft_ms: float = 0.0
    avg_verify_ms: float = 0.0

def load_prompts_from_file(filepath: Path, count: int = 50) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse prompts.txt and select count easy and count hard prompts.
    Selects prompts with similar lengths from the middle range.
    """
    easy_prompts = []
    hard_prompts = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse EASY prompts
    import re
    easy_section = re.search(r'EASY BENCHMARKS.*?HARD BENCHMARKS', content, re.DOTALL)
    if easy_section:
        easy_matches = re.findall(r'\[(\d+)\]\s+benchmark_[\d\-]+.*?Actual length:\s*(\d+)\s*chars.*?User:\s*(.*?)(?=\n\n\[|$)', 
                                  easy_section.group(), re.DOTALL)
        for idx, (id_str, length, text) in enumerate(easy_matches[:count*2], 1):  # Get more for filtering
            text = text.strip()[:400]  # Limit length for display
            easy_prompts.append({"id": idx, "length": int(length), "text": text})
    
    # Parse HARD prompts
    hard_section = re.search(r'HARD BENCHMARKS.*?(?=$)', content, re.DOTALL)
    if hard_section:
        hard_matches = re.findall(r'\[(\d+)\]\s+benchmark_[\d\-]+.*?Actual length:\s*(\d+)\s*chars.*?User:\s*(.*?)(?=\n\n\[|$)', 
                                  hard_section.group(), re.DOTALL)
        for idx, (id_str, length, text) in enumerate(hard_matches[:count*2], 1):
            text = text.strip()[:400]
            hard_prompts.append({"id": idx, "length": int(length), "text": text})
    
    # Filter to get count prompts with similar lengths (within target range)
    def select_balanced(prompts: List[Dict], target_count: int, target_len: int = 300) -> List[Dict]:
        """Select prompts closest to target length"""
        sorted_prompts = sorted(prompts, key=lambda x: abs(x['length'] - target_len))
        selected = sorted_prompts[:target_count]
        return sorted(selected, key=lambda x: x['id'])  # Re-sort by id
    
    easy_selected = select_balanced(easy_prompts, count, target_len=300)
    hard_selected = select_balanced(hard_prompts, count, target_len=500)
    
    # Reassign sequential IDs
    for i, p in enumerate(easy_selected, 1):
        p['id'] = i
    for i, p in enumerate(hard_selected, 1):
        p['id'] = i
    
    logger.info(f"Loaded {len(easy_selected)} easy prompts (length range: {min(p['length'] for p in easy_selected)}-{max(p['length'] for p in easy_selected)})")
    logger.info(f"Loaded {len(hard_selected)} hard prompts (length range: {min(p['length'] for p in hard_selected)}-{max(p['length'] for p in hard_selected)})")
    
    return easy_selected, hard_selected


# Load prompts dynamically
PROMPTS_FILE = Path(__file__).parent.parent / 'benchmarks' / 'prompts.txt'
EASY_PROMPTS, HARD_PROMPTS = load_prompts_from_file(PROMPTS_FILE, count=50)

def test_direct(server_url: str, prompt: str, max_tokens: int = 128) -> Tuple[int, float]:
    """Test direct generation, returns (tokens, time_ms)"""
    start = time.time()
    try:
        response = requests.post(
            f"{server_url}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": TEMPERATURE},
            timeout=120.0
        )
        response.raise_for_status()
        result = response.json()
        total_ms = (time.time() - start) * 1000
        return result['tokens_generated'], total_ms
    except Exception as e:
        logger.error(f"Direct failed: {e}")
        return 0, 0

def test_speculative(client: EdgeClient, prompt: str, k: int) -> Tuple[int, float, int, float, float, float]:
    """Test speculative with given K, returns (tokens, time_ms, rounds, acceptance, draft_ms, verify_ms)"""
    start = time.time()
    try:
        def policy(round_id, draft_tokens):
            return k
        
        metrics = client.generate(prompt=prompt, policy=policy, policy_name=f"StaticK{k}")
        total_ms = (time.time() - start) * 1000
        
        avg_draft = metrics.total_edge_draft_time_ms / metrics.total_rounds if metrics.total_rounds > 0 else 0
        avg_verify = metrics.total_server_verify_time_ms / metrics.total_rounds if metrics.total_rounds > 0 else 0
        
        return (
            metrics.generated_tokens,
            total_ms,
            metrics.total_rounds,
            metrics.acceptance_ratio,
            avg_draft,
            avg_verify
        )
    except Exception as e:
        logger.error(f"Speculative K={k} failed: {e}")
        return 0, 0, 0, 0, 0, 0

def run_experiment():
    server_url = "http://localhost:6006"
    results = []
    
    logger.info("="*80)
    logger.info("Comprehensive Benchmark: Easy vs Hard Prompts")
    logger.info("Methods: Direct, K=2, K=4, K=6")
    logger.info(f"Total tests: {len(EASY_PROMPTS) + len(HARD_PROMPTS)} prompts × 4 methods = {(len(EASY_PROMPTS) + len(HARD_PROMPTS)) * 4} tests")
    logger.info(f"Easy: {len(EASY_PROMPTS)} prompts, Hard: {len(HARD_PROMPTS)} prompts")
    logger.info("="*80)
    
    # Initialize speculative client once
    logger.info("\n[Setup] Loading draft model...")
    model_manager = VLLMModelManager(MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    cloud_client = create_http_cloud_client(server_url, timeout=120.0)
    edge_client = EdgeClient(model_manager, draft_generator, cloud_client, 
                              max_new_tokens=128, temperature=TEMPERATURE)
    logger.info("[Setup] Ready!\n")
    
    # Test all combinations
    test_count = 0
    total_tests = (len(EASY_PROMPTS) + len(HARD_PROMPTS)) * 3
    
    for prompt_type, prompts in [("easy", EASY_PROMPTS), ("hard", HARD_PROMPTS)]:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {prompt_type.upper()} PROMPTS ({len(prompts)} tests × 3 methods)")
        logger.info(f"{'='*80}")
        
        for prompt_data in prompts:
            prompt_id = prompt_data["id"]
            prompt_text = prompt_data["text"]
            prompt_len = prompt_data["length"]
            
            logger.info(f"\n[{prompt_type.upper()} Prompt {prompt_id}] Length: {prompt_len} chars")
            
            # 1. Test Direct
            test_count += 1
            logger.info(f"  [{test_count}/{total_tests}] Direct...")
            tokens, time_ms = test_direct(server_url, prompt_text)
            if tokens > 0:
                results.append(TestResult(
                    method='direct', prompt_type=prompt_type, prompt_id=prompt_id,
                    prompt_length=prompt_len, tokens_generated=tokens,
                    total_time_ms=time_ms, tokens_per_second=tokens/(time_ms/1000)
                ))
                logger.info(f"      ✓ {tokens} tokens, {time_ms:.0f}ms, {tokens/(time_ms/1000):.2f} tok/s")
            time.sleep(0.5)
            
            # 2. Test K=2
            test_count += 1
            logger.info(f"  [{test_count}/{total_tests}] Speculative K=2...")
            tokens, time_ms, rounds, acc, draft_ms, verify_ms = test_speculative(edge_client, prompt_text, k=2)
            if tokens > 0:
                results.append(TestResult(
                    method='spec_k2', prompt_type=prompt_type, prompt_id=prompt_id,
                    prompt_length=prompt_len, tokens_generated=tokens,
                    total_time_ms=time_ms, tokens_per_second=tokens/(time_ms/1000),
                    num_rounds=rounds, acceptance_rate=acc,
                    avg_draft_ms=draft_ms, avg_verify_ms=verify_ms
                ))
                logger.info(f"      ✓ {tokens} tokens, {time_ms:.0f}ms, {tokens/(time_ms/1000):.2f} tok/s")
                logger.info(f"        Rounds: {rounds}, Acceptance: {acc*100:.1f}%")
            time.sleep(0.5)
            
            # 3. Test K=4
            test_count += 1
            logger.info(f"  [{test_count}/{total_tests}] Speculative K=4...")
            tokens, time_ms, rounds, acc, draft_ms, verify_ms = test_speculative(edge_client, prompt_text, k=4)
            if tokens > 0:
                results.append(TestResult(
                    method='spec_k4', prompt_type=prompt_type, prompt_id=prompt_id,
                    prompt_length=prompt_len, tokens_generated=tokens,
                    total_time_ms=time_ms, tokens_per_second=tokens/(time_ms/1000),
                    num_rounds=rounds, acceptance_rate=acc,
                    avg_draft_ms=draft_ms, avg_verify_ms=verify_ms
                ))
                logger.info(f"      ✓ {tokens} tokens, {time_ms:.0f}ms, {tokens/(time_ms/1000):.2f} tok/s")
                logger.info(f"        Rounds: {rounds}, Acceptance: {acc*100:.1f}%")
            time.sleep(0.5)
            
            # 4. Test K=6
            test_count += 1
            logger.info(f"  [{test_count}/{total_tests}] Speculative K=6...")
            tokens, time_ms, rounds, acc, draft_ms, verify_ms = test_speculative(edge_client, prompt_text, k=6)
            if tokens > 0:
                results.append(TestResult(
                    method='spec_k6', prompt_type=prompt_type, prompt_id=prompt_id,
                    prompt_length=prompt_len, tokens_generated=tokens,
                    total_time_ms=time_ms, tokens_per_second=tokens/(time_ms/1000),
                    num_rounds=rounds, acceptance_rate=acc,
                    avg_draft_ms=draft_ms, avg_verify_ms=verify_ms
                ))
                logger.info(f"      ✓ {tokens} tokens, {time_ms:.0f}ms, {tokens/(time_ms/1000):.2f} tok/s")
                logger.info(f"        Rounds: {rounds}, Acceptance: {acc*100:.1f}%")
            time.sleep(0.5)
    
    # Analysis
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE - ANALYSIS")
    logger.info("="*80)
    
    # Group by prompt type and method
    analysis = {}
    for r in results:
        key = (r.prompt_type, r.method)
        if key not in analysis:
            analysis[key] = []
        analysis[key].append(r)
    
    # Print summary
    logger.info("\nResults Summary by Prompt Type and Method:")
    logger.info("-" * 80)
    
    for prompt_type in ['easy', 'hard']:
        logger.info(f"\n{prompt_type.upper()} PROMPTS:")
        for method in ['direct', 'spec_k2', 'spec_k4', 'spec_k6']:
            key = (prompt_type, method)
            if key in analysis:
                entries = analysis[key]
                avg_tps = np.mean([e.tokens_per_second for e in entries])
                avg_tokens = np.mean([e.tokens_generated for e in entries])
                
                if method.startswith('spec'):
                    avg_acc = np.mean([e.acceptance_rate for e in entries])
                    avg_rounds = np.mean([e.num_rounds for e in entries])
                    logger.info(f"  {method:10s}: {avg_tps:5.2f} tok/s, {avg_tokens:5.1f} tokens, "
                               f"{avg_acc*100:5.1f}% acc, {avg_rounds:5.1f} rounds")
                else:
                    logger.info(f"  {method:10s}: {avg_tps:5.2f} tok/s, {avg_tokens:5.1f} tokens")
    
    # Speedup analysis
    logger.info("\nSpeedup Analysis (vs Direct):")
    logger.info("-" * 80)
    for prompt_type in ['easy', 'hard']:
        direct_key = (prompt_type, 'direct')
        if direct_key in analysis:
            direct_tps = np.mean([e.tokens_per_second for e in analysis[direct_key]])
            logger.info(f"\n{prompt_type.upper()}:")
            logger.info(f"  Direct baseline: {direct_tps:.2f} tok/s")
            
            for method in ['spec_k2', 'spec_k4', 'spec_k6']:
                key = (prompt_type, method)
                if key in analysis:
                    spec_tps = np.mean([e.tokens_per_second for e in analysis[key]])
                    ratio = spec_tps / direct_tps
                    logger.info(f"  {method}: {spec_tps:.2f} tok/s ({ratio:.2f}x of direct)")
    
    # Save results
    outputs_dir = Path(__file__).parent / 'outputs'
    outputs_dir.mkdir(exist_ok=True)
    output_file = outputs_dir / 'comprehensive_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'results': [asdict(r) for r in results],
            'summary': {
                'easy_direct': np.mean([e.tokens_per_second for e in analysis.get(('easy', 'direct'), [])]),
                'easy_k2': np.mean([e.tokens_per_second for e in analysis.get(('easy', 'spec_k2'), [])]),
                'easy_k4': np.mean([e.tokens_per_second for e in analysis.get(('easy', 'spec_k4'), [])]),
                'easy_k6': np.mean([e.tokens_per_second for e in analysis.get(('easy', 'spec_k6'), [])]),
                'hard_direct': np.mean([e.tokens_per_second for e in analysis.get(('hard', 'direct'), [])]),
                'hard_k2': np.mean([e.tokens_per_second for e in analysis.get(('hard', 'spec_k2'), [])]),
                'hard_k4': np.mean([e.tokens_per_second for e in analysis.get(('hard', 'spec_k4'), [])]),
                'hard_k6': np.mean([e.tokens_per_second for e in analysis.get(('hard', 'spec_k6'), [])]),
            }
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("="*80)
    
    return results

if __name__ == "__main__":
    run_experiment()
