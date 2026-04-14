"""
Comprehensive Benchmark: K=6, K=8, K=10 (Run 3)
Tests: 10 easy + 10 hard prompts × (Direct + K=6 + K=8 + K=10) = 40 tests
Output: outputs_run3/comprehensive_results.json, comprehensive_log.txt
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

# Add parent directory to path (draft/ folder)
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
    method: str
    prompt_type: str
    prompt_id: int
    prompt_length: int
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float
    num_rounds: int = 0
    acceptance_rate: float = 0.0
    avg_draft_ms: float = 0.0
    avg_verify_ms: float = 0.0
    avg_rtt_ms: float = 0.0
    avg_network_ms: float = 0.0
    total_edge_time_ms: float = 0.0
    total_server_time_ms: float = 0.0
    total_network_time_ms: float = 0.0
    # New metrics for better analysis
    server_pure_inference_ms: float = 0.0  # Server time excluding network
    avg_draft_confidence: float = 0.0  # Draft model confidence score
    avg_rejection_position: float = 0.0  # Average position of first rejection

def load_prompts_from_file(filepath: Path, count: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """Parse prompts.txt and select count easy and count hard prompts."""
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
        for idx, (id_str, length, text) in enumerate(easy_matches[:count*2], 1):
            text = text.strip()[:400]
            easy_prompts.append({"id": idx, "length": int(length), "text": text})
    
    # Parse HARD prompts
    hard_section = re.search(r'HARD BENCHMARKS.*?(?=$)', content, re.DOTALL)
    if hard_section:
        hard_matches = re.findall(r'\[(\d+)\]\s+benchmark_[\d\-]+.*?Actual length:\s*(\d+)\s*chars.*?User:\s*(.*?)(?=\n\n\[|$)', 
                                  hard_section.group(), re.DOTALL)
        for idx, (id_str, length, text) in enumerate(hard_matches[:count*2], 1):
            text = text.strip()[:400]
            hard_prompts.append({"id": idx, "length": int(length), "text": text})
    
    # Select balanced prompts
    def select_balanced(prompts: List[Dict], target_count: int, target_len: int = 300) -> List[Dict]:
        sorted_prompts = sorted(prompts, key=lambda x: abs(x['length'] - target_len))
        selected = sorted_prompts[:target_count]
        return sorted(selected, key=lambda x: x['id'])
    
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

# Load prompts from file
PROMPTS_FILE = Path(__file__).parent.parent / 'benchmarks' / 'prompts.txt'
EASY_PROMPTS, HARD_PROMPTS = load_prompts_from_file(PROMPTS_FILE, count=10)

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

def test_speculative(client: EdgeClient, prompt: str, k: int) -> Tuple[int, float, int, float, float, float, float, float, float, float, float, float, float, float]:
    """Test speculative with full metrics capture including new analysis metrics"""
    start = time.time()
    try:
        def policy(round_id, draft_tokens):
            return k
        
        metrics = client.generate(prompt=prompt, policy=policy, policy_name=f"StaticK{k}")
        total_ms = (time.time() - start) * 1000
        
        rounds = metrics.total_rounds
        # Per-round averages
        avg_draft = metrics.total_edge_draft_time_ms / rounds if rounds > 0 else 0
        avg_verify = metrics.total_server_verify_time_ms / rounds if rounds > 0 else 0
        avg_rtt = metrics.average_rtt_ms
        avg_network = metrics.total_network_time_ms / rounds if rounds > 0 else 0
        
        # Totals for analysis
        total_edge = metrics.total_edge_draft_time_ms
        total_server = metrics.total_server_verify_time_ms
        total_network = metrics.total_network_time_ms
        
        # NEW: Calculate server pure inference time (excluding network overhead in measurement)
        # server_verify_time_ms is already pure server inference time
        server_pure_inference = avg_verify
        
        # NEW: Calculate average rejection position from round_details
        # rejection_position = accepted_len (0-indexed, so if accepted=3 out of K=5, rejected at position 3)
        rejection_positions = []
        for rd in metrics.round_details:
            if rd['accepted'] < rd['drafted']:
                # Rejection happened at this position (0-indexed)
                rejection_positions.append(rd['accepted'])
            elif rd['accepted'] == rd['drafted'] and rd['accepted'] > 0:
                # Full acceptance, mark as K (no rejection in this round)
                rejection_positions.append(rd['drafted'])  # Use drafted as "no rejection"
        
        avg_rejection_pos = sum(rejection_positions) / len(rejection_positions) if rejection_positions else k
        
        # NEW: Draft confidence - use acceptance ratio as proxy (we don't have per-token probs in metrics)
        # In a better implementation, we'd track average draft token probability
        draft_confidence = metrics.acceptance_ratio  # Proxy for now
        
        return (
            metrics.generated_tokens,
            total_ms,
            rounds,
            metrics.acceptance_ratio,
            avg_draft,
            avg_verify,
            avg_rtt,
            avg_network,
            total_edge,
            total_server,
            total_network,
            server_pure_inference,
            draft_confidence,
            avg_rejection_pos
        )
    except Exception as e:
        logger.error(f"Speculative K={k} failed: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

def run_experiment():
    server_url = "http://localhost:6006"
    results = []
    
    # Output directory for this run
    outputs_dir = Path(__file__).parent / 'outputs_run3'
    outputs_dir.mkdir(exist_ok=True)
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE BENCHMARK RUN 3: K=6, K=8, K=10")
    logger.info("Methods: Direct, K=6, K=8, K=10")
    logger.info(f"Total tests: {len(EASY_PROMPTS) + len(HARD_PROMPTS)} prompts × 4 methods = {(len(EASY_PROMPTS) + len(HARD_PROMPTS)) * 4} tests")
    logger.info(f"Output: {outputs_dir}")
    logger.info("="*80)
    
    # Initialize
    logger.info("\n[Setup] Loading draft model...")
    model_manager = VLLMModelManager(MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    cloud_client = create_http_cloud_client(server_url, timeout=120.0)
    edge_client = EdgeClient(model_manager, draft_generator, cloud_client, 
                              max_new_tokens=128, temperature=TEMPERATURE)
    logger.info("[Setup] Ready!\n")
    
    test_count = 0
    total_tests = (len(EASY_PROMPTS) + len(HARD_PROMPTS)) * 4
    
    # Run tests
    for prompt_type, prompt_list in [('easy', EASY_PROMPTS), ('hard', HARD_PROMPTS)]:
        logger.info(f"\n{'='*40}")
        logger.info(f"Testing {prompt_type.upper()} PROMPTS ({len(prompt_list)} total)")
        logger.info(f"{'='*40}\n")
        
        for prompt_data in prompt_list:
            prompt_id = prompt_data['id']
            prompt_text = prompt_data['text']
            prompt_len = prompt_data['length']
            
            logger.info(f"\n[{'EASY' if prompt_type == 'easy' else 'HARD'} Prompt {prompt_id}] Length: {prompt_len} chars")
            
            # 1. Test Direct
            test_count += 1
            logger.info(f"  [{test_count}/{total_tests}] Direct...")
            tokens, time_ms = test_direct(server_url, prompt_text, max_tokens=128)
            if tokens > 0:
                results.append(TestResult(
                    method='direct', prompt_type=prompt_type, prompt_id=prompt_id,
                    prompt_length=prompt_len, tokens_generated=tokens,
                    total_time_ms=time_ms, tokens_per_second=tokens/(time_ms/1000)
                ))
                logger.info(f"      ✓ {tokens} tokens, {time_ms:.0f}ms, {tokens/(time_ms/1000):.2f} tok/s")
            time.sleep(0.5)
            
            # 2. Test K=6
            test_count += 1
            logger.info(f"  [{test_count}/{total_tests}] Speculative K=6...")
            tokens, time_ms, rounds, acc, draft_ms, verify_ms, rtt_ms, net_ms, tot_edge, tot_srv, tot_net, srv_pure, draft_conf, reject_pos = test_speculative(edge_client, prompt_text, k=6)
            if tokens > 0:
                results.append(TestResult(
                    method='spec_k6', prompt_type=prompt_type, prompt_id=prompt_id,
                    prompt_length=prompt_len, tokens_generated=tokens,
                    total_time_ms=time_ms, tokens_per_second=tokens/(time_ms/1000),
                    num_rounds=rounds, acceptance_rate=acc,
                    avg_draft_ms=draft_ms, avg_verify_ms=verify_ms,
                    avg_rtt_ms=rtt_ms, avg_network_ms=net_ms,
                    total_edge_time_ms=tot_edge,
                    total_server_time_ms=tot_srv,
                    total_network_time_ms=tot_net,
                    server_pure_inference_ms=srv_pure,
                    avg_draft_confidence=draft_conf,
                    avg_rejection_position=reject_pos
                ))
                logger.info(f"      ✓ {tokens} tokens, {time_ms:.0f}ms, {tokens/(time_ms/1000):.2f} tok/s")
                logger.info(f"        Rounds: {rounds}, Accept: {acc*100:.1f}%, RTT: {rtt_ms:.0f}ms, Net%: {net_ms/(draft_ms+verify_ms+net_ms)*100:.0f}%, RejectPos: {reject_pos:.1f}")
            time.sleep(0.5)
            
            # 3. Test K=8
            test_count += 1
            logger.info(f"  [{test_count}/{total_tests}] Speculative K=8...")
            tokens, time_ms, rounds, acc, draft_ms, verify_ms, rtt_ms, net_ms, tot_edge, tot_srv, tot_net, srv_pure, draft_conf, reject_pos = test_speculative(edge_client, prompt_text, k=8)
            if tokens > 0:
                results.append(TestResult(
                    method='spec_k8', prompt_type=prompt_type, prompt_id=prompt_id,
                    prompt_length=prompt_len, tokens_generated=tokens,
                    total_time_ms=time_ms, tokens_per_second=tokens/(time_ms/1000),
                    num_rounds=rounds, acceptance_rate=acc,
                    avg_draft_ms=draft_ms, avg_verify_ms=verify_ms,
                    avg_rtt_ms=rtt_ms, avg_network_ms=net_ms,
                    total_edge_time_ms=tot_edge,
                    total_server_time_ms=tot_srv,
                    total_network_time_ms=tot_net,
                    server_pure_inference_ms=srv_pure,
                    avg_draft_confidence=draft_conf,
                    avg_rejection_position=reject_pos
                ))
                logger.info(f"      ✓ {tokens} tokens, {time_ms:.0f}ms, {tokens/(time_ms/1000):.2f} tok/s")
                logger.info(f"        Rounds: {rounds}, Accept: {acc*100:.1f}%, RTT: {rtt_ms:.0f}ms, Net%: {net_ms/(draft_ms+verify_ms+net_ms)*100:.0f}%, RejectPos: {reject_pos:.1f}")
            time.sleep(0.5)
            
            # 4. Test K=10
            test_count += 1
            logger.info(f"  [{test_count}/{total_tests}] Speculative K=10...")
            tokens, time_ms, rounds, acc, draft_ms, verify_ms, rtt_ms, net_ms, tot_edge, tot_srv, tot_net, srv_pure, draft_conf, reject_pos = test_speculative(edge_client, prompt_text, k=10)
            if tokens > 0:
                results.append(TestResult(
                    method='spec_k10', prompt_type=prompt_type, prompt_id=prompt_id,
                    prompt_length=prompt_len, tokens_generated=tokens,
                    total_time_ms=time_ms, tokens_per_second=tokens/(time_ms/1000),
                    num_rounds=rounds, acceptance_rate=acc,
                    avg_draft_ms=draft_ms, avg_verify_ms=verify_ms,
                    avg_rtt_ms=rtt_ms, avg_network_ms=net_ms,
                    total_edge_time_ms=tot_edge,
                    total_server_time_ms=tot_srv,
                    total_network_time_ms=tot_net,
                    server_pure_inference_ms=srv_pure,
                    avg_draft_confidence=draft_conf,
                    avg_rejection_position=reject_pos
                ))
                logger.info(f"      ✓ {tokens} tokens, {time_ms:.0f}ms, {tokens/(time_ms/1000):.2f} tok/s")
                logger.info(f"        Rounds: {rounds}, Accept: {acc*100:.1f}%, RTT: {rtt_ms:.0f}ms, Net%: {net_ms/(draft_ms+verify_ms+net_ms)*100:.0f}%, RejectPos: {reject_pos:.1f}")
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
    
    for prompt_type in ['easy', 'hard']:
        logger.info(f"\n{prompt_type.upper()} PROMPTS:")
        
        key = (prompt_type, 'direct')
        if key in analysis:
            direct_tps = np.mean([e.tokens_per_second for e in analysis[key]])
            logger.info(f"  Direct baseline: {direct_tps:.2f} tok/s")
            
            for method in ['spec_k6', 'spec_k8', 'spec_k10']:
                key = (prompt_type, method)
                if key in analysis:
                    spec_tps = np.mean([e.tokens_per_second for e in analysis[key]])
                    ratio = spec_tps / direct_tps
                    avg_rtt = np.mean([e.avg_rtt_ms for e in analysis[key]])
                    avg_accept = np.mean([e.acceptance_rate for e in analysis[key]]) * 100
                    avg_reject_pos = np.mean([e.avg_rejection_position for e in analysis[key]])
                    avg_srv_inference = np.mean([e.server_pure_inference_ms for e in analysis[key]])
                    logger.info(f"  {method}: {spec_tps:.2f} tok/s ({ratio:.2f}x of direct), RTT: {avg_rtt:.0f}ms, Accept: {avg_accept:.1f}%, RejectPos: {avg_reject_pos:.1f}, SrvInf: {avg_srv_inference:.0f}ms")
    
    # Save results
    output_file = outputs_dir / 'comprehensive_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'results': [asdict(r) for r in results],
            'summary': {
                'easy_direct': np.mean([e.tokens_per_second for e in analysis.get(('easy', 'direct'), [])]),
                'easy_k6': np.mean([e.tokens_per_second for e in analysis.get(('easy', 'spec_k6'), [])]),
                'easy_k8': np.mean([e.tokens_per_second for e in analysis.get(('easy', 'spec_k8'), [])]),
                'easy_k10': np.mean([e.tokens_per_second for e in analysis.get(('easy', 'spec_k10'), [])]),
                'hard_direct': np.mean([e.tokens_per_second for e in analysis.get(('hard', 'direct'), [])]),
                'hard_k6': np.mean([e.tokens_per_second for e in analysis.get(('hard', 'spec_k6'), [])]),
                'hard_k8': np.mean([e.tokens_per_second for e in analysis.get(('hard', 'spec_k8'), [])]),
                'hard_k10': np.mean([e.tokens_per_second for e in analysis.get(('hard', 'spec_k10'), [])]),
                # New detailed metrics
                'easy_k6_accept': np.mean([e.acceptance_rate for e in analysis.get(('easy', 'spec_k6'), [])]),
                'easy_k8_accept': np.mean([e.acceptance_rate for e in analysis.get(('easy', 'spec_k8'), [])]),
                'easy_k10_accept': np.mean([e.acceptance_rate for e in analysis.get(('easy', 'spec_k10'), [])]),
                'hard_k6_accept': np.mean([e.acceptance_rate for e in analysis.get(('hard', 'spec_k6'), [])]),
                'hard_k8_accept': np.mean([e.acceptance_rate for e in analysis.get(('hard', 'spec_k8'), [])]),
                'hard_k10_accept': np.mean([e.acceptance_rate for e in analysis.get(('hard', 'spec_k10'), [])]),
                'easy_k6_reject_pos': np.mean([e.avg_rejection_position for e in analysis.get(('easy', 'spec_k6'), [])]),
                'easy_k8_reject_pos': np.mean([e.avg_rejection_position for e in analysis.get(('easy', 'spec_k8'), [])]),
                'easy_k10_reject_pos': np.mean([e.avg_rejection_position for e in analysis.get(('easy', 'spec_k10'), [])]),
                'hard_k6_reject_pos': np.mean([e.avg_rejection_position for e in analysis.get(('hard', 'spec_k6'), [])]),
                'hard_k8_reject_pos': np.mean([e.avg_rejection_position for e in analysis.get(('hard', 'spec_k8'), [])]),
                'hard_k10_reject_pos': np.mean([e.avg_rejection_position for e in analysis.get(('hard', 'spec_k10'), [])]),
            }
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("="*80)
    
    return results

if __name__ == "__main__":
    run_experiment()
