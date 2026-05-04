#!/usr/bin/env python3
"""Quick test: Direct vs StaticK3 vs StaticK5 vs AsyncK3"""
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import requests

sys.path.insert(0, str(Path(__file__).parent))

from client.edge_client import EdgeClient
from client.async_edge_client import AsyncEdgeClient
from client.http_cloud_client import create_http_cloud_client
from config import DRAFT_GPU_MEM as GPU_MEMORY_UTILIZATION, DRAFT_MAX_LEN as MAX_MODEL_LEN, DRAFT_MODEL_NAME as MODEL_NAME, DRAFT_MODEL_PATH as MODEL_PATH
from draft_generator import VLLMDraftGenerator
from model_manager import VLLMModelManager
from prompt_loader import load_prompts_by_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:6006"
MAX_TOKENS = 128
K_VALUES = [3, 5]
ASYNC_K = 3
PROMPT_COUNT = 2  # 2 easy + 2 hard


@dataclass
class QuickResult:
    method: str
    prompt_type: str
    prompt_id: int
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float
    acceptance_rate: float = 0.0
    num_rounds: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    network_time_ms: float = 0.0
    avg_rtt_ms: float = 0.0
    pipeline_efficiency: float = 0.0  # async only


def load_prompts():
    """Load PROMPT_COUNT easy + PROMPT_COUNT hard prompts from data/prompt.json."""
    simple_prompts, complex_prompts = load_prompts_by_type(
        count_per_type=PROMPT_COUNT
    )
    
    # Map Simple -> easy, Complex -> hard for compatibility
    easy_prompts = [{"id": p["id"], "text": p["text"][:300]} for p in simple_prompts]
    hard_prompts = [{"id": p["id"], "text": p["text"][:300]} for p in complex_prompts]
    
    logger.info(f"Loaded {len(easy_prompts)} easy + {len(hard_prompts)} hard prompts from data/prompt.json")
    return easy_prompts, hard_prompts


def test_direct(prompt: str) -> Tuple[int, float]:
    """Test direct generation."""
    start = time.time()
    try:
        response = requests.post(
            f"{SERVER_URL}/generate",
            json={"prompt": prompt, "max_tokens": MAX_TOKENS, "temperature": 0.0},
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        total_ms = (time.time() - start) * 1000
        return result["tokens_generated"], total_ms
    except Exception as e:
        logger.error(f"Direct failed: {e}")
        return 0, 0


def test_speculative(client: EdgeClient, prompt: str, k: int) -> Tuple[int, float, float, int, float, float, float, float]:
    """Test sync speculative decoding. Returns (tokens, time_ms, accept_rate, rounds, draft_ms, verify_ms, net_ms, rtt_ms)"""
    start = time.time()
    try:
        metrics = client.generate(
            prompt=prompt, 
            policy=lambda _round_id, _draft_tokens: k, 
            policy_name=f"StaticK{k}"
        )
        total_ms = (time.time() - start) * 1000
        return (
            metrics.generated_tokens,
            total_ms,
            metrics.acceptance_ratio,
            metrics.total_rounds,
            metrics.total_edge_draft_time_ms,
            metrics.total_server_verify_time_ms,
            metrics.total_network_time_ms,
            metrics.average_rtt_ms,
        )
    except Exception as e:
        logger.error(f"Speculative K={k} failed: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0


def run_quick_test():
    """Run quick comparison test."""
    logger.info("=" * 70)
    logger.info("QUICK TEST: Direct vs StaticK3 vs StaticK5 vs AsyncK3")
    logger.info(f"Draft model: {MODEL_NAME}")
    logger.info(f"GPU util: {GPU_MEMORY_UTILIZATION}, Max len: {MAX_MODEL_LEN}")
    logger.info("=" * 70)
    
    # Load prompts
    easy_prompts, hard_prompts = load_prompts()
    
    cloud_client = create_http_cloud_client(SERVER_URL, timeout=60.0)
    
    # Load draft model
    logger.info("Loading draft model...")
    model_manager = VLLMModelManager(MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, gpu_id=1)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    
    # Sync client
    edge_client = EdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=cloud_client,
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    
    # Async client (lookahead=2)
    async_client = AsyncEdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=cloud_client,
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
        lookahead=2,
    )
    logger.info("Draft model loaded!")
    
    results: List[QuickResult] = []
    
    # Test each prompt
    for prompt_type, prompts in [("easy", easy_prompts), ("hard", hard_prompts)]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {prompt_type.upper()} prompts")
        logger.info("=" * 60)
        
        for prompt_data in prompts:
            pid = prompt_data["id"]
            text = prompt_data["text"]
            logger.info(f"\n[Prompt {pid}] {text[:50]}...")
            
            # Direct
            logger.info("  Testing Direct...")
            tokens, time_ms = test_direct(text)
            if tokens > 0:
                tps = tokens / (time_ms / 1000)
                results.append(QuickResult("direct", prompt_type, pid, tokens, time_ms, tps))
                logger.info(f"    -> {tokens} tokens, {time_ms:.0f}ms, {tps:.2f} tok/s")
            time.sleep(0.5)
            
            # Speculative K=3, K=5
            for k in K_VALUES:
                logger.info(f"  Testing Speculative K={k}...")
                tokens, time_ms, acc, rounds, draft_ms, verify_ms, net_ms, rtt_ms = test_speculative(edge_client, text, k)
                if tokens > 0:
                    tps = tokens / (time_ms / 1000)
                    results.append(QuickResult(
                        f"k{k}", prompt_type, pid, tokens, time_ms, tps, acc, rounds,
                        draft_time_ms=draft_ms, verify_time_ms=verify_ms, network_time_ms=net_ms, avg_rtt_ms=rtt_ms
                    ))
                    logger.info(f"    -> {tokens} tokens, {time_ms:.0f}ms, {tps:.2f} tok/s, "
                              f"accept={acc:.1%}, rounds={rounds}, "
                              f"draft={draft_ms:.0f}ms, verify={verify_ms:.0f}ms, net={net_ms:.0f}ms")
                time.sleep(0.5)
            
            # Async K=3
            logger.info(f"  Testing Async K={ASYNC_K}...")
            start = time.time()
            try:
                async_metrics = async_client.generate(
                    prompt=text,
                    policy=lambda _round_id, _draft_tokens: ASYNC_K,
                    policy_name=f"AsyncK{ASYNC_K}"
                )
                total_ms = (time.time() - start) * 1000
                if async_metrics.generated_tokens > 0:
                    tps = async_metrics.generated_tokens / (total_ms / 1000)
                    results.append(QuickResult(
                        f"async_k{ASYNC_K}", prompt_type, pid, async_metrics.generated_tokens, total_ms, tps,
                        async_metrics.acceptance_ratio, async_metrics.total_rounds,
                        draft_time_ms=async_metrics.total_edge_draft_time_ms,
                        verify_time_ms=async_metrics.total_server_verify_time_ms,
                        network_time_ms=async_metrics.total_network_time_ms,
                        avg_rtt_ms=async_metrics.average_rtt_ms,
                        pipeline_efficiency=async_metrics.pipeline_efficiency,
                    ))
                    logger.info(f"    -> {async_metrics.generated_tokens} tokens, {total_ms:.0f}ms, {tps:.2f} tok/s, "
                              f"accept={async_metrics.acceptance_ratio:.1%}, rounds={async_metrics.total_rounds}, "
                              f"pipe_eff={async_metrics.pipeline_efficiency:.1%}, "
                              f"draft={async_metrics.total_edge_draft_time_ms:.0f}ms, verify={async_metrics.total_server_verify_time_ms:.0f}ms, net={async_metrics.total_network_time_ms:.0f}ms")
            except Exception as e:
                logger.error(f"Async K={ASYNC_K} failed: {e}")
            time.sleep(0.5)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    from collections import defaultdict
    method_stats = defaultdict(lambda: {"tps": [], "accept": [], "draft_ms": [], "verify_ms": [], "net_ms": []})
    
    for r in results:
        method_stats[r.method]["tps"].append(r.tokens_per_second)
        if r.acceptance_rate > 0:
            method_stats[r.method]["accept"].append(r.acceptance_rate)
        if r.draft_time_ms > 0:
            method_stats[r.method]["draft_ms"].append(r.draft_time_ms)
        if r.verify_time_ms > 0:
            method_stats[r.method]["verify_ms"].append(r.verify_time_ms)
        if r.network_time_ms > 0:
            method_stats[r.method]["net_ms"].append(r.network_time_ms)
    
    method_order = ["direct"] + [f"k{k}" for k in K_VALUES] + [f"async_k{ASYNC_K}"]
    for method in method_order:
        if method in method_stats:
            avg_tps = sum(method_stats[method]["tps"]) / len(method_stats[method]["tps"])
            avg_accept = sum(method_stats[method]["accept"]) / len(method_stats[method]["accept"]) if method_stats[method]["accept"] else 0
            avg_draft = sum(method_stats[method]["draft_ms"]) / len(method_stats[method]["draft_ms"]) if method_stats[method]["draft_ms"] else 0
            avg_verify = sum(method_stats[method]["verify_ms"]) / len(method_stats[method]["verify_ms"]) if method_stats[method]["verify_ms"] else 0
            avg_net = sum(method_stats[method]["net_ms"]) / len(method_stats[method]["net_ms"]) if method_stats[method]["net_ms"] else 0
            logger.info(f"{method:10s}: {avg_tps:6.2f} tok/s (accept: {avg_accept:.1%}, draft: {avg_draft:.0f}ms, verify: {avg_verify:.0f}ms, net: {avg_net:.0f}ms)")
    
    # Speedup calculation
    if "direct" in method_stats:
        direct_tps = sum(method_stats["direct"]["tps"]) / len(method_stats["direct"]["tps"])
        for method in [f"k{k}" for k in K_VALUES] + [f"async_k{ASYNC_K}"]:
            if method in method_stats:
                spec_tps = sum(method_stats[method]["tps"]) / len(method_stats[method]["tps"])
                speedup = spec_tps / direct_tps
                logger.info(f"Speedup {method} vs Direct: {speedup:.2f}x")
    
    # Save results
    output_dir = Path(__file__).parent / "experiments" / "outputs_quick"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "quick_test_results.json"
    with open(out_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"\nResults saved to: {out_file}")

    logger.info("\n(Speculative speedup requires network latency > draft latency — expected on remote setups)")
    return True


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
