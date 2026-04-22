#!/usr/bin/env python3
"""Quick test: 2 easy + 2 hard prompts comparing Direct vs K=2,4"""
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
from client.http_cloud_client import create_http_cloud_client
from config import DRAFT_GPU_MEM as GPU_MEMORY_UTILIZATION, DRAFT_MAX_LEN as MAX_MODEL_LEN, DRAFT_MODEL_NAME as MODEL_NAME, DRAFT_MODEL_PATH as MODEL_PATH
from draft_generator import VLLMDraftGenerator
from model_manager import VLLMModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:6006"
MAX_TOKENS = 128  # Increased for better comparison
K_VALUES = [2, 4]
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


def load_prompts():
    """Load 2 easy + 2 hard prompts."""
    prompts_file = Path(__file__).parent / "benchmarks" / "prompts.txt"
    
    with open(prompts_file, "r") as f:
        content = f.read()
    
    import re
    
    # Extract easy prompts
    easy_section = re.search(r"EASY BENCHMARKS.*?HARD BENCHMARKS", content, re.DOTALL)
    easy_matches = re.findall(
        r"\[(\d+)\]\s+benchmark_[\d\-]+.*?User:\s*(.*?)(?=\n\n\[|$)",
        easy_section.group() if easy_section else "",
        re.DOTALL,
    )
    easy_prompts = [{"id": i+1, "text": text.strip()[:300]} for i, (_, text) in enumerate(easy_matches[:2])]
    
    # Extract hard prompts
    hard_section = re.search(r"HARD BENCHMARKS.*?(?=$)", content, re.DOTALL)
    hard_matches = re.findall(
        r"\[(\d+)\]\s+benchmark_[\d\-]+.*?User:\s*(.*?)(?=\n\n\[|$)",
        hard_section.group() if hard_section else "",
        re.DOTALL,
    )
    hard_prompts = [{"id": i+1, "text": text.strip()[:300]} for i, (_, text) in enumerate(hard_matches[:2])]
    
    logger.info(f"Loaded {len(easy_prompts)} easy + {len(hard_prompts)} hard prompts")
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


def test_speculative(client: EdgeClient, prompt: str, k: int) -> Tuple[int, float, float, int]:
    """Test speculative decoding."""
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
            metrics.total_rounds
        )
    except Exception as e:
        logger.error(f"Speculative K={k} failed: {e}")
        return 0, 0, 0, 0


def run_quick_test():
    """Run quick comparison test."""
    logger.info("=" * 70)
    logger.info("QUICK TEST: Direct vs Speculative (K=2,4)")
    logger.info(f"Draft model: {MODEL_NAME}")
    logger.info(f"GPU util: {GPU_MEMORY_UTILIZATION}, Max len: {MAX_MODEL_LEN}")
    logger.info("=" * 70)
    
    # Load prompts
    easy_prompts, hard_prompts = load_prompts()
    
    cloud_client = create_http_cloud_client(SERVER_URL, timeout=60.0)
    
    # Load draft model
    logger.info("Loading draft model...")
    model_manager = VLLMModelManager(MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    edge_client = EdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=cloud_client,
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
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
            
            # Speculative K=2,4
            for k in K_VALUES:
                logger.info(f"  Testing Speculative K={k}...")
                tokens, time_ms, acc, rounds = test_speculative(edge_client, text, k)
                if tokens > 0:
                    tps = tokens / (time_ms / 1000)
                    results.append(QuickResult(
                        f"k{k}", prompt_type, pid, tokens, time_ms, tps, acc, rounds
                    ))
                    logger.info(f"    -> {tokens} tokens, {time_ms:.0f}ms, {tps:.2f} tok/s, "
                              f"accept={acc:.1%}, rounds={rounds}")
                time.sleep(0.5)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    # Group by method and calculate averages
    from collections import defaultdict
    method_stats = defaultdict(lambda: {"tps": [], "accept": []})
    
    for r in results:
        method_stats[r.method]["tps"].append(r.tokens_per_second)
        if r.acceptance_rate > 0:
            method_stats[r.method]["accept"].append(r.acceptance_rate)
    
    for method in ["direct", "k2", "k4"]:
        if method in method_stats:
            avg_tps = sum(method_stats[method]["tps"]) / len(method_stats[method]["tps"])
            avg_accept = sum(method_stats[method]["accept"]) / len(method_stats[method]["accept"]) if method_stats[method]["accept"] else 0
            logger.info(f"{method:10s}: {avg_tps:6.2f} tok/s (accept: {avg_accept:.1%})")
    
    # Speedup calculation
    if "direct" in method_stats:
        direct_tps = sum(method_stats["direct"]["tps"]) / len(method_stats["direct"]["tps"])
        for method in ["k2", "k4"]:
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

    # Note: on a single machine both models share PCIe bandwidth, so
    # speculative decoding may not outperform direct generation here.
    # Speedup is expected when verify is a remote (higher-latency) server.
    logger.info("\n(Speculative speedup requires network latency > draft latency — expected on remote setups)")
    return True


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
