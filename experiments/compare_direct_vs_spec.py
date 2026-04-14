"""
简化的Direct vs Speculative性能对比实验
"""
import logging
import json
import time
import sys
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
import requests

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
    prompt: str
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float
    # Speculative specific
    num_rounds: int = 0
    acceptance_rate: float = 0.0
    avg_draft_ms: float = 0.0
    avg_verify_ms: float = 0.0

def load_test_prompts() -> List[str]:
    """加载测试prompts"""
    return [
        "I had a conversation with your aunt yesterday. I wrote her a letter about how content I feel. There are so many things that have been making me happy lately, like spending more time with my friends and taking on new challenges at work. And you know what, I'm grateful to have such a supportive sister...",
        "I love everything about autumn, from the way the leaves change color to the cool weather. It's a time of year when you can cozy up with a cup of coffee and a good book, or go for a long walk and enjoy the scenery. There's something so peaceful and calming about autumn that I can't help but feel nost...",
        "Doctor, I wanted to talk to you about my health regimen. I try to be very conscious of the amount of liquids I consume. I typically drink eight cups of water a day, but I'll cut down to six or seven if I'm feeling bloated. My go-to drink is unsweetened green tea, which I enjoy hot or iced....",
    ]

def test_direct(server_url: str, prompt: str, max_tokens: int = 128) -> TestResult:
    """测试Direct生成"""
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
        tokens = result['tokens_generated']
        
        return TestResult(
            method='direct',
            prompt=prompt[:50],
            tokens_generated=tokens,
            total_time_ms=total_ms,
            tokens_per_second=tokens / (total_ms / 1000)
        )
    except Exception as e:
        logger.error(f"Direct failed: {e}")
        return None

def test_speculative(client: EdgeClient, prompt: str) -> TestResult:
    """测试Speculative生成"""
    start = time.time()
    try:
        def policy(round_id, draft_tokens):
            return 2  # K=2
        
        metrics = client.generate(prompt=prompt, policy=policy, policy_name="StaticK2")
        total_ms = (time.time() - start) * 1000
        
        return TestResult(
            method='speculative',
            prompt=prompt[:50],
            tokens_generated=metrics.generated_tokens,
            total_time_ms=total_ms,
            tokens_per_second=metrics.generated_tokens / (total_ms / 1000),
            num_rounds=metrics.total_rounds,
            acceptance_rate=metrics.acceptance_ratio,
            avg_draft_ms=metrics.total_edge_draft_time_ms / metrics.total_rounds if metrics.total_rounds > 0 else 0,
            avg_verify_ms=metrics.total_server_verify_time_ms / metrics.total_rounds if metrics.total_rounds > 0 else 0
        )
    except Exception as e:
        logger.error(f"Speculative failed: {e}")
        return None

def main():
    server_url = "http://localhost:6006"
    prompts = load_test_prompts()
    
    logger.info("="*80)
    logger.info("Direct vs Speculative Decoding 性能对比")
    logger.info("="*80)
    
    # 测试Direct
    logger.info("\n[1/2] 测试Direct生成...")
    direct_results = []
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"  [{i}/{len(prompts)}] {prompt[:50]}...")
        result = test_direct(server_url, prompt)
        if result:
            direct_results.append(result)
            logger.info(f"    ✓ {result.tokens_generated} tokens, {result.total_time_ms:.0f}ms, {result.tokens_per_second:.2f} tok/s")
        time.sleep(1)
    
    # 测试Speculative
    logger.info("\n[2/2] 测试Speculative生成...")
    logger.info("  加载draft模型...")
    model_manager = VLLMModelManager(MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    cloud_client = create_http_cloud_client(server_url, timeout=120.0)
    edge_client = EdgeClient(model_manager, draft_generator, cloud_client, max_new_tokens=128, temperature=TEMPERATURE)
    
    spec_results = []
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"  [{i}/{len(prompts)}] {prompt[:50]}...")
        result = test_speculative(edge_client, prompt)
        if result:
            spec_results.append(result)
            logger.info(f"    ✓ {result.tokens_generated} tokens, {result.total_time_ms:.0f}ms, {result.tokens_per_second:.2f} tok/s")
            logger.info(f"      轮数: {result.num_rounds}, 接受率: {result.acceptance_rate*100:.1f}%")
        time.sleep(1)
    
    # 分析结果
    logger.info("\n" + "="*80)
    logger.info("实验结果")
    logger.info("="*80)
    
    if direct_results and spec_results:
        direct_avg_tps = sum(r.tokens_per_second for r in direct_results) / len(direct_results)
        spec_avg_tps = sum(r.tokens_per_second for r in spec_results) / len(spec_results)
        spec_avg_acc = sum(r.acceptance_rate for r in spec_results) / len(spec_results)
        spec_avg_rounds = sum(r.num_rounds for r in spec_results) / len(spec_results)
        
        logger.info(f"\nDirect:")
        logger.info(f"  平均速度: {direct_avg_tps:.2f} tokens/s")
        logger.info(f"\nSpeculative:")
        logger.info(f"  平均速度: {spec_avg_tps:.2f} tokens/s")
        logger.info(f"  平均接受率: {spec_avg_acc*100:.1f}%")
        logger.info(f"  平均轮数: {spec_avg_rounds:.1f}")
        logger.info(f"\n性能对比:")
        speedup = direct_avg_tps / spec_avg_tps
        if speedup > 1:
            logger.info(f"  Direct比Speculative快 {speedup:.2f}x")
        else:
            logger.info(f"  Speculative比Direct快 {1/speedup:.2f}x")
        
        # 保存结果
        output_dir = Path(__file__).parent
        results = {
            'direct': [asdict(r) for r in direct_results],
            'speculative': [asdict(r) for r in spec_results],
            'summary': {
                'direct_avg_tps': direct_avg_tps,
                'spec_avg_tps': spec_avg_tps,
                'spec_avg_acceptance': spec_avg_acc,
                'speedup': speedup
            }
        }
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n结果已保存到: {output_dir / 'results.json'}")
    
    logger.info("\n" + "="*80)
    logger.info("实验完成")
    logger.info("="*80)

if __name__ == "__main__":
    main()
