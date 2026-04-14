#!/usr/bin/env python3
"""Main entry point for vLLM-based speculative decoding edge client.

Recommended Python: /home/albert/learn/l-vllm/.venv/bin/python
(vLLM 0.17.0 with GGUF support)
"""
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
from config import (
    MODEL_PATH, GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN, CLOUD_URL, MAX_NEW_TOKENS, TEMPERATURE
)
from vllm_model_manager import VLLMModelManager
from vllm_draft_generator import VLLMDraftGenerator
from client.http_cloud_client import create_http_cloud_client
from client.edge_client import EdgeClient
from policies.policies import static_k_policy


def main():
    """Run vLLM-based speculative decoding edge client."""
    logger.info("=== vLLM Speculative Decoding Edge Client ===")
    
    # Step 1: Load model with vLLM
    logger.info("Loading draft model with vLLM...")
    model_manager = VLLMModelManager(
        model_path=MODEL_PATH,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
    )
    llm, tokenizer = model_manager.load()
    logger.info(f"Model loaded: {model_manager.get_model_info()}")
    
    # Step 2: Create draft generator
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    
    # Step 3: Create cloud client
    logger.info(f"Connecting to cloud server: {CLOUD_URL}")
    cloud_client = create_http_cloud_client(
        server_url=CLOUD_URL,
        timeout=60.0,
        retry_attempts=3
    )
    
    # Step 4: Create edge client
    edge_client = EdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=cloud_client,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
    )
    
    # Step 5: Run test generation
    test_prompt = "What is the capital of France?"
    logger.info(f"Running test generation with prompt: {test_prompt!r}")
    
    # Use static K=5 policy
    policy = static_k_policy(k=5)
    
    metrics = edge_client.generate(
        prompt=test_prompt,
        policy=policy,
        policy_name="static_k5"
    )
    
    # Print results
    logger.info("=== Generation Complete ===")
    logger.info(f"Generated {metrics.generated_tokens} tokens in {metrics.total_rounds} rounds")
    logger.info(f"Acceptance ratio: {metrics.acceptance_ratio:.2%}")
    logger.info(f"Total latency: {metrics.total_latency_ms:.2f}ms")
    logger.info(f"Draft time: {metrics.total_edge_draft_time_ms:.2f}ms")
    logger.info(f"Verify time: {metrics.total_server_verify_time_ms:.2f}ms")
    logger.info(f"Network time: {metrics.total_network_time_ms:.2f}ms")
    
    # Decode final output
    full_text = edge_client.decode_tokens(
        tokenizer.encode(test_prompt) + []  # Would need to track generated tokens
    )
    logger.info(f"Output text length: {len(full_text)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
