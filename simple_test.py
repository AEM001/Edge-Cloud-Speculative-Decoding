"""Simple test to run speculative decoding with the cloud server on port 5090."""
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from root (vLLM versions)
from config import MODEL_NAME, MODEL_PATH, TEMPERATURE, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN
from model_manager import VLLMModelManager
from draft_generator import VLLMDraftGenerator
from client.edge_client import EdgeClient
from client.http_cloud_client import create_http_cloud_client

def main():
    """Run a simple test."""
    logger.info("=== Simple Speculative Decoding Test ===")
    
    # Server URL - via SSH tunnel to RTX 5090 verification server
    # Make sure SSH tunnel is running: ssh -p 20514 -L 6006:localhost:6006 root@connect.westd.seetacloud.com -N
    server_url = "http://localhost:6006"
    logger.info(f"Cloud server: {server_url}")
    
    # Load draft model
    logger.info(f"Loading draft model: {MODEL_NAME}")
    model_manager = VLLMModelManager(
        model_path=MODEL_PATH,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
    )
    llm, tokenizer = model_manager.load()
    logger.info(f"Model loaded: {model_manager.get_model_info()}")
    
    # Initialize draft generator
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    
    # Create HTTP cloud client
    cloud_client = create_http_cloud_client(
        server_url=server_url,
        timeout=60.0,
        retry_attempts=3
    )
    
    # Check server health
    if not cloud_client.check_health():
        logger.error("Cloud server is not healthy!")
        sys.exit(1)
    
    # Initialize edge client
    edge_client = EdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=cloud_client,
        max_new_tokens=32,  # Short test
        temperature=TEMPERATURE
    )
    
    # Simple static K=2 policy
    def static_k_2(round_id, draft_tokens):
        return 2
    
    # Test prompt
    test_prompt = "Hello, how are you today?"
    logger.info(f"\nTest prompt: {test_prompt!r}")
    
    # Generate
    logger.info("Starting generation...")
    metrics = edge_client.generate(
        prompt=test_prompt,
        policy=static_k_2,
        policy_name="StaticK2"
    )
    
    # Print results
    logger.info("\n=== Results ===")
    logger.info(f"Generated tokens: {metrics.generated_tokens}")
    logger.info(f"Total rounds: {metrics.total_rounds}")
    logger.info(f"Acceptance ratio: {metrics.acceptance_ratio:.2%}")
    logger.info(f"Total latency: {metrics.total_latency_ms:.2f}ms")
    logger.info(f"Avg edge draft time: {metrics.total_edge_draft_time_ms / metrics.total_rounds:.2f}ms")
    logger.info(f"Avg server verify time: {metrics.total_server_verify_time_ms / metrics.total_rounds:.2f}ms")
    
    logger.info("\n✓ Test complete!")

if __name__ == "__main__":
    main()
