"""Main entry point for speculative decoding edge node on Ubuntu 3060."""
import logging
from config import MODEL_NAME, MODEL_PATH, MAX_DRAFT_TOKENS, TEMPERATURE, TOP_P, VERBOSE
from model_manager import ModelManager
from client.draft_generator import DraftGenerator
from protocol import DraftRequest

def setup_logging():
    """Configure logging."""
    level = logging.DEBUG if VERBOSE else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Initialize and run the draft model server."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== PyTorch Speculative Decoding Edge Node (Ubuntu 3060) ===")
    
    # Initialize model manager with custom storage and quantization
    manager = ModelManager(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        quantization=True  # Enable 4-bit quantization
    )
    
    # Load model to specified path
    model, tokenizer = manager.load()
    
    logger.info(f"Model info: {manager.get_model_info()}")
    
    # Initialize draft generator
    generator = DraftGenerator(model, tokenizer)
    
    # Test generation
    logger.info("\n--- Test Draft Generation ---")
    
    # Test with a simple prompt
    test_prompt = "Hello, how are you?"
    test_tokens = tokenizer.encode(test_prompt)
    
    logger.info(f"Test prompt: {test_prompt!r}")
    logger.info(f"Prompt tokens: {test_tokens}")
    
    # Create draft request
    request = DraftRequest(
        verified_prefix=test_tokens,
        num_draft_tokens=MAX_DRAFT_TOKENS
    )
    
    # Generate draft tokens
    response = generator.generate_draft_tokens(
        request,
        temperature=TEMPERATURE,
        top_p=TOP_P
    )
    
    # Display results
    logger.info(f"\nGenerated {len(response.draft_token_ids)} draft tokens:")
    
    for i, (token_id, prob, logprob) in enumerate(zip(
        response.draft_token_ids,
        response.probabilities,
        response.logprobs
    )):
        token_text = tokenizer.decode([token_id])
        stats = response.confidence_stats
        logger.info(f"  [{i}] token={token_id}, text={token_text!r}")
        logger.info(f"       prob={prob:.4f}, logprob={logprob:.4f}")
        logger.info(f"       max_prob={stats['max_probs'][i]:.4f}, "
                   f"entropy={stats['entropies'][i]:.4f}, "
                   f"top_margin={stats['top_margins'][i]:.4f}")
    
    # Full text of draft tokens
    draft_text = tokenizer.decode(response.draft_token_ids)
    logger.info(f"\nDraft text: {draft_text!r}")
    
    logger.info("\n=== Ready for speculative decoding requests ===")
    
    return generator, manager


if __name__ == "__main__":
    generator, manager = main()
