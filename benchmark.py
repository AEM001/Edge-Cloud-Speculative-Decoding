"""
Main benchmark runner for speculative decoding.

This script runs the full benchmark suite:
- Tests all policies (Static K: 2, 4, 6)
- Tests all network regimes (good, medium, bad, bursty)
- Tests all workload groups (easy, hard)
- Logs all metrics

Usage:
    python benchmark.py [--policy POLICY] [--regime REGIME] [--workload WORKLOAD]
"""
import logging
import argparse
from pathlib import Path

from config import (
    MODEL_NAME, MODEL_PATH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P,
    STATIC_K_VALUES, NETWORK_REGIMES, WORKLOAD_GROUPS,
    MOCK_ACCEPTANCE_RATE, MOCK_VERIFY_TIME_MS,
    OUTPUT_DIR, METRICS_DIR, LOG_LEVEL
)
from client.model_manager import ModelManager
from client.draft_generator import DraftGenerator
from client.edge_client import EdgeClient
from server.cloud_server_interface import create_mock_cloud_client
from policies.policies import get_policy
from benchmarks.workloads import get_prompts_by_group
from benchmarks.network_simulator import NetworkSimulator, get_network_regime
from benchmarks.metrics_logger import MetricsLogger


def setup_logging():
    """Configure logging."""
    level = getattr(logging, LOG_LEVEL.upper())
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_benchmark(
    policies: list,
    regimes: list,
    workload_groups: list,
    use_mock: bool = True
):
    """
    Run the full benchmark suite.
    
    Args:
        policies: List of policy names to test
        regimes: List of network regimes to test
        workload_groups: List of workload groups to test
        use_mock: Whether to use mock cloud server (for local testing)
    """
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("STARTING SPECULATIVE DECODING BENCHMARK")
    logger.info("="*80)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(METRICS_DIR)
    
    # Load draft model
    logger.info(f"Loading draft model: {MODEL_NAME}")
    model_manager = ModelManager(model_name=MODEL_NAME, model_path=MODEL_PATH)
    model, tokenizer = model_manager.load()
    
    # Initialize draft generator
    draft_generator = DraftGenerator(model, tokenizer)
    
    # Initialize cloud client
    if use_mock:
        logger.warning("Using MOCK cloud server for local testing.")
        logger.warning("Implement actual cloud server on Ubuntu with 3060 GPU for real benchmarks.")
        cloud_client = create_mock_cloud_client(base_acceptance_rate=MOCK_ACCEPTANCE_RATE)
    else:
        logger.error("Real cloud client not yet implemented. Use --mock for local testing.")
        raise NotImplementedError("Real cloud client requires Ubuntu implementation.")
    
    # Get EOS token ID
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
    
    # Run benchmark for each combination
    total_conditions = len(policies) * len(regimes) * len(workload_groups)
    condition_num = 0
    
    for policy_name in policies:
        for regime_name in regimes:
            for workload_group in workload_groups:
                condition_num += 1
                logger.info("\n" + "="*80)
                logger.info(f"CONDITION {condition_num}/{total_conditions}")
                logger.info(f"Policy: {policy_name}, Regime: {regime_name}, Workload: {workload_group}")
                logger.info("="*80)
                
                # Get policy function
                policy = get_policy(policy_name)
                
                # Create network simulator
                regime = get_network_regime(regime_name)
                network_sim = NetworkSimulator(regime)
                
                # Wrap cloud client with network simulation
                cloud_client_with_network = network_sim.wrap_cloud_client(cloud_client)
                
                # Initialize edge client
                edge_client = EdgeClient(
                    model_manager=model_manager,
                    draft_generator=draft_generator,
                    cloud_client=cloud_client_with_network,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    eos_token_id=eos_token_id
                )
                
                # Get prompts for this workload group
                prompts = get_prompts_by_group(workload_group)
                logger.info(f"Running {len(prompts)} prompts")
                
                # Run each prompt
                for i, prompt in enumerate(prompts):
                    logger.info(f"\n[{i+1}/{len(prompts)}] Running prompt: {prompt[:50]}...")
                    
                    try:
                        # Generate with speculative decoding
                        metrics = edge_client.generate(
                            prompt=prompt,
                            policy=policy,
                            policy_name=policy_name
                        )
                        
                        # Log metrics
                        metrics_logger.log_request(metrics)
                        
                        logger.info(
                            f"  Tokens: {metrics.generated_tokens}, "
                            f"Rounds: {metrics.total_rounds}, "
                            f"Acceptance: {metrics.acceptance_ratio:.2%}, "
                            f"Latency: {metrics.total_latency_ms:.2f}ms"
                        )
                        
                    except Exception as e:
                        logger.error(f"Error running prompt {i+1}: {e}")
                        continue
                
                # Aggregate condition metrics
                condition_name = f"{policy_name}_{regime_name}_{workload_group}"
                metrics_logger.aggregate_condition(
                    condition_name=condition_name,
                    policy_name=policy_name,
                    network_regime=regime_name,
                    workload_group=workload_group
                )
    
    # Export all metrics
    logger.info("\n" + "="*80)
    logger.info("EXPORTING METRICS")
    logger.info("="*80)
    
    metrics_logger.export_json()
    metrics_logger.export_csv()
    metrics_logger.export_detailed_csv()
    
    # Print summary
    metrics_logger.print_summary()
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"Results saved to: {METRICS_DIR}")
    logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmark")
    parser.add_argument(
        "--policy",
        type=str,
        choices=["static_k_2", "static_k_4", "static_k_6", "all"],
        default="all",
        help="Policy to test (default: all)"
    )
    parser.add_argument(
        "--regime",
        type=str,
        choices=["good", "medium", "bad", "bursty", "all"],
        default="all",
        help="Network regime to test (default: all)"
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=["easy", "hard", "all"],
        default="all",
        help="Workload group to test (default: all)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock cloud server for local testing (default: True)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Determine which items to test
    policies = STATIC_K_VALUES if args.policy == "all" else [args.policy]
    regimes = NETWORK_REGIMES if args.regime == "all" else [args.regime]
    workload_groups = WORKLOAD_GROUPS if args.workload == "all" else [args.workload]
    
    # Convert K values to policy names
    if args.policy == "all":
        policies = [f"static_k_{k}" for k in STATIC_K_VALUES]
    
    run_benchmark(
        policies=policies,
        regimes=regimes,
        workload_groups=workload_groups,
        use_mock=args.mock
    )


if __name__ == "__main__":
    main()
