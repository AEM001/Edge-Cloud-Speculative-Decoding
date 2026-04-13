"""
Main experiment runner for static K baseline.

This script runs the full experiment matrix:
- 3 policies: K=2, K=4, K=6
- 2 workload groups: Easy, Hard
- 4 network regimes: Good, Medium, Bad, Bursty

Total: 3 × 2 × 4 = 24 conditions
"""
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

from config import MODEL_NAME, MODEL_PATH, TEMPERATURE
from model_manager import ModelManager
from client.draft_generator import DraftGenerator
from client.edge_client import EdgeClient
from client.http_cloud_client import create_http_cloud_client
from client.network_wrapper import create_network_wrapped_client
from benchmarks.workloads import load_workloads
from benchmarks.metrics_logger import MetricsLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Static K policies
def static_k_policy(k_value: int):
    """Create a static K policy function."""
    def policy(round_id: int, draft_tokens: List) -> int:
        return k_value
    return policy


class ExperimentRunner:
    """Runs the full experiment matrix."""
    
    def __init__(
        self,
        cloud_server_url: str,
        output_dir: str = "results",
        max_new_tokens: int = 128,
        enable_network_simulation: bool = True,
        num_prompts: int = None
    ):
        """
        Initialize experiment runner.
        
        Args:
            cloud_server_url: URL of cloud verification server
            output_dir: Directory to save results
            max_new_tokens: Maximum tokens to generate per request
            enable_network_simulation: Enable network simulation
        """
        self.cloud_server_url = cloud_server_url
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens
        self.enable_network_simulation = enable_network_simulation
        self.num_prompts = num_prompts
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model manager and draft generator
        logger.info("Initializing draft model...")
        self.model_manager = ModelManager(
            model_name=MODEL_NAME,
            model_path=MODEL_PATH
        )
        model, tokenizer = self.model_manager.load()
        
        self.draft_generator = DraftGenerator(model, tokenizer)
        logger.info("Draft model initialized")
        
        # Create base HTTP client
        self.base_http_client = create_http_cloud_client(
            server_url=cloud_server_url,
            timeout=60.0,
            retry_attempts=3
        )
        
        # Check server health
        if not self.base_http_client.check_health():
            logger.warning("Cloud server health check failed!")
        
        # Load workloads
        logger.info("Loading workloads...")
        self.workloads = load_workloads()
        logger.info(f"Loaded {len(self.workloads['easy'])} easy prompts, {len(self.workloads['hard'])} hard prompts")
        
        # Truncate prompts if num_prompts is set
        if self.num_prompts:
            self.workloads['easy'] = self.workloads['easy'][:self.num_prompts]
            self.workloads['hard'] = self.workloads['hard'][:self.num_prompts]
            logger.info(f"Truncated to {self.num_prompts} prompts per workload")
    
    def run_full_experiment(self):
        """Run the full experiment matrix."""
        logger.info("=" * 80)
        logger.info("STARTING FULL EXPERIMENT")
        logger.info("=" * 80)
        
        # Define experiment matrix
        k_values = [2, 4, 6]
        workload_groups = ["easy", "hard"]
        network_regimes = ["good", "medium", "bad", "bursty"]
        
        total_conditions = len(k_values) * len(workload_groups) * len(network_regimes)
        logger.info(f"Total conditions: {total_conditions}")
        
        # Create experiment metadata
        experiment_metadata = {
            "start_time": datetime.now().isoformat(),
            "cloud_server_url": self.cloud_server_url,
            "max_new_tokens": self.max_new_tokens,
            "network_simulation_enabled": self.enable_network_simulation,
            "k_values": k_values,
            "workload_groups": workload_groups,
            "network_regimes": network_regimes,
            "total_conditions": total_conditions,
            "draft_model": MODEL_NAME,
        }
        
        # Save metadata
        metadata_path = self.output_dir / "experiment_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(experiment_metadata, f, indent=2)
        
        # Run all conditions
        condition_id = 0
        all_results = []
        
        for k in k_values:
            for workload_group in workload_groups:
                for network_regime in network_regimes:
                    condition_id += 1
                    
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info(f"CONDITION {condition_id}/{total_conditions}")
                    logger.info(f"K={k}, Workload={workload_group}, Network={network_regime}")
                    logger.info("=" * 80)
                    
                    # Run condition
                    results = self.run_condition(
                        k_value=k,
                        workload_group=workload_group,
                        network_regime=network_regime,
                        condition_id=condition_id
                    )
                    
                    all_results.append({
                        "condition_id": condition_id,
                        "k": k,
                        "workload": workload_group,
                        "network": network_regime,
                        "results": results
                    })
                    
                    logger.info(f"Condition {condition_id} complete")
        
        # Save all results
        results_path = self.output_dir / "all_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 80)
    
    def run_condition(
        self,
        k_value: int,
        workload_group: str,
        network_regime: str,
        condition_id: int
    ) -> Dict[str, Any]:
        """
        Run a single experimental condition.
        
        Args:
            k_value: Static K value
            workload_group: "easy" or "hard"
            network_regime: "good", "medium", "bad", or "bursty"
            condition_id: Unique condition identifier
        
        Returns:
            Dictionary with condition results
        """
        # Create network-wrapped client
        cloud_client = create_network_wrapped_client(
            cloud_client=self.base_http_client,
            regime=network_regime,
            enable_simulation=self.enable_network_simulation
        )
        
        # Create edge client
        edge_client = EdgeClient(
            model_manager=self.model_manager,
            draft_generator=self.draft_generator,
            cloud_client=cloud_client,
            max_new_tokens=self.max_new_tokens,
            temperature=TEMPERATURE
        )
        
        # Create policy
        policy = static_k_policy(k_value)
        policy_name = f"StaticK{k_value}"
        
        # Get prompts for this workload
        prompts = self.workloads[workload_group]
        
        # Create metrics logger
        condition_name = f"k{k_value}_{workload_group}_{network_regime}"
        metrics_logger = MetricsLogger(
            output_dir=str(self.output_dir / condition_name)
        )
        
        # Run all prompts
        logger.info(f"Running {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts):
            logger.info(f"  Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
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
                    f"    Generated {metrics.generated_tokens} tokens in {metrics.total_rounds} rounds, "
                    f"acceptance={metrics.acceptance_ratio:.2%}, "
                    f"latency={metrics.total_latency_ms:.0f}ms"
                )
                
            except Exception as e:
                logger.error(f"    Failed: {e}")
                continue
        
        # Compute and save aggregate metrics
        condition_result = metrics_logger.aggregate_condition(
            condition_name=condition_name,
            policy_name=policy_name,
            network_regime=network_regime,
            workload_group=workload_group
        )
        metrics_logger.export_json()
        metrics_logger.export_csv()
        metrics_logger.export_detailed_csv()
        
        if condition_result is None:
            logger.warning(f"No results for condition {condition_name}")
            return {}
        
        aggregate_metrics = condition_result.to_dict()
        
        logger.info(f"Condition summary:")
        logger.info(f"  Mean latency: {aggregate_metrics['mean_latency_ms']:.2f}ms")
        logger.info(f"  Mean acceptance ratio: {aggregate_metrics['mean_acceptance_ratio']:.2%}")
        logger.info(f"  Mean tokens/sec: {aggregate_metrics['mean_tokens_per_sec']:.2f}")
        
        return aggregate_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run static K baseline experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="Cloud server URL (e.g., http://49.234.57.210:8000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--no-network-simulation",
        action="store_true",
        help="Disable network simulation (use real network only)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of prompts per workload (default: all)"
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test connection to server and exit"
    )
    
    args = parser.parse_args()
    
    # Test connection if requested
    if args.test_connection:
        logger.info(f"Testing connection to {args.server_url}...")
        client = create_http_cloud_client(args.server_url)
        if client.check_health():
            logger.info("✓ Connection successful!")
            sys.exit(0)
        else:
            logger.error("✗ Connection failed!")
            sys.exit(1)
    
    # Run experiment
    runner = ExperimentRunner(
        cloud_server_url=args.server_url,
        output_dir=args.output_dir,
        max_new_tokens=args.max_tokens,
        enable_network_simulation=not args.no_network_simulation,
        num_prompts=args.num_prompts
    )
    
    runner.run_full_experiment()


if __name__ == "__main__":
    main()
