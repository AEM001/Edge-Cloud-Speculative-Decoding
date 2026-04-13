"""
Metrics logger for speculative decoding benchmarks.

Logs per-request and per-condition metrics as specified in the architecture.
"""
import json
import csv
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import statistics

from ..client.edge_client import RequestMetrics


logger = logging.getLogger(__name__)


@dataclass
class ConditionMetrics:
    """
    Per-condition metrics aggregated across all requests.
    
    Metrics as specified:
    - mean latency
    - p50 latency
    - p95 latency
    - mean TPOT
    - mean tokens/sec
    - mean acceptance ratio
    - mean wasted drafted tokens
    - variance of acceptance ratio across prompts
    """
    condition_name: str
    policy_name: str
    network_regime: str
    workload_group: str
    
    # Latency metrics
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    
    # Throughput metrics
    mean_tpot_ms: float
    mean_tokens_per_sec: float
    
    # Acceptance metrics
    mean_acceptance_ratio: float
    mean_wasted_tokens: float
    variance_acceptance_ratio: float
    
    # Additional stats
    num_requests: int
    total_tokens: int
    total_rounds: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsLogger:
    """
    Logger for benchmark metrics.
    
    Collects per-request metrics and aggregates them per condition.
    Exports results to JSON and CSV.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize metrics logger.
        
        Args:
            output_dir: Directory to save metric files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store per-request metrics
        self.request_metrics: List[RequestMetrics] = []
        
        # Store per-condition metrics
        self.condition_metrics: List[ConditionMetrics] = []
        
        # Metadata
        self.start_time = datetime.now()
        self.run_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"MetricsLogger initialized with output_dir: {self.output_dir}")
    
    def log_request(self, metrics: RequestMetrics):
        """
        Log per-request metrics.
        
        Args:
            metrics: RequestMetrics from a single generation
        """
        self.request_metrics.append(metrics)
        logger.debug(f"Logged request metrics: {metrics.request_id}")
    
    def aggregate_condition(
        self,
        condition_name: str,
        policy_name: str,
        network_regime: str,
        workload_group: str,
        request_filter: callable = None
    ) -> ConditionMetrics:
        """
        Aggregate metrics for a specific condition.
        
        Args:
            condition_name: Name of the condition
            policy_name: Policy used
            network_regime: Network regime
            workload_group: Workload group ('easy' or 'hard')
            request_filter: Optional function to filter requests
        
        Returns:
            ConditionMetrics with aggregated statistics
        """
        # Filter requests
        if request_filter:
            filtered = [m for m in self.request_metrics if request_filter(m)]
        else:
            filtered = self.request_metrics
        
        if not filtered:
            logger.warning(f"No requests found for condition: {condition_name}")
            return None
        
        # Extract latency values
        latencies = [m.total_latency_ms for m in filtered]
        
        # Compute TPOT (Time Per Output Token)
        tpots = []
        for m in filtered:
            if m.generated_tokens > 0:
                tpot = m.total_latency_ms / m.generated_tokens
                tpots.append(tpot)
        
        # Compute tokens per second
        tokens_per_sec = []
        for m in filtered:
            if m.total_latency_ms > 0:
                tps = (m.generated_tokens / m.total_latency_ms) * 1000
                tokens_per_sec.append(tps)
        
        # Acceptance ratios
        acceptance_ratios = [m.acceptance_ratio for m in filtered]
        
        # Wasted tokens
        wasted_tokens = [m.wasted_drafted_tokens for m in filtered]
        
        # Compute statistics
        mean_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        
        mean_tpot = statistics.mean(tpots) if tpots else 0
        mean_tokens_per_sec = statistics.mean(tokens_per_sec) if tokens_per_sec else 0
        
        mean_acceptance = statistics.mean(acceptance_ratios) if acceptance_ratios else 0
        mean_wasted = statistics.mean(wasted_tokens) if wasted_tokens else 0
        
        variance_acceptance = statistics.variance(acceptance_ratios) if len(acceptance_ratios) > 1 else 0
        
        total_tokens = sum(m.generated_tokens for m in filtered)
        total_rounds = sum(m.total_rounds for m in filtered)
        
        condition = ConditionMetrics(
            condition_name=condition_name,
            policy_name=policy_name,
            network_regime=network_regime,
            workload_group=workload_group,
            mean_latency_ms=mean_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            mean_tpot_ms=mean_tpot,
            mean_tokens_per_sec=mean_tokens_per_sec,
            mean_acceptance_ratio=mean_acceptance,
            mean_wasted_tokens=mean_wasted,
            variance_acceptance_ratio=variance_acceptance,
            num_requests=len(filtered),
            total_tokens=total_tokens,
            total_rounds=total_rounds
        )
        
        self.condition_metrics.append(condition)
        logger.info(f"Aggregated condition metrics: {condition_name}")
        
        return condition
    
    def export_json(self, filename: str = None):
        """
        Export all metrics to JSON.
        
        Args:
            filename: Optional filename (default: metrics_<run_id>.json)
        """
        if filename is None:
            filename = f"metrics_{self.run_id}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "num_requests": len(self.request_metrics),
            "request_metrics": [asdict(m) for m in self.request_metrics],
            "condition_metrics": [c.to_dict() for c in self.condition_metrics]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported metrics to JSON: {filepath}")
    
    def export_csv(self, filename: str = None):
        """
        Export condition metrics to CSV.
        
        Args:
            filename: Optional filename (default: condition_metrics_<run_id>.csv)
        """
        if filename is None:
            filename = f"condition_metrics_{self.run_id}.csv"
        
        filepath = self.output_dir / filename
        
        if not self.condition_metrics:
            logger.warning("No condition metrics to export")
            return
        
        fieldnames = list(self.condition_metrics[0].to_dict().keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for condition in self.condition_metrics:
                writer.writerow(condition.to_dict())
        
        logger.info(f"Exported condition metrics to CSV: {filepath}")
    
    def export_detailed_csv(self, filename: str = None):
        """
        Export per-request metrics to CSV.
        
        Args:
            filename: Optional filename (default: request_metrics_<run_id>.csv)
        """
        if filename is None:
            filename = f"request_metrics_{self.run_id}.csv"
        
        filepath = self.output_dir / filename
        
        if not self.request_metrics:
            logger.warning("No request metrics to export")
            return
        
        fieldnames = list(asdict(self.request_metrics[0]).keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in self.request_metrics:
                writer.writerow(asdict(metrics))
        
        logger.info(f"Exported request metrics to CSV: {filepath}")
    
    def print_summary(self):
        """Print a summary of all condition metrics."""
        if not self.condition_metrics:
            logger.warning("No condition metrics to summarize")
            return
        
        print("\n" + "="*80)
        print("CONDITION METRICS SUMMARY")
        print("="*80)
        
        for condition in self.condition_metrics:
            print(f"\nCondition: {condition.condition_name}")
            print(f"  Policy: {condition.policy_name}")
            print(f"  Network: {condition.network_regime}")
            print(f"  Workload: {condition.workload_group}")
            print(f"  Requests: {condition.num_requests}")
            print(f"  Mean Latency: {condition.mean_latency_ms:.2f}ms")
            print(f"  P50 Latency: {condition.p50_latency_ms:.2f}ms")
            print(f"  P95 Latency: {condition.p95_latency_ms:.2f}ms")
            print(f"  Mean TPOT: {condition.mean_tpot_ms:.2f}ms")
            print(f"  Mean Tokens/sec: {condition.mean_tokens_per_sec:.2f}")
            print(f"  Mean Acceptance: {condition.mean_acceptance_ratio:.2%}")
            print(f"  Mean Wasted: {condition.mean_wasted_tokens:.2f}")
            print(f"  Acceptance Variance: {condition.variance_acceptance_ratio:.4f}")
        
        print("\n" + "="*80)
