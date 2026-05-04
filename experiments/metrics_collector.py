"""
Unified metrics collection, normalisation, and export for network experiments.

Responsibilities
----------------
- Define ``ExperimentResult`` — one flat record per (method, network, prompt).
- Convert raw EdgeClient / AsyncEdgeClient metrics → ExperimentResult.
- Aggregate results across prompts → ConditionSummary.
- Export to JSON and CSV.
- Print a human-readable comparison table.

No model loading, no network simulation — purely data wrangling.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single flat result record
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """One observation: one method × one network condition × one prompt."""

    # --- Identity ----------------------------------------------------------
    method: str                   # e.g. "direct", "spec_k3", "async_k5"
    mode: str                     # "direct" | "speculative" | "async"
    k_value: Optional[int]        # draft length K (None for direct)
    network_condition: str        # e.g. "WAN_High"
    prompt_id: int
    prompt_type: str              # "Simple" | "Complex"

    # --- Core throughput ---------------------------------------------------
    tokens_generated: int = 0
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0

    # --- Acceptance / efficiency -------------------------------------------
    acceptance_ratio: float = 0.0
    total_rounds: int = 0
    mean_k_chosen: float = 0.0

    # --- Timing breakdown (ms totals) --------------------------------------
    total_draft_time_ms: float = 0.0
    total_verify_time_ms: float = 0.0
    total_network_time_ms: float = 0.0
    average_rtt_ms: float = 0.0

    # --- Network payload ---------------------------------------------------
    uplink_bytes: int = 0
    downlink_bytes: int = 0

    # --- Async-only pipeline metrics --------------------------------------
    pipeline_efficiency: float = 0.0    # fraction of full-hit slots
    avg_bubble_ms: float = 0.0          # avg drafter stall per round
    prefetch_waste_ratio: float = 0.0   # wasted speculative tokens
    async_speedup_vs_sync: float = 0.0  # theoretical speedup factor
    total_rollbacks: int = 0

    # --- Simulated network overhead (from ThrottledCloudClient) -----------
    simulated_overhead_ms: float = 0.0
    simulated_uplink_delay_ms: float = 0.0
    simulated_downlink_delay_ms: float = 0.0

    # --- Error flag --------------------------------------------------------
    error: bool = False
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Converters from raw client metrics
# ---------------------------------------------------------------------------

def from_direct(
    tokens: int,
    latency_ms: float,
    network_condition: str,
    prompt_id: int,
    prompt_type: str,
    simulated_overhead_ms: float = 0.0,
    simulated_uplink_delay_ms: float = 0.0,
    simulated_downlink_delay_ms: float = 0.0,
    error: bool = False,
    error_message: str = "",
) -> ExperimentResult:
    """Build an ExperimentResult from a direct generation call."""
    tps = tokens / (latency_ms / 1000.0) if latency_ms > 0 and tokens > 0 else 0.0
    return ExperimentResult(
        method="direct",
        mode="direct",
        k_value=None,
        network_condition=network_condition,
        prompt_id=prompt_id,
        prompt_type=prompt_type,
        tokens_generated=tokens,
        total_latency_ms=latency_ms,
        tokens_per_second=tps,
        simulated_overhead_ms=simulated_overhead_ms,
        simulated_uplink_delay_ms=simulated_uplink_delay_ms,
        simulated_downlink_delay_ms=simulated_downlink_delay_ms,
        error=error,
        error_message=error_message,
    )


def from_sync_metrics(
    raw,          # client.edge_client.RequestMetrics
    k: int,
    network_condition: str,
    prompt_id: int,
    prompt_type: str,
    simulated_overhead_ms: float = 0.0,
    simulated_uplink_delay_ms: float = 0.0,
    simulated_downlink_delay_ms: float = 0.0,
    error: bool = False,
    error_message: str = "",
) -> ExperimentResult:
    """Build from synchronous EdgeClient.RequestMetrics."""
    tps = (
        raw.generated_tokens / (raw.total_latency_ms / 1000.0)
        if raw.total_latency_ms > 0 and raw.generated_tokens > 0
        else 0.0
    )
    return ExperimentResult(
        method=f"spec_k{k}",
        mode="speculative",
        k_value=k,
        network_condition=network_condition,
        prompt_id=prompt_id,
        prompt_type=prompt_type,
        tokens_generated=raw.generated_tokens,
        total_latency_ms=raw.total_latency_ms,
        tokens_per_second=tps,
        acceptance_ratio=raw.acceptance_ratio,
        total_rounds=raw.total_rounds,
        mean_k_chosen=raw.mean_K_chosen,
        total_draft_time_ms=raw.total_edge_draft_time_ms,
        total_verify_time_ms=raw.total_server_verify_time_ms,
        total_network_time_ms=raw.total_network_time_ms,
        average_rtt_ms=raw.average_rtt_ms,
        uplink_bytes=raw.uplink_bytes,
        downlink_bytes=raw.downlink_bytes,
        simulated_overhead_ms=simulated_overhead_ms,
        simulated_uplink_delay_ms=simulated_uplink_delay_ms,
        simulated_downlink_delay_ms=simulated_downlink_delay_ms,
        error=error,
        error_message=error_message,
    )


def from_async_metrics(
    raw,          # client.async_edge_client.AsyncRequestMetrics
    k: int,
    network_condition: str,
    prompt_id: int,
    prompt_type: str,
    simulated_overhead_ms: float = 0.0,
    simulated_uplink_delay_ms: float = 0.0,
    simulated_downlink_delay_ms: float = 0.0,
    error: bool = False,
    error_message: str = "",
) -> ExperimentResult:
    """Build from AsyncEdgeClient.AsyncRequestMetrics."""
    return ExperimentResult(
        method=f"async_k{k}",
        mode="async",
        k_value=k,
        network_condition=network_condition,
        prompt_id=prompt_id,
        prompt_type=prompt_type,
        tokens_generated=raw.generated_tokens,
        total_latency_ms=raw.total_latency_ms,
        tokens_per_second=raw.tokens_per_second,
        acceptance_ratio=raw.acceptance_ratio,
        total_rounds=raw.total_rounds,
        mean_k_chosen=raw.mean_k_chosen,
        total_draft_time_ms=raw.total_edge_draft_time_ms,
        total_verify_time_ms=raw.total_server_verify_time_ms,
        total_network_time_ms=raw.total_network_time_ms,
        average_rtt_ms=raw.average_rtt_ms,
        uplink_bytes=raw.uplink_bytes,
        downlink_bytes=raw.downlink_bytes,
        pipeline_efficiency=raw.pipeline_efficiency,
        avg_bubble_ms=raw.avg_bubble_ms,
        prefetch_waste_ratio=raw.prefetch_waste_ratio,
        async_speedup_vs_sync=raw.async_speedup_vs_sync,
        total_rollbacks=raw.total_rollbacks,
        simulated_overhead_ms=simulated_overhead_ms,
        simulated_uplink_delay_ms=simulated_uplink_delay_ms,
        simulated_downlink_delay_ms=simulated_downlink_delay_ms,
        error=error,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0

def _stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


@dataclass
class MethodSummary:
    """Aggregated statistics for one method under one network condition."""
    method: str
    network_condition: str
    n_samples: int = 0

    mean_tps: float = 0.0
    stdev_tps: float = 0.0
    mean_latency_ms: float = 0.0
    mean_acceptance_ratio: float = 0.0
    mean_rounds: float = 0.0
    mean_rtt_ms: float = 0.0
    mean_draft_ms: float = 0.0
    mean_verify_ms: float = 0.0
    mean_net_ms: float = 0.0
    mean_pipeline_efficiency: float = 0.0
    mean_avg_bubble_ms: float = 0.0
    mean_prefetch_waste_ratio: float = 0.0
    mean_async_speedup: float = 0.0
    mean_simulated_overhead_ms: float = 0.0
    speedup_vs_direct: float = 0.0      # filled in post-hoc


def aggregate(
    results: List[ExperimentResult],
    method: str,
    condition: str,
) -> MethodSummary:
    """Aggregate matching results into a MethodSummary."""
    subset = [r for r in results if r.method == method and r.network_condition == condition and not r.error]
    s = MethodSummary(method=method, network_condition=condition, n_samples=len(subset))
    if not subset:
        return s

    s.mean_tps = _mean([r.tokens_per_second for r in subset])
    s.stdev_tps = _stdev([r.tokens_per_second for r in subset])
    s.mean_latency_ms = _mean([r.total_latency_ms for r in subset])
    s.mean_acceptance_ratio = _mean([r.acceptance_ratio for r in subset])
    s.mean_rounds = _mean([r.total_rounds for r in subset])
    s.mean_rtt_ms = _mean([r.average_rtt_ms for r in subset])
    s.mean_draft_ms = _mean([r.total_draft_time_ms for r in subset])
    s.mean_verify_ms = _mean([r.total_verify_time_ms for r in subset])
    s.mean_net_ms = _mean([r.total_network_time_ms for r in subset])
    s.mean_pipeline_efficiency = _mean([r.pipeline_efficiency for r in subset])
    s.mean_avg_bubble_ms = _mean([r.avg_bubble_ms for r in subset])
    s.mean_prefetch_waste_ratio = _mean([r.prefetch_waste_ratio for r in subset])
    s.mean_async_speedup = _mean([r.async_speedup_vs_sync for r in subset])
    s.mean_simulated_overhead_ms = _mean([r.simulated_overhead_ms for r in subset])
    return s


# ---------------------------------------------------------------------------
# MetricsCollector — central store
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Accumulates ExperimentResults throughout an experiment run and provides
    export / reporting utilities.
    """

    def __init__(self):
        self._results: List[ExperimentResult] = []

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def add(self, result: ExperimentResult) -> None:
        self._results.append(result)
        status = "ERROR" if result.error else "OK"
        logger.info(
            "[%s] %s | net=%s | prompt=%d | %d tok | %.2f tok/s | "
            "accept=%.1f%% | rtt=%.0f ms | sim_overhead=%.0f ms",
            status,
            result.method,
            result.network_condition,
            result.prompt_id,
            result.tokens_generated,
            result.tokens_per_second,
            result.acceptance_ratio * 100,
            result.average_rtt_ms,
            result.simulated_overhead_ms,
        )

    def all_results(self) -> List[ExperimentResult]:
        return list(self._results)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self._results], f, indent=2)
        logger.info("Results saved → %s (%d records)", path, len(self._results))

    def save_csv(self, path: Path) -> None:
        if not self._results:
            logger.warning("No results to save to CSV")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self._results[0].to_dict().keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([r.to_dict() for r in self._results])
        logger.info("CSV saved → %s", path)

    def save_summary_json(self, path: Path) -> None:
        """Save aggregated per-(method, condition) summary."""
        summaries = self._build_summaries()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(s) for s in summaries], f, indent=2)
        logger.info("Summary saved → %s", path)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_table(self, conditions: Optional[List[str]] = None) -> None:
        """
        Print a human-readable comparison table grouped by network condition.

        For each condition rows are:
          method | tok/s | latency | accept | rtt | pipeline_eff | overhead
        """
        summaries = self._build_summaries()
        all_conditions = sorted({s.network_condition for s in summaries})
        if conditions:
            all_conditions = [c for c in all_conditions if c in conditions]

        method_order = _canonical_method_order(self._results)

        header = (
            f"{'Method':<14} {'tok/s':>7} {'±':>5} "
            f"{'lat(ms)':>9} {'accept%':>8} "
            f"{'rtt(ms)':>8} {'net(ms)':>8} "
            f"{'pipe_eff%':>10} {'overhead(ms)':>13}"
        )
        sep = "-" * len(header)

        for cond in all_conditions:
            print(f"\n{'═'*len(header)}")
            print(f"Network: {cond}")
            print(header)
            print(sep)
            cond_summaries = {s.method: s for s in summaries if s.network_condition == cond}
            for method in method_order:
                if method not in cond_summaries:
                    continue
                s = cond_summaries[method]
                speedup_str = f"  ({s.speedup_vs_direct:.2f}x)" if s.speedup_vs_direct > 0 else ""
                pipe_eff = f"{s.mean_pipeline_efficiency*100:.1f}" if s.mean_pipeline_efficiency > 0 else "  —"
                print(
                    f"{method:<14} "
                    f"{s.mean_tps:>7.2f} "
                    f"{s.stdev_tps:>5.2f} "
                    f"{s.mean_latency_ms:>9.0f} "
                    f"{s.mean_acceptance_ratio*100:>7.1f}% "
                    f"{s.mean_rtt_ms:>8.1f} "
                    f"{s.mean_net_ms:>8.1f} "
                    f"{pipe_eff:>10} "
                    f"{s.mean_simulated_overhead_ms:>13.1f}"
                    f"{speedup_str}"
                )

    def print_metric_spotlight(self) -> None:
        """
        Print focused views on key metrics across network conditions:
          1. Acceptance ratio by K and condition
          2. Pipeline efficiency (async) by K and condition
          3. Simulated network overhead impact
        """
        summaries = self._build_summaries()
        all_conditions = sorted({s.network_condition for s in summaries})

        # --- Acceptance ratio table ----------------------------------------
        spec_methods = sorted({s.method for s in summaries if s.method.startswith("spec_k")})
        if spec_methods:
            print("\n── Acceptance Ratio by K and Network Condition ──")
            header = f"{'Condition':<14}" + "".join(f"{m:>12}" for m in spec_methods)
            print(header)
            print("-" * len(header))
            cond_map = {s.network_condition: {} for s in summaries}
            for s in summaries:
                cond_map[s.network_condition][s.method] = s
            for cond in all_conditions:
                row = f"{cond:<14}"
                for m in spec_methods:
                    val = cond_map[cond].get(m)
                    row += f"{val.mean_acceptance_ratio*100:>11.1f}%" if val else f"{'—':>12}"
                print(row)

        # --- Pipeline efficiency table ------------------------------------
        async_methods = sorted({s.method for s in summaries if s.method.startswith("async_k")})
        if async_methods:
            print("\n── Pipeline Efficiency (Async) by K and Network Condition ──")
            header = f"{'Condition':<14}" + "".join(f"{m:>12}" for m in async_methods)
            print(header)
            print("-" * len(header))
            for cond in all_conditions:
                row = f"{cond:<14}"
                for m in async_methods:
                    val = cond_map[cond].get(m)
                    row += f"{val.mean_pipeline_efficiency*100:>11.1f}%" if val else f"{'—':>12}"
                print(row)

        # --- Speedup vs direct table -------------------------------------
        all_spec = sorted({s.method for s in summaries if s.method != "direct"})
        if all_spec:
            print("\n── Speedup vs Direct by Method and Network Condition ──")
            header = f"{'Condition':<14}" + "".join(f"{m:>14}" for m in all_spec)
            print(header)
            print("-" * len(header))
            for cond in all_conditions:
                row = f"{cond:<14}"
                for m in all_spec:
                    val = cond_map[cond].get(m)
                    row += f"{val.speedup_vs_direct:>13.2f}x" if val else f"{'—':>14}"
                print(row)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_summaries(self) -> List[MethodSummary]:
        methods = sorted({r.method for r in self._results})
        conditions = sorted({r.network_condition for r in self._results})
        summaries: List[MethodSummary] = []
        direct_tps: Dict[str, float] = {}

        for cond in conditions:
            for method in methods:
                s = aggregate(self._results, method, cond)
                summaries.append(s)
                if method == "direct":
                    direct_tps[cond] = s.mean_tps

        # Compute speedup vs direct
        for s in summaries:
            base = direct_tps.get(s.network_condition, 0.0)
            if base > 0 and s.mean_tps > 0:
                s.speedup_vs_direct = s.mean_tps / base

        return summaries


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _canonical_method_order(results: List[ExperimentResult]) -> List[str]:
    """Return methods in a sensible display order."""
    k_vals = sorted({r.k_value for r in results if r.k_value is not None})
    order = ["direct"]
    for k in k_vals:
        order.append(f"spec_k{k}")
    for k in k_vals:
        order.append(f"async_k{k}")
    # Add any remaining
    seen = set(order)
    for r in results:
        if r.method not in seen:
            order.append(r.method)
            seen.add(r.method)
    return order
