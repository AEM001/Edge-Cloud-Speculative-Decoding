"""
Network condition simulation for speculative decoding experiments.

Wraps any cloud_client callable and injects:
  - RTT-based latency (half applied each direction)
  - Separate uplink / downlink bandwidth throttling
  - Bursty profiles: time-based spike state machine that alternates between
    normal and degraded (spike) periods

Usage
-----
    from experiments.network_conditions import NetworkCondition, ThrottledCloudClient

    cond = NetworkCondition.good()
    client = ThrottledCloudClient(base_client, cond)
    # pass `client` anywhere EdgeClient expects a cloud_client
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from core.protocol import EdgeRequest, CloudResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NetworkCondition descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NetworkCondition:
    """
    Descriptor of a simulated network link.

    For steady-state profiles (good, medium) set the normal_* fields only.
    For bursty profiles also set spike_* and spike/normal durations — the
    ThrottledCloudClient will alternate between normal and spike states.
    """

    name: str

    # Normal (baseline) link parameters
    rtt_ms: float               # full round-trip time (split equally per direction)
    download_mbps: float        # downlink throughput
    upload_mbps: float          # uplink throughput

    # Bursty spike parameters (ignored when spike_duration_sec == 0)
    spike_rtt_ms: float = 0.0
    spike_download_mbps: float = 0.0
    spike_upload_mbps: float = 0.0
    spike_duration_sec: float = 0.0     # how long a burst lasts
    normal_duration_sec: float = 0.0    # how long normal period lasts between bursts

    # ------------------------------------------------------------------
    # Pre-defined network profiles
    # ------------------------------------------------------------------

    @classmethod
    def good(cls) -> "NetworkCondition":
        """Good: RTT 25 ms, 120 Mbps down, 30 Mbps up."""
        return cls(
            name="good",
            rtt_ms=25.0,
            download_mbps=120.0,
            upload_mbps=30.0,
        )

    @classmethod
    def medium(cls) -> "NetworkCondition":
        """Medium: RTT 55 ms, 35 Mbps down, 20 Mbps up."""
        return cls(
            name="medium",
            rtt_ms=55.0,
            download_mbps=35.0,
            upload_mbps=20.0,
        )

    @classmethod
    def bursty(cls) -> "NetworkCondition":
        """
        Bursty: alternates between a normal period (RTT 40 ms, 30/15 Mbps)
        and a spike period (RTT 250 ms, 5/2 Mbps) every 8 s / 2 s.
        """
        return cls(
            name="bursty",
            rtt_ms=40.0,
            download_mbps=30.0,
            upload_mbps=15.0,
            spike_rtt_ms=250.0,
            spike_download_mbps=5.0,
            spike_upload_mbps=2.0,
            spike_duration_sec=2.0,
            normal_duration_sec=8.0,
        )

    @classmethod
    def all_profiles(cls):
        """Canonical ordered list of profiles used in experiments."""
        return [
            cls.good(),
            cls.medium(),
            cls.bursty(),
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_bursty(self) -> bool:
        return self.spike_duration_sec > 0

    def current_link_params(self, wall_time: float) -> Tuple[float, float, float]:
        """
        Return (one_way_latency_ms, download_mbps, upload_mbps) at `wall_time`.

        For bursty profiles, cycles through normal → spike → normal …
        For steady profiles, always returns the normal values.
        """
        if not self.is_bursty:
            return self.rtt_ms / 2.0, self.download_mbps, self.upload_mbps

        cycle = self.normal_duration_sec + self.spike_duration_sec
        phase = wall_time % cycle
        if phase < self.normal_duration_sec:
            return self.rtt_ms / 2.0, self.download_mbps, self.upload_mbps
        else:
            return self.spike_rtt_ms / 2.0, self.spike_download_mbps, self.spike_upload_mbps

    @staticmethod
    def _payload_delay_ms(payload_bytes: int, throughput_mbps: float) -> float:
        if throughput_mbps <= 0:
            return 0.0
        return (payload_bytes * 8 / 1_000_000 / throughput_mbps) * 1000.0

    def __str__(self) -> str:
        if self.is_bursty:
            return (
                f"{self.name}(rtt={self.rtt_ms}ms↔{self.spike_rtt_ms}ms, "
                f"bw={self.download_mbps}/{self.upload_mbps}↔"
                f"{self.spike_download_mbps}/{self.spike_upload_mbps} Mbps, "
                f"cycle={self.normal_duration_sec}s+{self.spike_duration_sec}s)"
            )
        return (
            f"{self.name}(rtt={self.rtt_ms}ms, "
            f"dl={self.download_mbps}Mbps, ul={self.upload_mbps}Mbps)"
        )


# ---------------------------------------------------------------------------
# Throttled cloud client wrapper
# ---------------------------------------------------------------------------

@dataclass
class NetworkCallStats:
    """Per-call network overhead statistics accumulated by ThrottledCloudClient."""
    num_calls: int = 0
    total_simulated_uplink_delay_ms: float = 0.0
    total_simulated_downlink_delay_ms: float = 0.0
    total_simulated_overhead_ms: float = 0.0
    total_uplink_bytes: int = 0
    total_downlink_bytes: int = 0


class ThrottledCloudClient:
    """
    wraps the normal request thing with the simulated time(transmit and payload),
    the basic time is :
    t0 = time.perf_counter()
        response: CloudResponse = self.base_client(request)
        actual_server_ms = (time.perf_counter() - t0) * 1000
    """

    def __init__(
        self,
        base_client: Callable[[EdgeRequest], CloudResponse],
        condition: NetworkCondition,
    ):
        self.base_client = base_client
        self.condition = condition
        self.stats = NetworkCallStats()
        self._start_wall = time.perf_counter()  # reference for bursty cycle

    def __call__(self, request: EdgeRequest) -> CloudResponse:
        cond = self.condition
        now = time.perf_counter() - self._start_wall

        one_way_ms, dl_mbps, ul_mbps = cond.current_link_params(now)
        in_spike = cond.is_bursty and (one_way_ms == cond.spike_rtt_ms / 2.0)

        # --- Uplink payload ------------------------------------------------
        uplink_payload = json.dumps(request.to_dict()).encode("utf-8")
        uplink_bytes = len(uplink_payload)
        uplink_bw_delay = NetworkCondition._payload_delay_ms(uplink_bytes, ul_mbps)
        uplink_delay = one_way_ms + uplink_bw_delay

        _sleep_ms(uplink_delay)

        # --- Actual call ---------------------------------------------------
        t0 = time.perf_counter()
        response: CloudResponse = self.base_client(request)
        actual_server_ms = (time.perf_counter() - t0) * 1000

        # --- Downlink payload ----------------------------------------------
        downlink_payload = json.dumps(response.to_dict()).encode("utf-8")
        downlink_bytes = len(downlink_payload)
        downlink_bw_delay = NetworkCondition._payload_delay_ms(downlink_bytes, dl_mbps)
        downlink_delay = one_way_ms + downlink_bw_delay

        _sleep_ms(downlink_delay)

        # --- Patch RTT on response ----------------------------------------
        simulated_overhead = uplink_delay + downlink_delay
        response.rtt_ms = actual_server_ms + simulated_overhead

        # --- Accumulate stats ---------------------------------------------
        s = self.stats
        s.num_calls += 1
        s.total_simulated_uplink_delay_ms += uplink_delay
        s.total_simulated_downlink_delay_ms += downlink_delay
        s.total_simulated_overhead_ms += simulated_overhead
        s.total_uplink_bytes += uplink_bytes
        s.total_downlink_bytes += downlink_bytes

        logger.debug(
            "[%s%s] ul=%.1f ms (%d B @ %.0f Mbps)  dl=%.1f ms (%d B @ %.0f Mbps)  "
            "overhead=%.1f ms  rtt=%.1f ms",
            cond.name, "↑spike" if in_spike else "",
            uplink_delay, uplink_bytes, ul_mbps,
            downlink_delay, downlink_bytes, dl_mbps,
            simulated_overhead, response.rtt_ms,
        )

        return response

    def verify_batch(self, requests: List[EdgeRequest]) -> List[CloudResponse]:
        if not requests:
            return []

        if not hasattr(self.base_client, "verify_batch"):
            # Fall back to sequential calls (still throttled) if base client
            # lacks native batch support.
            return [self(req) for req in requests]

        cond = self.condition
        now = time.perf_counter() - self._start_wall
        one_way_ms, dl_mbps, ul_mbps = cond.current_link_params(now)

        uplink_payload = json.dumps({"requests": [req.to_dict() for req in requests]}).encode("utf-8")
        uplink_bytes = len(uplink_payload)
        uplink_bw_delay = NetworkCondition._payload_delay_ms(uplink_bytes, ul_mbps)
        uplink_delay = one_way_ms + uplink_bw_delay
        _sleep_ms(uplink_delay)

        t0 = time.perf_counter()
        responses: List[CloudResponse] = self.base_client.verify_batch(requests)
        actual_server_ms = (time.perf_counter() - t0) * 1000

        downlink_payload = json.dumps({"responses": [resp.to_dict() for resp in responses]}).encode("utf-8")
        downlink_bytes = len(downlink_payload)
        downlink_bw_delay = NetworkCondition._payload_delay_ms(downlink_bytes, dl_mbps)
        downlink_delay = one_way_ms + downlink_bw_delay
        _sleep_ms(downlink_delay)

        simulated_overhead = uplink_delay + downlink_delay
        for resp in responses:
            resp.rtt_ms = actual_server_ms + simulated_overhead

        stats = self.stats
        stats.num_calls += 1
        stats.total_simulated_uplink_delay_ms += uplink_delay
        stats.total_simulated_downlink_delay_ms += downlink_delay
        stats.total_simulated_overhead_ms += simulated_overhead
        stats.total_uplink_bytes += uplink_bytes
        stats.total_downlink_bytes += downlink_bytes

        logger.debug(
            "[%s] batch ul=%.1f ms (%d B @ %.0f Mbps)  dl=%.1f ms (%d B @ %.0f Mbps)  overhead=%.1f ms  rtt≈%.1f ms",
            cond.name,
            uplink_delay,
            uplink_bytes,
            ul_mbps,
            downlink_delay,
            downlink_bytes,
            dl_mbps,
            simulated_overhead,
            responses[0].rtt_ms if responses else 0.0,
        )

        return responses

    def reset_stats(self) -> None:
        self.stats = NetworkCallStats()

    def get_stats_dict(self) -> dict:
        s = self.stats
        return {
            "condition": self.condition.name,
            "num_calls": s.num_calls,
            "total_simulated_uplink_delay_ms": round(s.total_simulated_uplink_delay_ms, 2),
            "total_simulated_downlink_delay_ms": round(s.total_simulated_downlink_delay_ms, 2),
            "total_simulated_overhead_ms": round(s.total_simulated_overhead_ms, 2),
            "total_uplink_bytes": s.total_uplink_bytes,
            "total_downlink_bytes": s.total_downlink_bytes,
            "avg_overhead_per_call_ms": round(
                s.total_simulated_overhead_ms / s.num_calls if s.num_calls else 0, 2
            ),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sleep_ms(ms: float) -> None:
    """Sleep for `ms` milliseconds (no-op for ≤ 0)."""
    if ms > 0:
        time.sleep(ms / 1000.0)
