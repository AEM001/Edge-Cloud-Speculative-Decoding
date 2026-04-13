"""
Network regime simulator for benchmarking.

Four regimes as specified:
1. Good: RTT = 10ms, bandwidth = 100 Mbps
2. Medium: RTT = 40ms, bandwidth = 20 Mbps
3. Bad: RTT = 100ms, bandwidth = 5 Mbps
4. Bursty: Switches every 20 seconds between state A (20ms, 50Mbps) and state B (120ms, 5Mbps)
"""
import time
import random
import threading
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from protocol import EdgeRequest, CloudResponse


class NetworkRegime(Enum):
    """Network regime types."""
    GOOD = "good"
    MEDIUM = "medium"
    BAD = "bad"
    BURSTY = "bursty"


@dataclass
class NetworkCondition:
    """Network condition parameters."""
    rtt_ms: float
    bandwidth_mbps: float


# Network regime definitions
REGIME_CONDITIONS = {
    NetworkRegime.GOOD: NetworkCondition(rtt_ms=10.0, bandwidth_mbps=100.0),
    NetworkRegime.MEDIUM: NetworkCondition(rtt_ms=40.0, bandwidth_mbps=20.0),
    NetworkRegime.BAD: NetworkCondition(rtt_ms=100.0, bandwidth_mbps=5.0),
    NetworkRegime.BURSTY: NetworkCondition(rtt_ms=0.0, bandwidth_mbps=0.0),  # Special case
}

# Bursty regime states
BURSTY_STATE_A = NetworkCondition(rtt_ms=20.0, bandwidth_mbps=50.0)
BURSTY_STATE_B = NetworkCondition(rtt_ms=120.0, bandwidth_mbps=5.0)
BURSTY_SWITCH_INTERVAL_SEC = 20.0


class NetworkSimulator:
    """
    Simulates network conditions for benchmarking.
    
    For local testing, this adds artificial delay and bandwidth constraints
    to simulate different network regimes.
    """
    
    def __init__(self, regime: NetworkRegime):
        """
        Initialize network simulator.
        
        Args:
            regime: Network regime to simulate
        """
        self.regime = regime
        self.start_time = time.time()
        
        # Bursty regime state tracking
        self.bursty_state = "A"
        self.last_bursty_switch = time.time()
        
        logger_name = f"NetworkSimulator({regime.value})"
        self.logger = logging.getLogger(logger_name)
        
        self.logger.info(f"Initialized with regime: {regime.value}")
    
    def _get_current_condition(self) -> NetworkCondition:
        """Get current network condition based on regime and time."""
        if self.regime != NetworkRegime.BURSTY:
            return REGIME_CONDITIONS[self.regime]
        
        # Bursty regime: switch between states A and B
        now = time.time()
        if now - self.last_bursty_switch >= BURSTY_SWITCH_INTERVAL_SEC:
            self.bursty_state = "B" if self.bursty_state == "A" else "A"
            self.last_bursty_switch = now
            self.logger.debug(f"Bursty regime switched to state {self.bursty_state}")
        
        return BURSTY_STATE_A if self.bursty_state == "A" else BURSTY_STATE_B
    
    def simulate_delay(self, payload_size_bytes: int) -> float:
        """
        Simulate network delay based on current condition.
        
        Args:
            payload_size_bytes: Size of payload in bytes
        
        Returns:
            Simulated delay in milliseconds
        """
        condition = self._get_current_condition()
        
        # RTT component (round-trip time)
        rtt_delay = condition.rtt_ms
        
        # Bandwidth component (transmission time)
        # Convert Mbps to bytes/ms: 1 Mbps = 125,000 bytes/s = 125 bytes/ms
        bytes_per_ms = condition.bandwidth_mbps * 125
        transmission_delay_ms = payload_size_bytes / bytes_per_ms if bytes_per_ms > 0 else 0
        
        # Add small random jitter (±10%)
        jitter = random.uniform(-0.1, 0.1) * (rtt_delay + transmission_delay_ms)
        
        total_delay = rtt_delay + transmission_delay_ms + jitter
        return max(0, total_delay)
    
    def wrap_cloud_client(
        self,
        cloud_client,
        measure_rtt: bool = True
    ):
        """
        Wrap a cloud client to add network simulation.
        
        Args:
            cloud_client: Original cloud client function
            measure_rtt: Whether to measure and include RTT in response
        
        Returns:
            Wrapped cloud client with network simulation
        """
        def wrapped_client(request: EdgeRequest) -> CloudResponse:
            # Simulate uplink delay
            request_size = len(str(request.to_dict()))  # Rough estimate
            uplink_delay = self.simulate_delay(request_size)
            time.sleep(uplink_delay / 1000.0)
            
            # Call actual cloud client
            request_send_time = time.time()
            response = cloud_client(request)
            response_receive_time = time.time()
            
            # Simulate downlink delay
            response_size = len(str(response.to_dict()))  # Rough estimate
            downlink_delay = self.simulate_delay(response_size)
            time.sleep(downlink_delay / 1000.0)
            
            # Add RTT to response if requested
            if measure_rtt:
                total_rtt = uplink_delay + downlink_delay
                response.rtt_ms = total_rtt
            
            return response
        
        return wrapped_client
    
    def get_current_stats(self) -> dict:
        """Get current network statistics."""
        condition = self._get_current_condition()
        return {
            "regime": self.regime.value,
            "current_rtt_ms": condition.rtt_ms,
            "current_bandwidth_mbps": condition.bandwidth_mbps,
            "bursty_state": self.bursty_state if self.regime == NetworkRegime.BURSTY else None,
        }


def get_network_regime(regime_name: str) -> NetworkRegime:
    """
    Get network regime enum from string name.
    
    Args:
        regime_name: Name of regime ('good', 'medium', 'bad', 'bursty')
    
    Returns:
        NetworkRegime enum
    """
    try:
        return NetworkRegime(regime_name.lower())
    except ValueError:
        valid = [r.value for r in NetworkRegime]
        raise ValueError(f"Unknown regime: {regime_name}. Valid: {valid}")


# Import logging after definition to avoid circular import
import logging
