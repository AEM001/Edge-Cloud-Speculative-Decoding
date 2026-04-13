"""Network simulation wrapper for testing different network conditions."""
import time
import logging
from typing import Callable
from dataclasses import dataclass

from ..protocol import EdgeRequest, CloudResponse

logger = logging.getLogger(__name__)


@dataclass
class NetworkCondition:
    """Network condition parameters."""
    name: str
    rtt_ms: float  # Round-trip time in milliseconds
    bandwidth_mbps: float  # Bandwidth in Mbps
    
    def __str__(self):
        return f"{self.name} (RTT={self.rtt_ms}ms, BW={self.bandwidth_mbps}Mbps)"


# Define the 4 network regimes from the spec
NETWORK_REGIMES = {
    "good": NetworkCondition(name="Good", rtt_ms=10, bandwidth_mbps=100),
    "medium": NetworkCondition(name="Medium", rtt_ms=40, bandwidth_mbps=20),
    "bad": NetworkCondition(name="Bad", rtt_ms=100, bandwidth_mbps=5),
    "bursty": NetworkCondition(name="Bursty", rtt_ms=20, bandwidth_mbps=50),  # Will alternate
}


class NetworkSimulator:
    """
    Simulates network conditions by adding delays and bandwidth throttling.
    
    This wraps the actual cloud client and adds artificial delays to simulate
    different network conditions.
    """
    
    def __init__(
        self,
        cloud_client: Callable[[EdgeRequest], CloudResponse],
        condition: NetworkCondition,
        enable_simulation: bool = True
    ):
        """
        Initialize network simulator.
        
        Args:
            cloud_client: Actual cloud client function
            condition: Network condition to simulate
            enable_simulation: If False, no simulation is applied (passthrough)
        """
        self.cloud_client = cloud_client
        self.condition = condition
        self.enable_simulation = enable_simulation
        
        # For bursty network
        self.bursty_state = "A"  # A or B
        self.bursty_switch_time = time.time()
        self.bursty_interval = 20.0  # Switch every 20 seconds
        
        if enable_simulation:
            logger.info(f"NetworkSimulator enabled with condition: {condition}")
        else:
            logger.info("NetworkSimulator disabled (passthrough mode)")
    
    def __call__(self, request: EdgeRequest) -> CloudResponse:
        """
        Send request through simulated network.
        
        Args:
            request: EdgeRequest to send
        
        Returns:
            CloudResponse with simulated network effects
        """
        if not self.enable_simulation:
            # Passthrough mode - no simulation
            return self.cloud_client(request)
        
        # Get current network condition (may change for bursty)
        current_condition = self._get_current_condition()
        
        # Calculate payload sizes (rough estimate)
        request_size_bytes = self._estimate_request_size(request)
        
        # Calculate transmission delay based on bandwidth
        # transmission_time = size / bandwidth
        transmission_delay_ms = (request_size_bytes * 8) / (current_condition.bandwidth_mbps * 1000)
        
        # Add half RTT before sending (one-way latency)
        one_way_latency_ms = current_condition.rtt_ms / 2
        
        # Apply pre-request delay (one-way latency + transmission time)
        pre_delay_ms = one_way_latency_ms + transmission_delay_ms
        time.sleep(pre_delay_ms / 1000)
        
        # Send actual request
        response = self.cloud_client(request)
        
        # Estimate response size
        response_size_bytes = self._estimate_response_size(response)
        
        # Calculate response transmission delay
        response_transmission_delay_ms = (response_size_bytes * 8) / (current_condition.bandwidth_mbps * 1000)
        
        # Add post-request delay (one-way latency + response transmission)
        post_delay_ms = one_way_latency_ms + response_transmission_delay_ms
        time.sleep(post_delay_ms / 1000)
        
        # Update RTT in response to reflect simulated network
        total_simulated_delay_ms = pre_delay_ms + post_delay_ms
        if response.rtt_ms:
            response.rtt_ms += total_simulated_delay_ms
        else:
            response.rtt_ms = total_simulated_delay_ms
        
        logger.debug(
            f"Network simulation: {current_condition.name}, "
            f"added_delay={total_simulated_delay_ms:.2f}ms, "
            f"total_rtt={response.rtt_ms:.2f}ms"
        )
        
        return response
    
    def _get_current_condition(self) -> NetworkCondition:
        """Get current network condition (handles bursty switching)."""
        if self.condition.name != "Bursty":
            return self.condition
        
        # Handle bursty network switching
        current_time = time.time()
        elapsed = current_time - self.bursty_switch_time
        
        if elapsed >= self.bursty_interval:
            # Switch state
            self.bursty_state = "B" if self.bursty_state == "A" else "A"
            self.bursty_switch_time = current_time
            logger.info(f"Bursty network switched to state {self.bursty_state}")
        
        # Return condition based on state
        if self.bursty_state == "A":
            return NetworkCondition(name="Bursty-A", rtt_ms=20, bandwidth_mbps=50)
        else:
            return NetworkCondition(name="Bursty-B", rtt_ms=120, bandwidth_mbps=5)
    
    def _estimate_request_size(self, request: EdgeRequest) -> int:
        """
        Estimate request size in bytes.
        
        This is a rough estimate based on the data being sent.
        """
        # Each token ID is ~4 bytes, each float is ~8 bytes
        size = 0
        size += len(request.prefix_ids) * 4
        size += len(request.draft_ids) * 4
        size += len(request.draft_logprobs) * 8
        size += 100  # Overhead for JSON structure, metadata, etc.
        return size
    
    def _estimate_response_size(self, response: CloudResponse) -> int:
        """
        Estimate response size in bytes.
        """
        size = 0
        size += len(response.accepted_token_ids) * 4
        size += 4  # correction_token_id
        size += 50  # Overhead for JSON structure, timing fields, etc.
        return size


def create_network_wrapped_client(
    cloud_client: Callable[[EdgeRequest], CloudResponse],
    regime: str = "good",
    enable_simulation: bool = True
) -> Callable[[EdgeRequest], CloudResponse]:
    """
    Create a network-wrapped cloud client.
    
    Args:
        cloud_client: Actual cloud client function
        regime: Network regime name ("good", "medium", "bad", "bursty")
        enable_simulation: If False, returns unwrapped client
    
    Returns:
        Network-wrapped cloud client function
    """
    if not enable_simulation:
        logger.info("Network simulation disabled")
        return cloud_client
    
    if regime not in NETWORK_REGIMES:
        logger.warning(f"Unknown regime '{regime}', using 'good'")
        regime = "good"
    
    condition = NETWORK_REGIMES[regime]
    simulator = NetworkSimulator(
        cloud_client=cloud_client,
        condition=condition,
        enable_simulation=True
    )
    
    return simulator
