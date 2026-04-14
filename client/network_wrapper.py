"""Network wrapper for cloud client with simulation support."""
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.network_simulator import NetworkSimulator, get_network_regime


def create_network_wrapped_client(cloud_client, regime: str, enable_simulation: bool = True):
    """
    Create a network-wrapped cloud client.
    
    Args:
        cloud_client: Base cloud client callable
        regime: Network regime name ('good', 'medium', 'bad', 'bursty')
        enable_simulation: Whether to enable network simulation
    
    Returns:
        Wrapped cloud client with network simulation if enabled
    """
    if not enable_simulation:
        return cloud_client
    
    network_regime = get_network_regime(regime)
    simulator = NetworkSimulator(network_regime)
    
    return simulator.wrap_cloud_client(cloud_client)
