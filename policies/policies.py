"""
Policies for speculative decoding K selection.

Policy 1: Static K - fixed draft lengths (2, 4, 6)
Policy 2: [Placeholder for dynamic policy]
"""
from typing import List, Callable
from protocol import TokenInfo  # noqa: E402

# Policy 1: Static K
# Test exactly three fixed values: K = 2, 4, 6
# Why three: enough to show under-speculation, middle point, over-speculation

def static_k_policy(k: int):
    """Create a static K policy with given K value."""
    def policy(round_id: int, draft_tokens: List[TokenInfo]) -> int:
        return k
    return policy


def static_k_2(round_id: int, draft_tokens: List[TokenInfo]) -> int:
    """Static K = 2 policy."""
    return 2


def static_k_4(round_id: int, draft_tokens: List[TokenInfo]) -> int:
    """Static K = 4 policy."""
    return 4


def static_k_6(round_id: int, draft_tokens: List[TokenInfo]) -> int:
    """Static K = 6 policy."""
    return 6


# Policy registry
POLICIES: dict[str, Callable[[int, List[TokenInfo]], int]] = {
    "static_k_2": static_k_2,
    "static_k_4": static_k_4,
    "static_k_6": static_k_6,
}


def get_policy(policy_name: str) -> Callable[[int, List[TokenInfo]], int]:
    """
    Get a policy function by name.
    
    Args:
        policy_name: Name of the policy
    
    Returns:
        Policy function
    """
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(POLICIES.keys())}")
    return POLICIES[policy_name]


# Policy 2: [Placeholder for dynamic policy]
# This can be implemented later based on research requirements
# Examples could include:
# - Confidence-based K (use entropy/top-margin to determine K)
# - Adaptive K based on recent acceptance history
# - Learning-based policy using reinforcement learning
