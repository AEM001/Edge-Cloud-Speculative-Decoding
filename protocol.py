"""Data structures for speculative decoding protocol."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class TokenInfo:
    """Information about a single drafted token."""
    token_id: int
    logprob: float
    probability: float
    max_prob: float          # Confidence stat: max probability in distribution
    entropy: float           # Confidence stat: entropy of distribution
    top_margin: float        # Confidence stat: top-1 minus top-2 probability margin


@dataclass
class DraftRequest:
    """Request from server to generate draft tokens."""
    verified_prefix: List[int]  # Token IDs of verified context
    num_draft_tokens: int = 5


@dataclass
class DraftResponse:
    """Response containing drafted tokens and lightweight stats."""
    draft_token_ids: List[int]
    logprobs: List[float]
    probabilities: List[float]
    confidence_stats: dict  # Contains max_probs, entropies, top_margins as lists
    
    def to_dict(self):
        """Serialize to dictionary for network transmission."""
        return {
            "draft_token_ids": self.draft_token_ids,
            "logprobs": self.logprobs,
            "probabilities": self.probabilities,
            "confidence_stats": self.confidence_stats
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DraftResponse":
        """Deserialize from dictionary."""
        return cls(
            draft_token_ids=data["draft_token_ids"],
            logprobs=data["logprobs"],
            probabilities=data["probabilities"],
            confidence_stats=data["confidence_stats"]
        )


@dataclass
class EdgeRequest:
    """Request from edge (Mac) to cloud (Ubuntu) for verification."""
    request_id: str
    round_id: int
    prefix_ids: List[int]
    draft_ids: List[int]
    draft_logprobs: List[float]
    edge_draft_time_ms: float
    policy_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        """Serialize to dictionary for network transmission."""
        return {
            "request_id": self.request_id,
            "round_id": self.round_id,
            "prefix_ids": self.prefix_ids,
            "draft_ids": self.draft_ids,
            "draft_logprobs": self.draft_logprobs,
            "edge_draft_time_ms": self.edge_draft_time_ms,
            "policy_metadata": self.policy_metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EdgeRequest":
        """Deserialize from dictionary."""
        return cls(
            request_id=data["request_id"],
            round_id=data["round_id"],
            prefix_ids=data["prefix_ids"],
            draft_ids=data["draft_ids"],
            draft_logprobs=data["draft_logprobs"],
            edge_draft_time_ms=data["edge_draft_time_ms"],
            policy_metadata=data.get("policy_metadata", {})
        )


@dataclass
class CloudResponse:
    """Response from cloud (Ubuntu) to edge (Mac) with verification results."""
    request_id: str
    round_id: int
    accepted_len: int
    accepted_token_ids: List[int]
    correction_token_id: Optional[int]
    server_verify_time_ms: float
    server_total_time_ms: float
    rtt_ms: Optional[float] = None
    
    def to_dict(self):
        """Serialize to dictionary for network transmission."""
        return {
            "request_id": self.request_id,
            "round_id": self.round_id,
            "accepted_len": self.accepted_len,
            "accepted_token_ids": self.accepted_token_ids,
            "correction_token_id": self.correction_token_id,
            "server_verify_time_ms": self.server_verify_time_ms,
            "server_total_time_ms": self.server_total_time_ms,
            "rtt_ms": self.rtt_ms
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CloudResponse":
        """Deserialize from dictionary."""
        return cls(
            request_id=data["request_id"],
            round_id=data["round_id"],
            accepted_len=data["accepted_len"],
            accepted_token_ids=data["accepted_token_ids"],
            correction_token_id=data.get("correction_token_id"),
            server_verify_time_ms=data["server_verify_time_ms"],
            server_total_time_ms=data["server_total_time_ms"],
            rtt_ms=data.get("rtt_ms")
        )
