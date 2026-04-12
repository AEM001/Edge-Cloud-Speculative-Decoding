"""Data structures for speculative decoding protocol."""
from dataclasses import dataclass
from typing import List, Optional
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
