from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class TokenInfo:
    token_id: int
    logprob: float


@dataclass
class DraftRequest:
    verified_prefix: List[int]
    num_draft_tokens: int = 5


@dataclass
class DraftResponse:
    draft_token_ids: List[int]
    logprobs: List[float]
    
    def to_dict(self):
        return {
            "draft_token_ids": self.draft_token_ids,
            "logprobs": self.logprobs
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DraftResponse":
        return cls(
            draft_token_ids=data["draft_token_ids"],
            logprobs=data["logprobs"]
        )


@dataclass
class EdgeRequest:
    request_id: str
    round_id: int
    prefix_ids: List[int]
    draft_ids: List[int]
    draft_logprobs: List[float]
    edge_draft_time_ms: float
    policy_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        import math
        safe_logprobs = [
            -1e10 if math.isinf(lp) and lp < 0 else lp 
            for lp in self.draft_logprobs
        ]
        return {
            "request_id": self.request_id,
            "round_id": self.round_id,
            "prefix_ids": self.prefix_ids,
            "draft_ids": self.draft_ids,
            "draft_logprobs": safe_logprobs,
            "edge_draft_time_ms": self.edge_draft_time_ms,
            "policy_metadata": self.policy_metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EdgeRequest":
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
    request_id: str
    round_id: int
    accepted_len: int
    accepted_token_ids: List[int]
    correction_token_id: Optional[int]
    server_verify_time_ms: float
    server_total_time_ms: float
    rtt_ms: Optional[float] = None
    
    def to_dict(self):
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
