from dataclasses import dataclass
from typing import List, Optional


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
    prefix_ids: List[int]
    draft_ids: List[int]
    
    def to_dict(self):
        return {
            "request_id": self.request_id,
            "prefix_ids": self.prefix_ids,
            "draft_ids": self.draft_ids,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EdgeRequest":
        return cls(
            request_id=data["request_id"],
            prefix_ids=data["prefix_ids"],
            draft_ids=data["draft_ids"],
        )


@dataclass
class CloudResponse:
    request_id: str
    accepted_len: int
    correction_token_id: Optional[int]
    server_verify_time_ms: float
    rtt_ms: Optional[float] = None
    
    def to_dict(self):
        return {
            "request_id": self.request_id,
            "accepted_len": self.accepted_len,
            "correction_token_id": self.correction_token_id,
            "server_verify_time_ms": self.server_verify_time_ms,
            "rtt_ms": self.rtt_ms,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CloudResponse":
        return cls(
            request_id=data["request_id"],
            accepted_len=data["accepted_len"],
            correction_token_id=data.get("correction_token_id"),
            server_verify_time_ms=data["server_verify_time_ms"],
            rtt_ms=data.get("rtt_ms"),
        )
