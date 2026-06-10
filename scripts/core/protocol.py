from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
    rtt_ms: Optional[float] = None  # Pure network delay
    end_to_end_ms: Optional[float] = None  # End-to-end latency (server + network)
    
    # Detailed timing breakdown
    model_time_ms: Optional[float] = None  # Actual model processing time
    http_overhead_ms: Optional[float] = None  # FastAPI + serialization overhead
    network_tx_ms: Optional[float] = None  # Time to send request (uplink transmission)
    network_rx_ms: Optional[float] = None  # Time to receive response (downlink transmission)
    
    def to_dict(self):
        return {
            "request_id": self.request_id,
            "accepted_len": self.accepted_len,
            "correction_token_id": self.correction_token_id,
            "server_verify_time_ms": self.server_verify_time_ms,
            "rtt_ms": self.rtt_ms,
            "end_to_end_ms": self.end_to_end_ms,
            "model_time_ms": self.model_time_ms,
            "http_overhead_ms": self.http_overhead_ms,
            "network_tx_ms": self.network_tx_ms,
            "network_rx_ms": self.network_rx_ms,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CloudResponse":
        return cls(
            request_id=data["request_id"],
            accepted_len=data["accepted_len"],
            correction_token_id=data.get("correction_token_id"),
            server_verify_time_ms=data["server_verify_time_ms"],
            rtt_ms=data.get("rtt_ms"),
            end_to_end_ms=data.get("end_to_end_ms"),
            model_time_ms=data.get("model_time_ms"),
            http_overhead_ms=data.get("http_overhead_ms"),
            network_tx_ms=data.get("network_tx_ms"),
            network_rx_ms=data.get("network_rx_ms"),
        )


@dataclass
class SpecExtendRequest:
    """Wire request for SpecExtend draft verification.

    Linear drafts only need ``draft_ids``. Tree-shaped drafts additionally set
    position, parent, and attention-mask fields.
    """

    request_id: str
    prefix_ids: List[int]
    draft_ids: List[int]
    draft_position_ids: Optional[List[int]] = None
    parent_indices: Optional[List[int]] = None
    draft_attention_mask: Optional[List[List[int]]] = None
    retrieve_attn_scores: bool = False
    retrieval_chunk_size: int = 64
    retrieve_top_k: int = 16
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "prefix_ids": self.prefix_ids,
            "draft_ids": self.draft_ids,
            "draft_position_ids": self.draft_position_ids,
            "parent_indices": self.parent_indices,
            "draft_attention_mask": self.draft_attention_mask,
            "retrieve_attn_scores": self.retrieve_attn_scores,
            "retrieval_chunk_size": self.retrieval_chunk_size,
            "retrieve_top_k": self.retrieve_top_k,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpecExtendRequest":
        return cls(
            request_id=data["request_id"],
            prefix_ids=data["prefix_ids"],
            draft_ids=data["draft_ids"],
            draft_position_ids=data.get("draft_position_ids"),
            parent_indices=data.get("parent_indices"),
            draft_attention_mask=data.get("draft_attention_mask"),
            retrieve_attn_scores=bool(data.get("retrieve_attn_scores", False)),
            retrieval_chunk_size=int(data.get("retrieval_chunk_size", 64)),
            retrieve_top_k=int(data.get("retrieve_top_k", 16)),
            metadata=data.get("metadata") or {},
        )


@dataclass
class SpecExtendResponse:
    """Cloud response for SpecExtend linear verification and retrieval feedback."""

    request_id: str
    accepted_len: int
    correction_token_id: Optional[int]
    accepted_indices: List[int]
    server_verify_time_ms: float
    target_attn_scores: Optional[List[float]] = None
    selected_chunk_ids: Optional[List[int]] = None
    model_time_ms: Optional[float] = None
    http_overhead_ms: Optional[float] = None
    rtt_ms: Optional[float] = None
    cloud_observability: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "accepted_len": self.accepted_len,
            "correction_token_id": self.correction_token_id,
            "accepted_indices": self.accepted_indices,
            "server_verify_time_ms": self.server_verify_time_ms,
            "target_attn_scores": self.target_attn_scores,
            "selected_chunk_ids": self.selected_chunk_ids,
            "model_time_ms": self.model_time_ms,
            "http_overhead_ms": self.http_overhead_ms,
            "rtt_ms": self.rtt_ms,
            "cloud_observability": self.cloud_observability,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpecExtendResponse":
        return cls(
            request_id=data["request_id"],
            accepted_len=data["accepted_len"],
            correction_token_id=data.get("correction_token_id"),
            accepted_indices=data.get("accepted_indices") or [],
            server_verify_time_ms=data["server_verify_time_ms"],
            target_attn_scores=data.get("target_attn_scores"),
            selected_chunk_ids=data.get("selected_chunk_ids"),
            model_time_ms=data.get("model_time_ms"),
            http_overhead_ms=data.get("http_overhead_ms"),
            rtt_ms=data.get("rtt_ms"),
            cloud_observability=data.get("cloud_observability"),
        )
