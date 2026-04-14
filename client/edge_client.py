"""Edge client for speculative decoding on Mac."""
import time
import uuid
import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field

import sys
from pathlib import Path
# Import from root directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocol import (
    EdgeRequest, CloudResponse, DraftRequest, DraftResponse,
    TokenInfo
)
from draft_generator import VLLMDraftGenerator
from model_manager import VLLMModelManager

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Per-request metrics as specified in the architecture."""
    request_id: str
    prompt: str
    
    # Timing metrics
    total_latency_ms: float = 0.0
    total_edge_draft_time_ms: float = 0.0
    total_server_verify_time_ms: float = 0.0
    total_network_time_ms: float = 0.0
    average_rtt_ms: float = 0.0
    
    # Token metrics
    generated_tokens: int = 0
    total_rounds: int = 0
    mean_K_chosen: float = 0.0
    
    # Acceptance metrics
    acceptance_ratio: float = 0.0
    total_drafted_tokens: int = 0
    total_accepted_drafted_tokens: int = 0
    wasted_drafted_tokens: int = 0
    
    # Network metrics
    uplink_bytes: int = 0
    downlink_bytes: int = 0
    
    # Per-round details for analysis
    round_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def compute_derived_metrics(self):
        """Compute derived metrics from collected data."""
        if self.total_drafted_tokens > 0:
            self.acceptance_ratio = self.total_accepted_drafted_tokens / self.total_drafted_tokens
        self.wasted_drafted_tokens = self.total_drafted_tokens - self.total_accepted_drafted_tokens
        
        if self.total_rounds > 0:
            self.mean_K_chosen = self.total_drafted_tokens / self.total_rounds


class EdgeClient:
    """
    Edge client for speculative decoding.
    
    Responsibilities:
    - Load draft model and tokenizer
    - Maintain current verified prefix
    - Generate draft chunk of length K
    - Record token-level stats
    - Send draft request to cloud
    - Receive verification result
    - Update prefix
    - Loop until max tokens or EOS
    """
    
    def __init__(
        self,
        model_manager: VLLMModelManager,
        draft_generator: VLLMDraftGenerator,
        cloud_client: Callable[[EdgeRequest], CloudResponse],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None
    ):
        self.model_manager = model_manager
        self.draft_generator = draft_generator
        self.cloud_client = cloud_client
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.eos_token_id = eos_token_id
        
        # Get tokenizer from draft generator
        self.tokenizer = draft_generator.tokenizer
        
        # If EOS not provided, try to get from tokenizer
        if self.eos_token_id is None and self.tokenizer:
            self.eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
        
        logger.info(f"EdgeClient initialized with max_new_tokens={max_new_tokens}, temperature={temperature}")
    
    def generate(
        self,
        prompt: str,
        policy: Callable[[int, List[TokenInfo]], int],
        policy_name: str = "unknown"
    ) -> RequestMetrics:
        """
        Generate text using speculative decoding with a given policy.
        
        Args:
            prompt: Input prompt text
            policy: Function that takes (round_id, draft_tokens) and returns K
            policy_name: Name of the policy for logging
        
        Returns:
            RequestMetrics with all collected metrics
        """
        request_id = str(uuid.uuid4())
        metrics = RequestMetrics(request_id=request_id, prompt=prompt)
        
        start_time = time.time()
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt)
        verified_prefix = list(prompt_ids)
        
        logger.info(f"Starting generation for request {request_id} with policy {policy_name}")
        logger.info(f"Prompt: {prompt!r}")
        logger.info(f"Prompt tokens: {len(prompt_ids)}")
        
        round_id = 0
        
        while len(verified_prefix) - len(prompt_ids) < self.max_new_tokens:
            # Check for EOS in newly generated tokens only (not in prompt)
            if self.eos_token_id and self.eos_token_id in verified_prefix[len(prompt_ids):]:
                logger.info(f"EOS token reached at round {round_id}")
                break
            
            # Step 1: Generate draft tokens
            draft_start = time.time()
            
            # Determine K using policy
            # For now, pass empty list since we don't have draft tokens yet
            K = policy(round_id, [])
            
            # Generate draft
            draft_request = DraftRequest(
                verified_prefix=verified_prefix,
                num_draft_tokens=K
            )
            
            draft_response = self.draft_generator.generate_draft_tokens(
                draft_request,
                temperature=self.temperature
            )
            
            draft_time_ms = (time.time() - draft_start) * 1000
            
            # Build token info list
            draft_tokens_info = []
            for i, (token_id, logprob, prob) in enumerate(zip(
                draft_response.draft_token_ids,
                draft_response.logprobs,
                draft_response.probabilities
            )):
                token_info = TokenInfo(
                    token_id=token_id,
                    logprob=logprob,
                    probability=prob,
                    max_prob=draft_response.confidence_stats["max_probs"][i],
                    entropy=draft_response.confidence_stats["entropies"][i],
                    top_margin=draft_response.confidence_stats["top_margins"][i]
                )
                draft_tokens_info.append(token_info)
            
            # Step 2: Send request to cloud
            request_start = time.time()
            
            edge_request = EdgeRequest(
                request_id=request_id,
                round_id=round_id,
                prefix_ids=verified_prefix,
                draft_ids=draft_response.draft_token_ids,
                draft_logprobs=draft_response.logprobs,
                edge_draft_time_ms=draft_time_ms,
                policy_metadata={"policy_name": policy_name, "K": K}
            )
            
            # Estimate payload size
            uplink_size = len(str(edge_request.to_dict()))  # Rough estimate
            metrics.uplink_bytes += uplink_size
            
            # Send to cloud
            cloud_response = self.cloud_client(edge_request)
            
            request_time_ms = (time.time() - request_start) * 1000
            
            # Estimate response size
            downlink_size = len(str(cloud_response.to_dict()))  # Rough estimate
            metrics.downlink_bytes += downlink_size
            
            # Step 3: Update prefix based on verification
            accepted_len = cloud_response.accepted_len
            accepted_tokens = cloud_response.accepted_token_ids
            correction_token = cloud_response.correction_token_id
            
            # Append accepted draft tokens
            verified_prefix.extend(accepted_tokens)
            
            # Append correction token if provided and not EOS
            if correction_token is not None:
                if correction_token != self.eos_token_id:
                    verified_prefix.append(correction_token)
            
            # Step 4: Record metrics
            metrics.total_rounds += 1
            metrics.total_drafted_tokens += len(draft_response.draft_token_ids)
            metrics.total_accepted_drafted_tokens += accepted_len
            metrics.total_edge_draft_time_ms += draft_time_ms
            metrics.total_server_verify_time_ms += cloud_response.server_verify_time_ms
            metrics.total_network_time_ms += (request_time_ms - cloud_response.server_verify_time_ms)
            
            if cloud_response.rtt_ms:
                # Update average RTT
                n = metrics.total_rounds
                metrics.average_rtt_ms = (metrics.average_rtt_ms * (n - 1) + cloud_response.rtt_ms) / n
            
            # Record round details
            round_detail = {
                "round_id": round_id,
                "K": K,
                "drafted": len(draft_response.draft_token_ids),
                "accepted": accepted_len,
                "draft_time_ms": draft_time_ms,
                "server_time_ms": cloud_response.server_verify_time_ms,
                "rtt_ms": cloud_response.rtt_ms,
                "correction": correction_token
            }
            metrics.round_details.append(round_detail)
            
            logger.debug(
                f"Round {round_id}: K={K}, drafted={len(draft_response.draft_token_ids)}, "
                f"accepted={accepted_len}, draft_time={draft_time_ms:.2f}ms, "
                f"server_time={cloud_response.server_verify_time_ms:.2f}ms"
            )
            
            round_id += 1
            
            # Check if we should stop (no tokens generated)
            if accepted_len == 0 and correction_token is None:
                logger.warning(f"No progress in round {round_id}, stopping")
                break
        
        # Finalize metrics
        metrics.generated_tokens = len(verified_prefix) - len(prompt_ids)
        metrics.total_latency_ms = (time.time() - start_time) * 1000
        metrics.compute_derived_metrics()
        
        logger.info(f"Generation complete: {metrics.generated_tokens} tokens in {metrics.total_rounds} rounds")
        logger.info(f"Acceptance ratio: {metrics.acceptance_ratio:.2%}")
        logger.info(f"Total latency: {metrics.total_latency_ms:.2f}ms")
        
        return metrics
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)
