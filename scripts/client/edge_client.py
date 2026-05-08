"""Edge client for speculative decoding on Mac."""
import time
import uuid
import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field

from core.protocol import (
    EdgeRequest, CloudResponse, DraftRequest, DraftResponse,
    TokenInfo,
)
from core.draft_generator import VLLMDraftGenerator
from core.model_manager import VLLMModelManager

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
        previous_net_output = None
        previous_policy_state = None
        
        while len(verified_prefix) - len(prompt_ids) < self.max_new_tokens:
            # Check for EOS in newly generated tokens only (not in prompt)
            if self.eos_token_id and self.eos_token_id in verified_prefix[len(prompt_ids):]:
                logger.info(f"EOS token reached at round {round_id}")
                break
            
            # Step 1: Generate draft tokens
            draft_start = time.time()
            
            # Determine K using policy
            # For now, pass empty list since we don't have draft tokens yet
            requested_K = int(policy(round_id, []))
            remaining_tokens = self.max_new_tokens - (len(verified_prefix) - len(prompt_ids))
            K = max(1, min(requested_K, remaining_tokens))
            policy_state = getattr(policy, "state_name", None)
            
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
            for token_id, logprob in zip(
                draft_response.draft_token_ids,
                draft_response.logprobs
            ):
                token_info = TokenInfo(
                    token_id=token_id,
                    logprob=logprob
                )
                draft_tokens_info.append(token_info)
            
            # Step 2: Send request to cloud
            request_start = time.time()
            
            edge_request = EdgeRequest(
                request_id=request_id,
                prefix_ids=verified_prefix,
                draft_ids=draft_response.draft_token_ids,
            )
            
            uplink_size = (len(verified_prefix) + len(draft_response.draft_token_ids)) * 4 + 64
            metrics.uplink_bytes += uplink_size
            
            # Send to cloud
            cloud_response = self.cloud_client(edge_request)
            
            request_time_ms = (time.time() - request_start) * 1000
            
            downlink_size = 48
            metrics.downlink_bytes += downlink_size
            
            # Step 3: Update prefix based on verification
            verify_prefix_len = len(verified_prefix)
            verify_draft_len = len(draft_response.draft_token_ids)
            accepted_len = cloud_response.accepted_len
            accepted_tokens = draft_response.draft_token_ids[:accepted_len]
            correction_token = cloud_response.correction_token_id
            remaining_before_commit = self.max_new_tokens - (len(verified_prefix) - len(prompt_ids))
            
            # Append accepted draft tokens
            accepted_to_append = accepted_tokens[:remaining_before_commit]
            verified_prefix.extend(accepted_to_append)
            remaining_after_accept = self.max_new_tokens - (len(verified_prefix) - len(prompt_ids))
            
            # Append correction token if provided and not EOS
            correction_emitted = False
            if correction_token is not None:
                if correction_token != self.eos_token_id and remaining_after_accept > 0:
                    verified_prefix.append(correction_token)
                    correction_emitted = True
            generated_this_round = len(accepted_to_append) + (1 if correction_emitted else 0)
            net_output = accepted_len + (1 if correction_token is not None else 0)
            
            # Step 4: Record metrics
            metrics.total_rounds += 1
            metrics.total_drafted_tokens += len(draft_response.draft_token_ids)
            metrics.total_accepted_drafted_tokens += accepted_len
            metrics.total_edge_draft_time_ms += draft_time_ms
            metrics.total_server_verify_time_ms += cloud_response.server_verify_time_ms
            # Network time = RTT (client-measured round-trip) - server verify time
            # rtt_ms is set by http_cloud_client as total round-trip including server
            if cloud_response.rtt_ms is not None:
                network_time_ms = cloud_response.rtt_ms - cloud_response.server_verify_time_ms
            else:
                network_time_ms = request_time_ms - cloud_response.server_verify_time_ms
            metrics.total_network_time_ms += max(0.0, network_time_ms)
            
            if cloud_response.rtt_ms:
                # Update average RTT
                n = metrics.total_rounds
                metrics.average_rtt_ms = (metrics.average_rtt_ms * (n - 1) + cloud_response.rtt_ms) / n
            
            # Record round details
            round_detail = {
                "round_id": round_id,
                "K": K,
                "requested_K": requested_K,
                "policy_state": policy_state,
                "prev_net_output": previous_net_output,
                "prev_policy_state": previous_policy_state,
                "drafted": len(draft_response.draft_token_ids),
                "accepted": accepted_len,
                "generated_this_round": generated_this_round,
                "net_output": net_output,
                "correction_emitted": correction_emitted,
                "draft_time_ms": draft_time_ms,
                "server_time_ms": cloud_response.server_verify_time_ms,
                "rtt_ms": cloud_response.rtt_ms,
                "correction": correction_token,
                "verify_prefix_len": verify_prefix_len,
                "verify_draft_len": verify_draft_len,
                "verify_input_len": verify_prefix_len + verify_draft_len,
            }
            metrics.round_details.append(round_detail)
            
            logger.debug(
                f"Round {round_id}: K={K}, drafted={len(draft_response.draft_token_ids)}, "
                f"accepted={accepted_len}, draft_time={draft_time_ms:.2f}ms, "
                f"server_time={cloud_response.server_verify_time_ms:.2f}ms"
            )

            if hasattr(policy, "on_round_complete"):
                policy.on_round_complete(round_detail)
            previous_net_output = net_output
            previous_policy_state = policy_state
            
            round_id += 1
            
            # Check if we should stop (no tokens generated)
            if generated_this_round == 0:
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
