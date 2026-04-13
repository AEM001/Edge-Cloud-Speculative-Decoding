"""
Cloud server interface for speculative decoding verification.

This module defines the interface that the cloud server (running on Ubuntu with 3060 GPU)
must implement. For local testing on Mac, a mock implementation is provided.

UBUNTU IMPLEMENTATION GUIDE:
=============================
To implement the cloud server on Ubuntu with a 3060 GPU:

1. Load the target model (larger than draft model, e.g., Qwen2.5-7B or similar)
2. Implement the verify_draft method to:
   - Receive EdgeRequest with prefix_ids and draft_ids
   - Run target model forward pass on prefix + draft tokens
   - Compute acceptance for each drafted token using the practical benchmark rule:
     accept drafted token if argmax(target_logits_i) == draft_token_id
   - Determine accepted_len (number of consecutive accepted tokens from start)
   - Get correction token (argmax of target logits at first rejection position)
   - Return CloudResponse with timing info

3. Expose this via HTTP API or gRPC for edge client to call

4. Log server-side metrics as specified in the architecture
"""
import time
import logging
from typing import Optional, List
from abc import ABC, abstractmethod
import random

from protocol import EdgeRequest, CloudResponse

logger = logging.getLogger(__name__)


class CloudServerInterface(ABC):
    """
    Abstract interface for cloud server verification.
    
    This is the interface that must be implemented on Ubuntu with 3060 GPU.
    """
    
    @abstractmethod
    def verify_draft(self, request: EdgeRequest) -> CloudResponse:
        """
        Verify a draft chunk against the target model.
        
        Args:
            request: EdgeRequest with prefix_ids, draft_ids, and metadata
        
        Returns:
            CloudResponse with accepted_len, correction_token, and timing info
        """
        pass


class MockCloudServer(CloudServerInterface):
    """
    Mock cloud server for local testing on Mac.
    
    This simulates cloud verification without needing a GPU or target model.
    It uses simple heuristics to simulate acceptance behavior.
    
    DO NOT USE THIS FOR PRODUCTION BENCHMARKS.
    This is only for testing the edge client locally.
    """
    
    def __init__(
        self,
        base_acceptance_rate: float = 0.7,
        acceptance_variance: float = 0.2,
        base_verify_time_ms: float = 50.0
    ):
        """
        Initialize mock server.
        
        Args:
            base_acceptance_rate: Base probability of accepting a draft token
            acceptance_variance: Variance in acceptance rate
            base_verify_time_ms: Base time to simulate verification
        """
        self.base_acceptance_rate = base_acceptance_rate
        self.acceptance_variance = acceptance_variance
        self.base_verify_time_ms = base_verify_time_ms
        
        logger.warning(
            "Using MockCloudServer - this is for local testing only. "
            "Implement actual CloudServerInterface on Ubuntu with 3060 GPU for real benchmarks."
        )
    
    def verify_draft(self, request: EdgeRequest) -> CloudResponse:
        """
        Simulate draft verification with mock behavior.
        
        Args:
            request: EdgeRequest with prefix_ids, draft_ids, and metadata
        
        Returns:
            CloudResponse with simulated verification results
        """
        verify_start = time.time()
        
        draft_ids = request.draft_ids
        num_drafts = len(draft_ids)
        
        if num_drafts == 0:
            return CloudResponse(
                request_id=request.request_id,
                round_id=request.round_id,
                accepted_len=0,
                accepted_token_ids=[],
                correction_token_id=None,
                server_verify_time_ms=0.0,
                server_total_time_ms=0.0
            )
        
        # Simulate acceptance: tokens are accepted consecutively from start
        # Use random acceptance with some correlation (earlier tokens more likely to be accepted)
        accepted_len = 0
        acceptance_threshold = self.base_acceptance_rate + random.uniform(
            -self.acceptance_variance,
            self.acceptance_variance
        )
        
        for i, token_id in enumerate(draft_ids):
            # Earlier tokens have higher acceptance probability
            position_factor = 1.0 - (i / num_drafts) * 0.3  # Decreases with position
            acceptance_prob = acceptance_threshold * position_factor
            
            if random.random() < acceptance_prob:
                accepted_len += 1
            else:
                break
        
        # Get accepted tokens
        accepted_token_ids = draft_ids[:accepted_len]
        
        # Generate correction token if rejection happened
        correction_token_id = None
        if accepted_len < num_drafts:
            # Simulate correction token (random token ID in reasonable range)
            correction_token_id = random.randint(0, 32000)  # Typical vocab size range
        
        # Simulate verification time based on number of drafts
        verify_time_ms = self.base_verify_time_ms * (1 + num_drafts * 0.1)
        verify_time_ms += random.uniform(-5, 5)  # Add some jitter
        
        verify_end = time.time()
        total_time_ms = (verify_end - verify_start) * 1000
        
        logger.debug(
            f"Mock verification: request={request.request_id}, round={request.round_id}, "
            f"drafted={num_drafts}, accepted={accepted_len}, time={verify_time_ms:.2f}ms"
        )
        
        return CloudResponse(
            request_id=request.request_id,
            round_id=request.round_id,
            accepted_len=accepted_len,
            accepted_token_ids=accepted_token_ids,
            correction_token_id=correction_token_id,
            server_verify_time_ms=verify_time_ms,
            server_total_time_ms=total_time_ms
        )


class CloudServerStub:
    """
    Stub class that demonstrates the expected cloud server implementation.
    
    This is a template/placeholder showing what the Ubuntu implementation should look like.
    Copy this class and implement the actual verification logic with a real target model.
    
    EXPECTED UBUNTU IMPLEMENTATION:
    ================================
    
    ```python
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    class UbuntuCloudServer(CloudServerInterface):
        def __init__(self, model_name: str):
            self.device = "cuda"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()
        
        def verify_draft(self, request: EdgeRequest) -> CloudResponse:
            verify_start = time.time()
            
            # Combine prefix and drafts
            input_ids = request.prefix_ids + request.draft_ids
            input_tensor = torch.tensor([input_ids]).to(self.device)
            
            # Run forward pass
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Get logits for draft positions (skip prefix)
            prefix_len = len(request.prefix_ids)
            draft_logits = logits[prefix_len:-1]  # [num_drafts, vocab_size]
            next_token_logits = logits[-1]  # [vocab_size] for correction
            
            # Verify each draft token using practical benchmark rule
            accepted_len = 0
            for i, draft_id in enumerate(request.draft_ids):
                predicted_id = torch.argmax(draft_logits[i]).item()
                if predicted_id == draft_id:
                    accepted_len += 1
                else:
                    break
            
            # Get correction token if rejection happened
            correction_token_id = None
            if accepted_len < len(request.draft_ids):
                correction_token_id = torch.argmax(next_token_logits).item()
            
            verify_time_ms = (time.time() - verify_start) * 1000
            
            return CloudResponse(
                request_id=request.request_id,
                round_id=request.round_id,
                accepted_len=accepted_len,
                accepted_token_ids=request.draft_ids[:accepted_len],
                correction_token_id=correction_token_id,
                server_verify_time_ms=verify_time_ms,
                server_total_time_ms=verify_time_ms
            )
    ```
    """
    
    def __init__(self):
        logger.warning(
            "CloudServerStub is a template. "
            "Implement UbuntuCloudServer on Ubuntu with 3060 GPU for actual benchmarks."
        )
    
    def verify_draft(self, request: EdgeRequest) -> CloudResponse:
        raise NotImplementedError(
            "This is a stub. Implement CloudServerInterface.verify_draft on Ubuntu."
        )


def create_mock_cloud_client(base_acceptance_rate: float = 0.7):
    """
    Create a mock cloud client function for local testing.
    
    Args:
        base_acceptance_rate: Base acceptance rate for simulation
    
    Returns:
        Function that takes EdgeRequest and returns CloudResponse
    """
    mock_server = MockCloudServer(base_acceptance_rate=base_acceptance_rate)
    
    def mock_client(request: EdgeRequest) -> CloudResponse:
        return mock_server.verify_draft(request)
    
    return mock_client
