"""HTTP client for connecting to cloud verification server."""
import time
import logging
import requests
from typing import Optional

from ..protocol import EdgeRequest, CloudResponse

logger = logging.getLogger(__name__)


class HTTPCloudClient:
    """
    HTTP client for communicating with cloud verification server.
    
    This client sends EdgeRequest to the cloud server and receives CloudResponse.
    It handles network timing and error handling.
    """
    
    def __init__(
        self,
        server_url: str,
        timeout: float = 60.0,
        retry_attempts: int = 3
    ):
        """
        Initialize HTTP cloud client.
        
        Args:
            server_url: Base URL of cloud server (e.g., "http://49.234.57.210:8000")
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.verify_endpoint = f"{self.server_url}/verify"
        
        logger.info(f"HTTPCloudClient initialized with server: {self.server_url}")
        
        # Test connection
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5.0)
            if response.status_code == 200:
                logger.info("Successfully connected to cloud server")
                logger.info(f"Server info: {response.json()}")
            else:
                logger.warning(f"Cloud server returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to cloud server: {e}")
            logger.warning("Will retry on first verification request")
    
    def __call__(self, request: EdgeRequest) -> CloudResponse:
        """
        Send verification request to cloud server.
        
        Args:
            request: EdgeRequest with draft tokens to verify
        
        Returns:
            CloudResponse with verification results
        """
        return self.verify(request)
    
    def verify(self, request: EdgeRequest) -> CloudResponse:
        """
        Send verification request to cloud server.
        
        Args:
            request: EdgeRequest with draft tokens to verify
        
        Returns:
            CloudResponse with verification results
        """
        request_start = time.time()
        
        # Convert request to dict
        request_data = request.to_dict()
        
        # Try with retries
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(
                    f"Sending verification request (attempt {attempt + 1}/{self.retry_attempts}): "
                    f"request_id={request.request_id}, round_id={request.round_id}"
                )
                
                response = requests.post(
                    self.verify_endpoint,
                    json=request_data,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                
                # Parse response
                response_data = response.json()
                cloud_response = CloudResponse.from_dict(response_data)
                
                # Calculate RTT
                rtt_ms = (time.time() - request_start) * 1000
                cloud_response.rtt_ms = rtt_ms
                
                logger.debug(
                    f"Received verification response: accepted={cloud_response.accepted_len}, "
                    f"rtt={rtt_ms:.2f}ms"
                )
                
                return cloud_response
                
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.retry_attempts})")
                if attempt < self.retry_attempts - 1:
                    time.sleep(1.0)  # Wait before retry
                    
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(1.0)  # Wait before retry
        
        # All retries failed
        logger.error(f"All retry attempts failed. Last error: {last_error}")
        raise RuntimeError(f"Failed to verify draft after {self.retry_attempts} attempts: {last_error}")
    
    def check_health(self) -> bool:
        """
        Check if cloud server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


def create_http_cloud_client(
    server_url: str,
    timeout: float = 60.0,
    retry_attempts: int = 3
):
    """
    Create an HTTP cloud client function for use with EdgeClient.
    
    Args:
        server_url: Base URL of cloud server
        timeout: Request timeout in seconds
        retry_attempts: Number of retry attempts
    
    Returns:
        Function that takes EdgeRequest and returns CloudResponse
    """
    client = HTTPCloudClient(
        server_url=server_url,
        timeout=timeout,
        retry_attempts=retry_attempts
    )
    
    return client
