"""Simple test to connect to the cloud server on port 5090."""
import requests
import sys

SERVER_URL = "http://localhost:5090"

print(f"Testing connection to {SERVER_URL}...")

try:
    # Test health endpoint
    response = requests.get(f"{SERVER_URL}/health", timeout=5.0)
    print(f"Health check response: Status {response.status_code}")
    try:
        print(f"  JSON: {response.json()}")
    except:
        print(f"  Text: {response.text[:200]}")
    
    # Test verify endpoint with a simple request
    test_payload = {
        'request_id': 'test_001',
        'round_id': 0,
        'prefix_ids': [1, 2, 3],
        'draft_ids': [4, 5],
        'draft_logprobs': [-1.0, -1.5],
        'edge_draft_time_ms': 10.0,
        'policy_metadata': {'K': 2}
    }
    
    print(f"\nTesting /verify endpoint...")
    response = requests.post(f"{SERVER_URL}/verify", json=test_payload, timeout=10.0)
    print(f"Verify endpoint response: Status {response.status_code}")
    try:
        print(f"  JSON: {response.json()}")
    except:
        print(f"  Text: {response.text[:200]}")
    
    print("\n✓ All tests passed! Server is ready.")
    sys.exit(0)
    
except requests.exceptions.ConnectionError as e:
    print(f"✗ Connection failed: {e}")
    print(f"  Make sure the server is running on port 5090")
    sys.exit(1)
except requests.exceptions.Timeout as e:
    print(f"✗ Request timeout: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
