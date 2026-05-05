



The [protocol.py](cci:7://file:///Users/Mac/code/research/Infra/draft/protocol.py:0:0-0:0) file defines the data structures for a **distributed speculative decoding system** where:
- **Edge (Mac)**: Runs a small draft model
- **Cloud (Ubuntu)**: Runs a large verification model
- They communicate over network

## Class meanings and flow:

### 1. **TokenInfo**
- **Purpose**: Metadata for a single drafted token
- **Fields**: `token_id`, `logprob`
- **Used in**: [edge_client.py](cci:7://file:///Users/Mac/code/research/Infra/draft/client/edge_client.py:0:0-0:0) line 153 (when building token info list for policy function)
- **Note**: Currently only used for policy decisions, but policy is called with empty list (dead code)

### 2. **DraftRequest**
- **Purpose**: Request to generate draft tokens locally
- **Fields**: `verified_prefix` (context tokens), `num_draft_tokens` (K)
- **Used in**: 
  - [draft_generator.py](cci:7://file:///Users/Mac/code/research/Infra/draft/draft_generator.py:0:0-0:0) line 17 (input to [generate_draft_tokens](cci:1://file:///Users/Mac/code/research/Infra/draft/draft_generator.py:14:4-75:9))
  - [edge_client.py](cci:7://file:///Users/Mac/code/research/Infra/draft/client/edge_client.py:0:0-0:0) line 135 (creates request for draft generation)
  - `quick_test.py` line 172 (testing)

### 3. **DraftResponse**
- **Purpose**: Response from draft generator with predicted tokens
- **Fields**: `draft_token_ids`, `logprobs`
- **Used in**:
  - [draft_generator.py](cci:7://file:///Users/Mac/code/research/Infra/draft/draft_generator.py:0:0-0:0) line 73 (return value)
  - [edge_client.py](cci:7://file:///Users/Mac/code/research/Infra/draft/client/edge_client.py:0:0-0:0) line 140 (receives response)
- **Serialization**: [to_dict()](cci:1://file:///Users/Mac/code/research/Infra/draft/protocol.py:46:4-60:9)/[from_dict()](cci:1://file:///Users/Mac/code/research/Infra/draft/protocol.py:62:4-72:9) for network transmission

### 4. **EdgeRequest** (Edge → Cloud)
- **Purpose**: Request sent from edge to cloud for verification
- **Fields**: 
  - `request_id`, `round_id`: Tracking
  - `prefix_ids`: Context tokens
  - `draft_ids`: Draft tokens to verify
  - `draft_logprobs`: Log-probabilities of draft tokens
  - `edge_draft_time_ms`: Timing metrics
  - `policy_metadata`: Policy info
- **Used in**:
  - [edge_client.py](cci:7://file:///Users/Mac/code/research/Infra/draft/client/edge_client.py:0:0-0:0) line 162 (sends to cloud)
  - `http_cloud_client.py` line 50 (receives for verification)
  - `quick_test.py` line 237 (testing)
- **Serialization**: Handles `-inf` logprobs for JSON compatibility

### 5. **CloudResponse** (Cloud → Edge)
- **Purpose**: Response from cloud with verification results
- **Fields**:
  - `accepted_len`: How many draft tokens were accepted
  - `accepted_token_ids`: Accepted tokens
  - `correction_token_id`: First token that differed (if any)
  - `server_verify_time_ms`, `server_total_time_ms`: Timing
  - `rtt_ms`: Round-trip time
- **Used in**:
  - `http_cloud_client.py` line 74 (return value)
  - [edge_client.py](cci:7://file:///Users/Mac/code/research/Infra/draft/client/edge_client.py:0:0-0:0) line 79 (receives response)
  - `async_edge_client.py` line 325 (async version)

## Flow diagram:
```
Edge (Mac)                          Cloud (Ubuntu)
─────────────────────────────────────────────────────
1. DraftRequest → draft_generator → DraftResponse
2. DraftResponse → EdgeRequest → [network] → Cloud
3. Cloud verifies → CloudResponse → [network] → Edge
4. Edge processes CloudResponse, updates prefix
5. Repeat until generation complete
```

The protocol classes are essentially the **API contract** between edge and cloud components, with serialization methods for network transmission.