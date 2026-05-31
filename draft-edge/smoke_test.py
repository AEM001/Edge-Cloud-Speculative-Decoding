import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from core.mlx_qwen_backend import MLXBackendConfig, MLXQwenDraftBackend
from client.http_cloud_client import HTTPCloudClient
from core.protocol import SpecExtendTreeRequest

MODEL_PATH = Path(__file__).parent / "models" / "Qwen3-0.6B-4bit-AWQ"
backend = MLXQwenDraftBackend(MLXBackendConfig(model_path=MODEL_PATH))
print("Backend loaded")

prompt = "The quick brown fox jumps over the lazy dog."
input_ids = list(backend.tokenizer.encode(prompt))
print(f"Prompt tokens: {len(input_ids)}")

result = backend.build_draft_tree(
    verified_prefix=input_ids,
    correction_token_id=None,
    nodes=4,
    threshold=0.7,
    max_depth=4,
)
print(f"Draft tree: {len(result.tree.input_ids)} nodes, {result.draft_time_ms:.1f}ms")
print(f"Draft tokens: {result.tree.input_ids}")
print(f"Decoded draft: {repr(backend.tokenizer.decode(result.tree.input_ids))}")

client = HTTPCloudClient("http://172.20.10.5:6007")
resp = client.verify_specextend_tree(SpecExtendTreeRequest(
    request_id="smoke-001",
    prefix_ids=input_ids,
    tree_input_ids=result.tree.input_ids,
    tree_position_ids=result.tree.position_ids,
    parent_indices=result.tree.parent_indices,
    tree_attention_mask=result.tree.attention_mask,
))
print(f"Cloud response: accepted_len={resp.accepted_len}, correction={resp.correction_token_id}, rtt={resp.rtt_ms:.1f}ms")
print("SMOKE TEST PASSED")
