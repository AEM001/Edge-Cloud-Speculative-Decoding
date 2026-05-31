import sys, json, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from core.mlx_qwen_backend import MLXBackendConfig, MLXQwenDraftBackend
from core.protocol import SpecExtendTreeRequest

MODEL_PATH = Path(__file__).parent / "models" / "Qwen3-0.6B-4bit-AWQ"
backend = MLXQwenDraftBackend(MLXBackendConfig(model_path=MODEL_PATH))

with open(Path(__file__).parent / "data" / "pg-19" / "pg19_2K.jsonl") as f:
    row = json.loads(f.readline())

text = row["text"]
input_ids = list(backend.tokenizer.encode(text))[:2048]
print(f"Prefix len: {len(input_ids)} tokens")

result = backend.build_draft_tree(
    verified_prefix=input_ids,
    correction_token_id=None,
    nodes=4,
    threshold=0.7,
    max_depth=4,
)
print(f"Draft: {len(result.tree.input_ids)} nodes, {result.draft_time_ms:.0f}ms")

session = requests.Session()
session.trust_env = False
req = SpecExtendTreeRequest(
    request_id="debug-long",
    prefix_ids=input_ids,
    tree_input_ids=result.tree.input_ids,
    tree_position_ids=result.tree.position_ids,
    parent_indices=result.tree.parent_indices,
    tree_attention_mask=result.tree.attention_mask,
)
resp = session.post("http://172.20.10.5:6007/specextend/verify",
    json=req.to_dict(), timeout=120)
print("status:", resp.status_code)
if resp.status_code != 200:
    print("error:", resp.text[:1000])
else:
    data = resp.json()
    print(f"accepted_len={data['accepted_len']}, correction={data['correction_token_id']}, server_ms={data['server_verify_time_ms']:.0f}ms")
