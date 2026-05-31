# ECSD Cloud Target

Pure cloud verifier for SpecExtend-style speculative decoding. This machine runs
the **target model only** (`Qwen3-14B-AWQ`). The draft/edge side has been moved
to `draft-edge/` and is intended for a Mac running `Qwen3-0.6B-AWQ` via MLX.

## Network Setup (Phone Hotspot)

1. Enable your phone's personal hotspot.
2. Connect **both** this cloud machine and the Mac draft machine to the hotspot.
3. On the cloud machine, note its local IP:
   ```bash
   hostname -I
   ```
4. The server binds `0.0.0.0:6007` by default, so any device on the hotspot
   can reach it at `http://<cloud-ip>:6007`.

## Model

Default target: `Qwen3-14B-AWQ` (AWQ-int4)

Download it first:

```bash
python models/download.py --model qwen3-14b-awq
```

## Start the Server

```bash
bash start_verify.sh
```

Optional overrides:

```bash
VERIFY_MODEL_PATH=/path/to/Qwen3-14B-AWQ VERIFY_GPU_ID=0 bash start_verify.sh
```

The server exposes:

- `GET /health` — model info and readiness
- `POST /verify` — linear speculative decoding compatibility endpoint
- `POST /specextend/verify` — full tree verification endpoint
- `POST /generate` — direct text generation baseline

## Project Structure

```text
├── scripts/
│   ├── core/
│   │   ├── protocol.py              # Wire protocol dataclasses
│   │   ├── qwen_specextend_backend.py  # Qwen3 target backend only
│   │   └── specextend_retrieval.py  # Chunk selection (used by target)
│   └── server/
│       └── verify_server.py         # FastAPI cloud server
├── draft-edge/                      # COPY THIS TO YOUR MAC
│   ├── README.md
│   ├── requirements-mac.txt
│   └── scripts/
│       ├── client/                  # HTTP client to this cloud server
│       ├── core/                    # Shared protocols + draft backend iface
│       └── experiments/             # Quick test runner
├── models/
│   └── download.py
├── tests/
│   └── test_specextend_core.py
├── test_verify_server_comprehensive.py
├── start_verify.sh
└── pyproject.toml
```

## Draft Edge (Mac)

Copy the `draft-edge/` folder to your Mac. You will need to implement an
**MLX draft backend** that conforms to `SpecExtendDraftBackend`. See
`draft-edge/README.md` for details.

## Testing the Server Locally

```bash
python test_verify_server_comprehensive.py
```

Set `VERIFY_SERVER_URL` if testing from another machine on the hotspot:

```bash
VERIFY_SERVER_URL=http://192.168.43.100:6007 python test_verify_server_comprehensive.py
```
