"""
Verify server — runs the target (7B) model and exposes an HTTP API
for draft verification and direct generation.

Start with:
    python -m server.verify_server --port 6006

Environment variables (override via .env or shell export):
    VERIFY_MODEL_PATH   — path to the 7B AWQ model dir
    VERIFY_GPU_MEM      — GPU memory utilization (default 0.90)
    VERIFY_MAX_LEN      — max model length (default 4096)
    VERIFY_QUANTIZATION — quantization method (default "awq")
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (env-driven so scripts can override without editing code)
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.getenv(name, default)

_DEFAULT_MODEL_PATH = "/root/code/Qwen2.5-7B-Instruct-AWQ"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    request_id: str
    round_id: int
    prefix_ids: List[int]
    draft_ids: List[int]
    draft_logprobs: List[float]
    edge_draft_time_ms: float
    policy_metadata: Dict[str, Any] = {}


class VerifyResponse(BaseModel):
    request_id: str
    round_id: int
    accepted_len: int
    accepted_token_ids: List[int]
    correction_token_id: Optional[int]
    server_verify_time_ms: float
    server_total_time_ms: float
    rtt_ms: Optional[float] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time_ms: float


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class CloudVerifier:
    """Loads the target model and performs greedy token-level verification."""

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        quantization: str = "awq",
    ):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("Loading verify model: %s", model_path)
        llm_kwargs: Dict[str, Any] = {
            "model": str(path),
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "max_model_len": max_model_len,
            "enforce_eager": True,
            "disable_log_stats": True,
        }
        if quantization and quantization.lower() != "none":
            llm_kwargs["quantization"] = quantization

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.model_path = str(path)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.quantization = quantization
        logger.info("Verify model loaded: %s", model_path)

    # ------------------------------------------------------------------
    def verify(
        self,
        prefix_ids: List[int],
        draft_ids: List[int],
        temperature: float = 0.0,
    ):
        """
        Greedy verification.  Returns (accepted_len, accepted_ids, correction, ms).
        """
        t0 = time.time()

        if not draft_ids:
            return 0, [], None, 0.0

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=1,
            logprobs=len(draft_ids) + 1,
            prompt_logprobs=len(draft_ids),
        )

        input_ids = prefix_ids + draft_ids
        prompt_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        outputs = self.llm.generate(
            prompts=[prompt_text],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        output = outputs[0]
        prompt_logprobs = output.prompt_logprobs

        accepted_len = 0
        prefix_len = len(prefix_ids)

        for i, draft_tok in enumerate(draft_ids):
            pos = prefix_len + i
            if pos < len(prompt_logprobs) and prompt_logprobs[pos]:
                lp_dict = prompt_logprobs[pos]
                best = max(
                    lp_dict.items(),
                    key=lambda kv: kv[1].logprob if hasattr(kv[1], "logprob") else kv[1],
                )[0]
                if best == draft_tok:
                    accepted_len += 1
                else:
                    break
            else:
                break

        accepted_ids = draft_ids[:accepted_len]
        correction = None
        if accepted_len < len(draft_ids):
            if output.outputs:
                correction = output.outputs[0].token_ids[0]

        ms = (time.time() - t0) * 1000
        return accepted_len, accepted_ids, correction, ms

    def get_info(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "quantization": self.quantization,
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="PicoSpec Verify Server", version="1.0.0")
_verifier: Optional[CloudVerifier] = None


@app.on_event("startup")
async def _startup():
    global _verifier
    model_path = _env("VERIFY_MODEL_PATH", _DEFAULT_MODEL_PATH)
    gpu_mem = float(_env("VERIFY_GPU_MEM", "0.90"))
    max_len = int(_env("VERIFY_MAX_LEN", "4096"))
    quant = _env("VERIFY_QUANTIZATION", "awq")
    _verifier = CloudVerifier(
        model_path=model_path,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_len,
        quantization=quant,
    )


@app.get("/health")
async def health():
    if _verifier is None:
        raise HTTPException(503, "Not ready")
    return {"status": "healthy", "model_info": _verifier.get_info()}


@app.post("/verify", response_model=VerifyResponse)
async def verify_draft(req: VerifyRequest):
    if _verifier is None:
        raise HTTPException(503, "Not ready")
    t0 = time.time()
    accepted_len, accepted_ids, correction, verify_ms = _verifier.verify(
        prefix_ids=req.prefix_ids,
        draft_ids=req.draft_ids,
    )
    total_ms = (time.time() - t0) * 1000
    return VerifyResponse(
        request_id=req.request_id,
        round_id=req.round_id,
        accepted_len=accepted_len,
        accepted_token_ids=accepted_ids,
        correction_token_id=correction,
        server_verify_time_ms=verify_ms,
        server_total_time_ms=total_ms,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _verifier is None:
        raise HTTPException(503, "Not ready")
    t0 = time.time()
    sp = SamplingParams(temperature=req.temperature, max_tokens=req.max_tokens)
    outputs = _verifier.llm.generate(
        prompts=[req.prompt], sampling_params=sp, use_tqdm=False
    )
    out = outputs[0].outputs[0]
    ms = (time.time() - t0) * 1000
    return GenerateResponse(
        text=out.text,
        tokens_generated=len(out.token_ids),
        generation_time_ms=ms,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="PicoSpec Verify Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()

    uvicorn.run(
        "server.verify_server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
