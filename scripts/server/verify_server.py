import argparse
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.protocol import SpecExtendRequest
from core.qwen_specextend_backend import (
    QwenBackendConfig,
    QwenSpecExtendTargetBackend,
    dtype_from_env,
)

logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


_DEFAULT_MODEL_PATH = os.getenv("VERIFY_MODEL_PATH", "/root/autodl-tmp/Qwen3-14B-AWQ")


class VerifyRequest(BaseModel):
    request_id: str
    prefix_ids: List[int]
    draft_ids: List[int]


class VerifyResponse(BaseModel):
    request_id: str
    accepted_len: int
    correction_token_id: Optional[int]
    server_verify_time_ms: float
    model_time_ms: Optional[float] = None
    http_overhead_ms: Optional[float] = None


class SpecExtendVerifyRequest(BaseModel):
    request_id: str
    prefix_ids: List[int]
    draft_ids: List[int]
    draft_position_ids: Optional[List[int]] = None
    parent_indices: Optional[List[int]] = None
    draft_attention_mask: Optional[List[List[int]]] = None
    retrieve_attn_scores: bool = False
    retrieval_chunk_size: int = 64
    retrieve_top_k: int = 16
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpecExtendVerifyResponse(BaseModel):
    request_id: str
    accepted_len: int
    correction_token_id: Optional[int]
    accepted_indices: List[int]
    server_verify_time_ms: float
    target_attn_scores: Optional[List[float]] = None
    selected_chunk_ids: Optional[List[int]] = None
    model_time_ms: Optional[float] = None
    http_overhead_ms: Optional[float] = None
    cloud_observability: Optional[Dict[str, Any]] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time_ms: float


_verifier: Optional[QwenSpecExtendTargetBackend] = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _verifier
    model_path = Path(_env("VERIFY_MODEL_PATH", _DEFAULT_MODEL_PATH))
    gpu_id = _env("VERIFY_GPU_ID", "0")
    device = _env("VERIFY_DEVICE", f"cuda:{gpu_id}")
    dtype = dtype_from_env(_env("VERIFY_DTYPE", "fp16"))
    max_len = int(_env("VERIFY_MAX_LEN", "32768"))
    attn_implementation = _env("VERIFY_ATTN_IMPLEMENTATION", "sdpa")

    logger.info("Loading custom Qwen SpecExtend target backend: %s", model_path)
    _verifier = QwenSpecExtendTargetBackend(
        QwenBackendConfig(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_model_len=max_len,
            attn_implementation=attn_implementation,
        )
    )
    logger.info("Custom Qwen target backend loaded")
    yield


app = FastAPI(title="ECSD Custom Qwen Verify Server", version="2.0.0", lifespan=_lifespan)


@app.get("/health")
async def health():
    if _verifier is None:
        raise HTTPException(503, "Not ready")
    return {
        "status": "healthy",
        "model_info": _verifier.get_info(),
        "runtime": _verifier.runtime_debug_info(),
    }


@app.post("/verify", response_model=VerifyResponse)
async def verify_draft(req: VerifyRequest):
    """Compatibility endpoint for linear speculative verification."""
    if _verifier is None:
        raise HTTPException(503, "Not ready")

    start = time.perf_counter()
    try:
        request = SpecExtendRequest(
            request_id=req.request_id,
            prefix_ids=req.prefix_ids,
            draft_ids=req.draft_ids,
            retrieve_attn_scores=False,
        )
        response = _verifier.verify(request)
    except Exception as exc:
        logger.error("VERIFY ERROR: id=%s, error=%s", req.request_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc

    total_ms = (time.perf_counter() - start) * 1000
    return VerifyResponse(
        request_id=req.request_id,
        accepted_len=response.accepted_len,
        correction_token_id=response.correction_token_id,
        server_verify_time_ms=total_ms,
        model_time_ms=response.model_time_ms,
        http_overhead_ms=max(0.0, total_ms - (response.model_time_ms or 0.0)),
    )


@app.post("/specextend/verify", response_model=SpecExtendVerifyResponse)
async def verify_specextend(req: SpecExtendVerifyRequest):
    if _verifier is None:
        raise HTTPException(503, "Not ready")

    try:
        response = _verifier.verify(SpecExtendRequest.from_dict(req.model_dump()))
    except Exception as exc:
        logger.error("SPECEXTEND VERIFY ERROR: id=%s, error=%s", req.request_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"SpecExtend verification failed: {exc}") from exc

    return SpecExtendVerifyResponse(**response.to_dict())


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _verifier is None:
        raise HTTPException(503, "Not ready")
    try:
        return GenerateResponse(**_verifier.generate_text(req.prompt, req.max_tokens, req.temperature))
    except Exception as exc:
        logger.error("GENERATE ERROR: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="ECSD Custom Qwen Verify Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
