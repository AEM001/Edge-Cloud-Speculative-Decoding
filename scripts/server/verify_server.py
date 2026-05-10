"""
Verify server — runs the target (7B) model and exposes an HTTP API
for draft verification and direct generation.

Start with:
    python3 scripts/server/verify_server.py --port 6006

Environment variables (override via .env or shell export):
    VERIFY_MODEL_PATH   — path to the 7B AWQ model dir
    VERIFY_GPU_MEM      — GPU memory utilization (default 0.90)
    VERIFY_MAX_LEN      — max model length (default 4096)
    VERIFY_QUANTIZATION — quantization method (default "awq")
"""

import argparse
import importlib.metadata
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (env-driven so scripts can override without editing code)
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

_DEFAULT_MODEL_PATH = "/root/code/draft/models/Qwen2.5-14B-Instruct-AWQ"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    request_id: str
    prefix_ids: List[int]
    draft_ids: List[int]


class VerifyResponse(BaseModel):
    request_id: str
    accepted_len: int
    correction_token_id: Optional[int]
    server_verify_time_ms: float


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
        tensor_parallel_size: int = 1,
    ):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("Loading verify model: %s", model_path)
        llm_kwargs: Dict[str, Any] = {
            "model": str(path),
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "max_model_len": max_model_len,
            "enable_prefix_caching": _bool_env("VERIFY_ENABLE_PREFIX_CACHING", True),
            "enforce_eager": _bool_env("VERIFY_ENFORCE_EAGER", False),
            "disable_log_stats": False,
            "disable_custom_all_reduce": True,
        }
        if quantization and quantization.lower() != "none":
            llm_kwargs["quantization"] = quantization

        self.llm_kwargs = dict(llm_kwargs)
        self.vllm_version = self._detect_vllm_version()
        self.attention_backend = os.getenv("VLLM_ATTENTION_BACKEND")
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.model_path = str(path)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.quantization = quantization
        self.tensor_parallel_size = tensor_parallel_size
        logger.info("Verify model loaded: %s", model_path)

    @staticmethod
    def _detect_vllm_version() -> str:
        try:
            return importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    def runtime_debug_info(self) -> Dict[str, Any]:
        return {
            "enable_prefix_caching": bool(self.llm_kwargs.get("enable_prefix_caching")),
            "enforce_eager": bool(self.llm_kwargs.get("enforce_eager")),
            "attention_backend": self.attention_backend,
            "vllm_version": self.vllm_version,
        }

    # ------------------------------------------------------------------
    def verify(
        self,
        prefix_ids: List[int],
        draft_ids: List[int],
        temperature: float = 0.0,
    ):
        """
        Greedy verification via a single prefill pass.
        Returns (accepted_len, correction_token_id, ms).

        How it works
        ------------
        Feed ``prefix_ids + draft_ids`` as the prompt with
        ``prompt_logprobs=1`` and ``max_tokens=1``.

        vLLM runs ONE prefill forward pass over all tokens, then ONE
        decode step for the correction token.  Total GPU work:
            prefill(prefix_len + K tokens)  +  1 decode step
        vs the old approach:
            K+1 sequential decode steps  (6× slower, measured empirically)

        Prefix caching means the ``prefix_ids`` KV entries are reused
        from the previous round — only the K draft positions are newly
        computed.

        ``prompt_logprobs`` semantics in vLLM
        -------------------------------------
        ``output.prompt_logprobs[i]`` contains requested logprob entries for
        position ``i``, conditioned on tokens 0..i-1.
        Positions 0..(prefix_len-1) are ``None`` (not requested).
        Positions prefix_len..(prefix_len+K-1) hold the top-1 target
        token we need for greedy verification. This verifier is correct
        for temperature=0 greedy decoding; stochastic speculative sampling
        needs target probabilities and rejection resampling instead.
        """
        t0 = time.time()

        if not draft_ids:
            return 0, None, 0.0

        num_draft = len(draft_ids)
        input_ids = prefix_ids + draft_ids

        sampling_params = SamplingParams(
            temperature=0.0,           # greedy
            max_tokens=1,              # one correction token
            logprobs=1,                # top-1 for the correction position
            prompt_logprobs=1,         # top-1 for each draft position
        )

        from vllm import TokensPrompt
        outputs = self.llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=input_ids)],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        output = outputs[0]
        plp = output.prompt_logprobs   # list len == len(input_ids), None for prefix

        accepted_len = 0
        prefix_len = len(prefix_ids)

        for j, draft_tok in enumerate(draft_ids):
            pos = prefix_len + j
            if plp is None or pos >= len(plp) or plp[pos] is None:
                break
            lp_dict = plp[pos]
            # argmax over the returned top-1 entries at this position
            best_tok = max(
                lp_dict.items(),
                key=lambda kv: kv[1].logprob if hasattr(kv[1], "logprob") else kv[1],
            )[0]
            if best_tok == draft_tok:
                accepted_len += 1
            else:
                break

        # Correction token: what the verify model would generate at the
        # first divergence point.  For accepted_len < K this is the
        # argmax at position prefix_len+accepted_len (already in plp).
        # For accepted_len == K it's the decode output token.
        correction = None
        if accepted_len < num_draft:
            pos = prefix_len + accepted_len
            if plp and pos < len(plp) and plp[pos]:
                correction = max(
                    plp[pos].items(),
                    key=lambda kv: kv[1].logprob if hasattr(kv[1], "logprob") else kv[1],
                )[0]
        else:
            # All K draft tokens accepted — correction is the next token
            if output.outputs and output.outputs[0].token_ids:
                correction = output.outputs[0].token_ids[0]

        ms = (time.time() - t0) * 1000
        return accepted_len, correction, ms

    def get_info(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "quantization": self.quantization,
            "max_model_len": self.llm_kwargs.get("max_model_len"),
            "tensor_parallel_size": self.llm_kwargs.get("tensor_parallel_size"),
            "enable_prefix_caching": self.llm_kwargs.get("enable_prefix_caching"),
            "enforce_eager": self.llm_kwargs.get("enforce_eager"),
            "attention_backend": self.attention_backend,
            "vllm_version": self.vllm_version,
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_verifier: Optional[CloudVerifier] = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _verifier
    model_path = _env("VERIFY_MODEL_PATH", _DEFAULT_MODEL_PATH)
    gpu_mem = float(_env("VERIFY_GPU_MEM", "0.55"))
    max_len = int(_env("VERIFY_MAX_LEN", "4096"))
    quant = _env("VERIFY_QUANTIZATION", "awq")
    tensor_parallel_size = int(_env("VERIFY_TENSOR_PARALLEL_SIZE", "2"))
    _verifier = CloudVerifier(
        model_path=model_path,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_len,
        quantization=quant,
        tensor_parallel_size=tensor_parallel_size,
    )
    yield


app = FastAPI(title="PicoSpec Verify Server", version="1.0.0", lifespan=_lifespan)


@app.get("/health")
async def health():
    if _verifier is None:
        raise HTTPException(503, "Not ready")
    return {"status": "healthy", "model_info": _verifier.get_info()}


@app.post("/verify", response_model=VerifyResponse)
async def verify_draft(req: VerifyRequest):
    if _verifier is None:
        raise HTTPException(503, "Not ready")
    accepted_len, correction, verify_ms = _verifier.verify(
        prefix_ids=req.prefix_ids,
        draft_ids=req.draft_ids,
    )
    return VerifyResponse(
        request_id=req.request_id,
        accepted_len=accepted_len,
        correction_token_id=correction,
        server_verify_time_ms=verify_ms,
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
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
