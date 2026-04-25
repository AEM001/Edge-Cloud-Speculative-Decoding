import argparse
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class PromptCase:
    prompt_id: int
    text: str
    source_length: int


@dataclass
class PromptMetrics:
    mode: str
    prompt_id: int
    prompt_length: int
    output_text: str
    completion_tokens: int
    prompt_tokens: int
    latency_ms: float
    tokens_per_second: float
    ttft_ms: Optional[float]
    spec_verify_count: Optional[int]
    acceptance_length: Optional[float]


@dataclass
class AggregateMetrics:
    mode: str
    prompt_count: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_tokens_per_second: float
    p50_tokens_per_second: float
    p95_tokens_per_second: float
    mean_completion_tokens: float
    mean_ttft_ms: Optional[float]
    mean_spec_verify_count: Optional[float]
    mean_acceptance_length: Optional[float]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--base-model-path", type=Path, default=root / "models" / "Qwen3-8B")
    parser.add_argument("--eagle-model-path", type=Path, default=root / "models" / "qwen3_8b_eagle3")
    parser.add_argument("--prompt-file", type=Path, default=root / "benchmarks" / "prompts.txt")
    parser.add_argument("--prompt-count", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=2)
    parser.add_argument("--speculative-algorithm", default="EAGLE3")
    parser.add_argument("--speculative-num-steps", type=int, default=6)
    parser.add_argument("--speculative-eagle-topk", type=int, default=10)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=32)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs")
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


def load_prompts(prompt_file: Path, prompt_count: int) -> List[PromptCase]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    content = prompt_file.read_text()
    matches = re.findall(
        r"\[(\d+)\]\s+benchmark_[\d\-]+.*?Actual length:\s*(\d+)\s*chars.*?User:\s*(.*?)(?=\n\n\[|$)",
        content,
        re.DOTALL,
    )
    prompts: List[PromptCase] = []
    for idx, (_, length, text) in enumerate(matches[:prompt_count], 1):
        prompts.append(PromptCase(prompt_id=idx, text=text.strip()[:2000], source_length=int(length)))
    if not prompts:
        raise RuntimeError(f"No prompts parsed from {prompt_file}")
    return prompts


def build_command(args: argparse.Namespace, mode: str) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        str(args.base_model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--cuda-graph-max-bs",
        str(args.cuda_graph_max_bs),
    ]
    if mode == "eagle":
        command.extend(
            [
                "--speculative-algorithm",
                args.speculative_algorithm,
                "--speculative-draft-model-path",
                str(args.eagle_model_path),
                "--speculative-num-steps",
                str(args.speculative_num_steps),
                "--speculative-eagle-topk",
                str(args.speculative_eagle_topk),
                "--speculative-num-draft-tokens",
                str(args.speculative_num_draft_tokens),
            ]
        )
    return command


def tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def wait_for_server(base_url: str, proc: subprocess.Popen[Any], startup_timeout: int, log_file: Path) -> Dict[str, Any]:
    deadline = time.time() + startup_timeout
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"SGLang server exited early with code {proc.returncode}.\n{tail_text(log_file)}"
            )
        try:
            health = requests.get(f"{base_url}/health", timeout=5)
            if health.ok:
                info = requests.get(f"{base_url}/get_server_info", timeout=10)
                if info.ok:
                    return info.json()
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(
        f"Timed out waiting for SGLang server at {base_url}. Last error: {last_error}\n{tail_text(log_file)}"
    )


def stop_server(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def launch_server(args: argparse.Namespace, mode: str, log_dir: Path) -> Tuple[subprocess.Popen[Any], Dict[str, Any], Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{mode}_server.log"
    handle = open(log_file, "w")
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    proc = subprocess.Popen(
        build_command(args, mode),
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
        text=True,
    )
    base_url = f"http://{args.host}:{args.port}"
    try:
        server_info = wait_for_server(base_url, proc, args.startup_timeout, log_file)
    except Exception:
        handle.close()
        raise
    handle.close()
    return proc, server_info, log_file


def request_generation(base_url: str, prompt: str, max_new_tokens: int, temperature: float, timeout: int) -> Dict[str, Any]:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": ["Question", "Assistant:", "<|separator|>", "<|eos|>"],
        },
        "stream": False,
    }
    response = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def metric_from_response(mode: str, prompt_case: PromptCase, response_json: Dict[str, Any]) -> PromptMetrics:
    meta = response_json.get("meta_info", {})
    latency_seconds = float(meta.get("e2e_latency", 0.0))
    completion_tokens = int(meta.get("completion_tokens", 0))
    prompt_tokens = int(meta.get("prompt_tokens", 0))
    ttft_seconds = meta.get("ttft")
    spec_verify_count = meta.get("spec_verify_ct")
    acceptance_length = None
    if spec_verify_count:
        acceptance_length = completion_tokens / float(spec_verify_count)
    tokens_per_second = completion_tokens / latency_seconds if latency_seconds > 0 else 0.0
    return PromptMetrics(
        mode=mode,
        prompt_id=prompt_case.prompt_id,
        prompt_length=prompt_case.source_length,
        output_text=response_json.get("text", ""),
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        latency_ms=latency_seconds * 1000,
        tokens_per_second=tokens_per_second,
        ttft_ms=float(ttft_seconds) * 1000 if ttft_seconds is not None else None,
        spec_verify_count=int(spec_verify_count) if spec_verify_count is not None else None,
        acceptance_length=acceptance_length,
    )


def p95(values: List[float]) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return ordered[index]


def mean_optional(values: List[Optional[float]]) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return float(statistics.mean(filtered))


def aggregate_metrics(mode: str, metrics: List[PromptMetrics]) -> AggregateMetrics:
    latency = [m.latency_ms for m in metrics]
    throughput = [m.tokens_per_second for m in metrics]
    return AggregateMetrics(
        mode=mode,
        prompt_count=len(metrics),
        mean_latency_ms=float(statistics.mean(latency)),
        p50_latency_ms=float(statistics.median(latency)),
        p95_latency_ms=float(p95(latency)),
        mean_tokens_per_second=float(statistics.mean(throughput)),
        p50_tokens_per_second=float(statistics.median(throughput)),
        p95_tokens_per_second=float(p95(throughput)),
        mean_completion_tokens=float(statistics.mean([m.completion_tokens for m in metrics])),
        mean_ttft_ms=mean_optional([m.ttft_ms for m in metrics]),
        mean_spec_verify_count=mean_optional([float(m.spec_verify_count) if m.spec_verify_count is not None else None for m in metrics]),
        mean_acceptance_length=mean_optional([m.acceptance_length for m in metrics]),
    )


def run_mode(args: argparse.Namespace, mode: str, prompts: List[PromptCase]) -> Tuple[List[PromptMetrics], AggregateMetrics, Dict[str, Any]]:
    time.sleep(5)
    proc, server_info, log_file = launch_server(args, mode, args.log_dir)
    metrics: List[PromptMetrics] = []
    base_url = f"http://{args.host}:{args.port}"
    try:
        for prompt_case in prompts:
            logging.info("Running %s prompt %s", mode, prompt_case.prompt_id)
            response_json = request_generation(
                base_url=base_url,
                prompt=prompt_case.text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                timeout=args.request_timeout,
            )
            metrics.append(metric_from_response(mode, prompt_case, response_json))
        aggregate = aggregate_metrics(mode, metrics)
        server_info["server_log"] = str(log_file)
        return metrics, aggregate, server_info
    finally:
        stop_server(proc)


def build_summary(base: AggregateMetrics, eagle: AggregateMetrics) -> Dict[str, Any]:
    throughput_speedup = None
    latency_reduction_ratio = None
    if base.mean_tokens_per_second > 0:
        throughput_speedup = eagle.mean_tokens_per_second / base.mean_tokens_per_second
    if eagle.mean_latency_ms > 0:
        latency_reduction_ratio = base.mean_latency_ms / eagle.mean_latency_ms
    return {
        "throughput_speedup_eagle_vs_base": throughput_speedup,
        "latency_ratio_base_vs_eagle": latency_reduction_ratio,
        "acceptance_length_eagle": eagle.mean_acceptance_length,
        "spec_verify_count_eagle": eagle.mean_spec_verify_count,
        "mean_tps_delta": eagle.mean_tokens_per_second - base.mean_tokens_per_second,
        "mean_latency_ms_delta": eagle.mean_latency_ms - base.mean_latency_ms,
    }


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_dir / f"benchmark_{timestamp}.log")

    prompts = load_prompts(args.prompt_file, args.prompt_count)

    if not args.base_model_path.exists():
        raise FileNotFoundError(f"Base model path not found: {args.base_model_path}")
    if not args.eagle_model_path.exists():
        raise FileNotFoundError(f"Eagle model path not found: {args.eagle_model_path}")

    logging.info("Loaded %d prompts", len(prompts))
    logging.info("Benchmarking base model: %s", args.base_model_path)
    base_metrics, base_aggregate, base_server_info = run_mode(args, "base", prompts)

    logging.info("Benchmarking Eagle speculative setup: %s", args.eagle_model_path)
    eagle_metrics, eagle_aggregate, eagle_server_info = run_mode(args, "eagle", prompts)

    payload = {
        "config": {
            "host": args.host,
            "port": args.port,
            "base_model_path": str(args.base_model_path),
            "eagle_model_path": str(args.eagle_model_path),
            "prompt_file": str(args.prompt_file),
            "prompt_count": args.prompt_count,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "dtype": args.dtype,
            "mem_fraction_static": args.mem_fraction_static,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "speculative_algorithm": args.speculative_algorithm,
            "speculative_num_steps": args.speculative_num_steps,
            "speculative_eagle_topk": args.speculative_eagle_topk,
            "speculative_num_draft_tokens": args.speculative_num_draft_tokens,
        },
        "server_info": {
            "base": base_server_info,
            "eagle": eagle_server_info,
        },
        "aggregates": {
            "base": asdict(base_aggregate),
            "eagle": asdict(eagle_aggregate),
        },
        "comparison": build_summary(base_aggregate, eagle_aggregate),
        "per_prompt": {
            "base": [asdict(m) for m in base_metrics],
            "eagle": [asdict(m) for m in eagle_metrics],
        },
    }

    output_path = args.output_dir / f"sglang_eagle_comparison_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2))
    logging.info("Saved benchmark output to %s", output_path)
    logging.info("Base mean tok/s: %.2f", base_aggregate.mean_tokens_per_second)
    logging.info("Eagle mean tok/s: %.2f", eagle_aggregate.mean_tokens_per_second)
    if payload["comparison"]["throughput_speedup_eagle_vs_base"] is not None:
        logging.info(
            "Eagle speedup vs base: %.2fx",
            payload["comparison"]["throughput_speedup_eagle_vs_base"],
        )


if __name__ == "__main__":
    main()
