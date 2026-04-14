"""
Comprehensive Benchmark: greedy K=2, K=4 (Run 4)
Tests: 10 easy + 10 hard prompts x (Direct + K=2 + K=4) = 60 tests
Output: outputs_run4/comprehensive_results.json, comprehensive_log.txt
"""
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from client.edge_client import EdgeClient
from client.http_cloud_client import create_http_cloud_client
from config import GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, MODEL_NAME, MODEL_PATH
from draft_generator import VLLMDraftGenerator
from model_manager import VLLMModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_TEMPERATURE = 0.0
PROMPT_COUNT = 10
MAX_TOKENS = 128


@dataclass
class TestResult:
    method: str
    prompt_type: str
    prompt_id: int
    prompt_length: int
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float
    num_rounds: int = 0
    acceptance_rate: float = 0.0
    avg_draft_ms: float = 0.0
    avg_verify_ms: float = 0.0
    avg_rtt_ms: float = 0.0
    avg_network_ms: float = 0.0
    total_edge_time_ms: float = 0.0
    total_server_time_ms: float = 0.0
    total_network_time_ms: float = 0.0
    server_pure_inference_ms: float = 0.0
    avg_draft_confidence: float = 0.0
    avg_rejection_position: float = 0.0


def load_prompts_from_file(filepath: Path, count: int = PROMPT_COUNT) -> Tuple[List[Dict], List[Dict]]:
    easy_prompts = []
    hard_prompts = []

    with open(filepath, "r") as f:
        content = f.read()

    import re

    easy_section = re.search(r"EASY BENCHMARKS.*?HARD BENCHMARKS", content, re.DOTALL)
    if easy_section:
        easy_matches = re.findall(
            r"\[(\d+)\]\s+benchmark_[\d\-]+.*?Actual length:\s*(\d+)\s*chars.*?User:\s*(.*?)(?=\n\n\[|$)",
            easy_section.group(),
            re.DOTALL,
        )
        for idx, (_, length, text) in enumerate(easy_matches[: count * 2], 1):
            easy_prompts.append({"id": idx, "length": int(length), "text": text.strip()[:400]})

    hard_section = re.search(r"HARD BENCHMARKS.*?(?=$)", content, re.DOTALL)
    if hard_section:
        hard_matches = re.findall(
            r"\[(\d+)\]\s+benchmark_[\d\-]+.*?Actual length:\s*(\d+)\s*chars.*?User:\s*(.*?)(?=\n\n\[|$)",
            hard_section.group(),
            re.DOTALL,
        )
        for idx, (_, length, text) in enumerate(hard_matches[: count * 2], 1):
            hard_prompts.append({"id": idx, "length": int(length), "text": text.strip()[:400]})

    def select_balanced(prompts: List[Dict], target_count: int, target_len: int) -> List[Dict]:
        sorted_prompts = sorted(prompts, key=lambda x: abs(x["length"] - target_len))
        selected = sorted_prompts[:target_count]
        return sorted(selected, key=lambda x: x["id"])

    easy_selected = select_balanced(easy_prompts, count, target_len=300)
    hard_selected = select_balanced(hard_prompts, count, target_len=500)

    for i, prompt in enumerate(easy_selected, 1):
        prompt["id"] = i
    for i, prompt in enumerate(hard_selected, 1):
        prompt["id"] = i

    logger.info(
        "Loaded %s easy prompts (length range: %s-%s)",
        len(easy_selected),
        min(p["length"] for p in easy_selected),
        max(p["length"] for p in easy_selected),
    )
    logger.info(
        "Loaded %s hard prompts (length range: %s-%s)",
        len(hard_selected),
        min(p["length"] for p in hard_selected),
        max(p["length"] for p in hard_selected),
    )
    return easy_selected, hard_selected


PROMPTS_FILE = Path(__file__).parent.parent / "benchmarks" / "prompts.txt"
EASY_PROMPTS, HARD_PROMPTS = load_prompts_from_file(PROMPTS_FILE, count=PROMPT_COUNT)


def test_direct(server_url: str, prompt: str, max_tokens: int = MAX_TOKENS) -> Tuple[int, float]:
    start = time.time()
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(
            f"{server_url}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": BENCHMARK_TEMPERATURE},
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        total_ms = (time.time() - start) * 1000
        return result["tokens_generated"], total_ms
    except Exception as exc:
        logger.error("Direct failed: %s", exc)
        return 0, 0


def test_speculative(
    client: EdgeClient, prompt: str, k: int
) -> Tuple[int, float, int, float, float, float, float, float, float, float, float, float, float, float]:
    start = time.time()
    try:
        metrics = client.generate(prompt=prompt, policy=lambda _round_id, _draft_tokens: k, policy_name=f"StaticK{k}")
        total_ms = (time.time() - start) * 1000
        rounds = metrics.total_rounds
        avg_draft = metrics.total_edge_draft_time_ms / rounds if rounds > 0 else 0
        avg_verify = metrics.total_server_verify_time_ms / rounds if rounds > 0 else 0
        avg_rtt = metrics.average_rtt_ms
        avg_network = metrics.total_network_time_ms / rounds if rounds > 0 else 0

        rejection_positions = []
        for detail in metrics.round_details:
            if detail["accepted"] < detail["drafted"]:
                rejection_positions.append(detail["accepted"])
            elif detail["accepted"] == detail["drafted"] and detail["accepted"] > 0:
                rejection_positions.append(detail["drafted"])

        avg_rejection_pos = sum(rejection_positions) / len(rejection_positions) if rejection_positions else k
        draft_confidence = metrics.acceptance_ratio

        return (
            metrics.generated_tokens,
            total_ms,
            rounds,
            metrics.acceptance_ratio,
            avg_draft,
            avg_verify,
            avg_rtt,
            avg_network,
            metrics.total_edge_draft_time_ms,
            metrics.total_server_verify_time_ms,
            metrics.total_network_time_ms,
            avg_verify,
            draft_confidence,
            avg_rejection_pos,
        )
    except Exception as exc:
        logger.error("Speculative K=%s failed: %s", k, exc)
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def run_experiment() -> List[TestResult]:
    server_url = "http://localhost:6006"
    outputs_dir = Path(__file__).parent / "outputs_run4"
    outputs_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE BENCHMARK RUN 4: greedy K=2, K=4")
    logger.info("Draft model: %s (%s)", MODEL_NAME, MODEL_PATH)
    logger.info("Temperature forced to %.1f for direct and speculative decoding", BENCHMARK_TEMPERATURE)
    logger.info("Methods: Direct, K=2, K=4")
    logger.info(
        "Total tests: %s prompts x 3 methods = %s tests",
        len(EASY_PROMPTS) + len(HARD_PROMPTS),
        (len(EASY_PROMPTS) + len(HARD_PROMPTS)) * 3,
    )
    logger.info("=" * 80)

    logger.info("[Setup] Loading draft model...")
    model_manager = VLLMModelManager(MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
    cloud_client = create_http_cloud_client(server_url, timeout=120.0)
    edge_client = EdgeClient(
        model_manager=model_manager,
        draft_generator=draft_generator,
        cloud_client=cloud_client,
        max_new_tokens=MAX_TOKENS,
        temperature=BENCHMARK_TEMPERATURE,
    )
    logger.info("[Setup] Ready")

    results: List[TestResult] = []
    test_count = 0
    total_tests = (len(EASY_PROMPTS) + len(HARD_PROMPTS)) * 3

    for prompt_type, prompt_list in [("easy", EASY_PROMPTS), ("hard", HARD_PROMPTS)]:
        logger.info("%s", "=" * 60)
        logger.info("Testing %s prompts (%s total)", prompt_type.upper(), len(prompt_list))
        logger.info("%s", "=" * 60)

        for prompt_data in prompt_list:
            prompt_id = prompt_data["id"]
            prompt_text = prompt_data["text"]
            prompt_len = prompt_data["length"]

            logger.info("[%s Prompt %s] Length: %s chars", prompt_type.upper(), prompt_id, prompt_len)

            test_count += 1
            logger.info("  [%s/%s] Direct...", test_count, total_tests)
            tokens, time_ms = test_direct(server_url, prompt_text)
            if tokens > 0:
                results.append(
                    TestResult(
                        method="direct",
                        prompt_type=prompt_type,
                        prompt_id=prompt_id,
                        prompt_length=prompt_len,
                        tokens_generated=tokens,
                        total_time_ms=time_ms,
                        tokens_per_second=tokens / (time_ms / 1000),
                    )
                )
                logger.info("      %s tokens, %.0fms, %.2f tok/s", tokens, time_ms, tokens / (time_ms / 1000))
            time.sleep(0.5)

            for k in (2, 4):
                test_count += 1
                logger.info("  [%s/%s] Speculative K=%s...", test_count, total_tests, k)
                result = test_speculative(edge_client, prompt_text, k=k)
                tokens, time_ms, rounds, acc, draft_ms, verify_ms, rtt_ms, net_ms, tot_edge, tot_srv, tot_net, srv_pure, draft_conf, reject_pos = result
                if tokens > 0:
                    results.append(
                        TestResult(
                            method=f"spec_k{k}",
                            prompt_type=prompt_type,
                            prompt_id=prompt_id,
                            prompt_length=prompt_len,
                            tokens_generated=tokens,
                            total_time_ms=time_ms,
                            tokens_per_second=tokens / (time_ms / 1000),
                            num_rounds=rounds,
                            acceptance_rate=acc,
                            avg_draft_ms=draft_ms,
                            avg_verify_ms=verify_ms,
                            avg_rtt_ms=rtt_ms,
                            avg_network_ms=net_ms,
                            total_edge_time_ms=tot_edge,
                            total_server_time_ms=tot_srv,
                            total_network_time_ms=tot_net,
                            server_pure_inference_ms=srv_pure,
                            avg_draft_confidence=draft_conf,
                            avg_rejection_position=reject_pos,
                        )
                    )
                    logger.info(
                        "      %s tokens, %.0fms, %.2f tok/s | rounds=%s accept=%.1f%% rtt=%.0fms reject_pos=%.1f",
                        tokens,
                        time_ms,
                        tokens / (time_ms / 1000),
                        rounds,
                        acc * 100,
                        rtt_ms,
                        reject_pos,
                    )
                time.sleep(0.5)

    analysis: Dict[Tuple[str, str], List[TestResult]] = {}
    for result in results:
        analysis.setdefault((result.prompt_type, result.method), []).append(result)

    summary = {}
    for prompt_type in ["easy", "hard"]:
        for method in ["direct", "spec_k2", "spec_k4"]:
            entries = analysis.get((prompt_type, method), [])
            if not entries:
                continue
            summary[f"{prompt_type}_{method}"] = float(np.mean([entry.tokens_per_second for entry in entries]))
            if method != "direct":
                summary[f"{prompt_type}_{method}_accept"] = float(np.mean([entry.acceptance_rate for entry in entries]))
                summary[f"{prompt_type}_{method}_rtt"] = float(np.mean([entry.avg_rtt_ms for entry in entries]))

    output_file = outputs_dir / "comprehensive_results.json"
    with open(output_file, "w") as f:
        json.dump({"results": [asdict(result) for result in results], "summary": summary}, f, indent=2)

    logger.info("Results saved to: %s", output_file)
    return results


if __name__ == "__main__":
    run_experiment()
