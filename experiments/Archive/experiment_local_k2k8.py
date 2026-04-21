"""
Local Experiment with Network Simulation: k=2,4,6,8
Tests: 10 easy + 10 hard prompts x (Direct + K=2 + K=4 + K=6 + K=8) = 100 tests
Network regimes: good, medium, bad, bursty
Output: outputs_local_k2k8/results.json
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
from client.network_wrapper import create_network_wrapped_client
from config_local import GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, MODEL_NAME, MODEL_PATH
from draft_generator import VLLMDraftGenerator
from model_manager import VLLMModelManager
from benchmarks.network_simulator import get_network_regime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_TEMPERATURE = 0.0
PROMPT_COUNT = 10
MAX_TOKENS = 128

# K values to test
K_VALUES = [2, 4, 6, 8]

# Network regimes
NETWORK_REGIMES = ["good", "medium", "bad", "bursty"]


@dataclass
class TestResult:
    method: str
    prompt_type: str
    prompt_id: int
    prompt_length: int
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float
    network_regime: str = "none"
    k_value: int = 0
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


def run_experiment_with_regime(server_url: str, regime: str, outputs_dir: Path) -> List[TestResult]:
    """Run experiment with a specific network regime."""
    logger.info("=" * 80)
    logger.info(f"NETWORK REGIME: {regime.upper()}")
    logger.info("=" * 80)
    
    # Create base HTTP client
    base_http_client = create_http_cloud_client(server_url, timeout=120.0)
    
    # Create network-wrapped client with simulation
    cloud_client = create_network_wrapped_client(
        base_http_client,
        regime=regime,
        enable_simulation=True
    )
    
    # Create edge client with the network-wrapped cloud client
    logger.info("[Setup] Loading draft model...")
    model_manager = VLLMModelManager(MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN)
    llm, tokenizer = model_manager.load()
    draft_generator = VLLMDraftGenerator(llm, tokenizer)
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
    total_tests = (len(EASY_PROMPTS) + len(HARD_PROMPTS)) * (1 + len(K_VALUES))  # Direct + K values

    for prompt_type, prompt_list in [("easy", EASY_PROMPTS), ("hard", HARD_PROMPTS)]:
        logger.info("%s", "=" * 60)
        logger.info("Testing %s prompts (%s total)", prompt_type.upper(), len(prompt_list))
        logger.info("%s", "=" * 60)

        for prompt_data in prompt_list:
            prompt_id = prompt_data["id"]
            prompt_text = prompt_data["text"]
            prompt_len = prompt_data["length"]

            logger.info("[%s Prompt %s] Length: %s chars", prompt_type.upper(), prompt_id, prompt_len)

            # Direct test
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
                        network_regime=regime,
                        k_value=0,
                    )
                )
                logger.info("      %s tokens, %.0fms, %.2f tok/s", tokens, time_ms, tokens / (time_ms / 1000))
            time.sleep(0.5)

            # Speculative tests for each K value
            for k in K_VALUES:
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
                            network_regime=regime,
                            k_value=k,
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
    
    return results


def run_experiment() -> List[TestResult]:
    server_url = "http://localhost:6006"
    outputs_dir = Path(__file__).parent / "outputs_local_k2k8"
    outputs_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("LOCAL EXPERIMENT WITH NETWORK SIMULATION: K=2,4,6,8")
    logger.info("Draft model: %s (%s)", MODEL_NAME, MODEL_PATH)
    logger.info("Verify server: %s", server_url)
    logger.info("K values: %s", K_VALUES)
    logger.info("Network regimes: %s", NETWORK_REGIMES)
    logger.info("Temperature: %.1f (greedy)", BENCHMARK_TEMPERATURE)
    logger.info("=" * 80)

    all_results: List[TestResult] = []
    
    # Run experiment for each network regime
    for regime in NETWORK_REGIMES:
        regime_results = run_experiment_with_regime(server_url, regime, outputs_dir)
        all_results.extend(regime_results)
        
        # Save intermediate results after each regime
        intermediate_file = outputs_dir / f"results_{regime}.json"
        with open(intermediate_file, "w") as f:
            json.dump({
                "regime": regime,
                "results": [asdict(result) for result in regime_results]
            }, f, indent=2)
        logger.info("Intermediate results saved to: %s", intermediate_file)
    
    # Compute summary statistics
    analysis: Dict[Tuple[str, str, str], List[TestResult]] = {}
    for result in all_results:
        key = (result.prompt_type, result.method, result.network_regime)
        analysis.setdefault(key, []).append(result)

    summary = {}
    for prompt_type in ["easy", "hard"]:
        for method in ["direct"] + [f"spec_k{k}" for k in K_VALUES]:
            for regime in NETWORK_REGIMES:
                key = (prompt_type, method, regime)
                entries = analysis.get(key, [])
                if not entries:
                    continue
                summary_key = f"{prompt_type}_{method}_{regime}"
                summary[f"{summary_key}_tps"] = float(np.mean([entry.tokens_per_second for entry in entries]))
                if method != "direct":
                    summary[f"{summary_key}_accept"] = float(np.mean([entry.acceptance_rate for entry in entries]))
                    summary[f"{summary_key}_rtt"] = float(np.mean([entry.avg_rtt_ms for entry in entries]))

    # Save final results
    output_file = outputs_dir / "comprehensive_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "k_values": K_VALUES,
                "network_regimes": NETWORK_REGIMES,
                "prompt_count": PROMPT_COUNT,
                "max_tokens": MAX_TOKENS,
                "temperature": BENCHMARK_TEMPERATURE,
                "draft_model": str(MODEL_NAME),
                "verify_server": server_url,
            },
            "results": [asdict(result) for result in all_results],
            "summary": summary
        }, f, indent=2)

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("Results saved to: %s", output_file)
    logger.info("=" * 80)
    
    return all_results


if __name__ == "__main__":
    run_experiment()
