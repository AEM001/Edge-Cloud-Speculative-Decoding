"""
Asynchronous Edge Client — PicoSpec-style parallel drafting pipeline.

Mirrors the PicoSpec §3.2 Parallel Drafting algorithm on top of the
existing EdgeClient / VLLMDraftGenerator infrastructure.

Key difference from EdgeClient (sync):
  Sync:  [draft S0] → [verify S0] → [draft S1] → [verify S1] → ...
         Latency = T_draft + T_verify per round.

  Async: [draft S0] ─────────────────────────────────────────► ...
                         [verify S0] → result ────────────────► ...
         Latency = max(T_draft, T_verify) per round (ideal, full-hit).

The verifier runs in a background thread, pulling from a draft_queue and
pushing results to a result_queue.  The drafter submits a batch and
immediately assumes it is fully accepted, advances the speculative prefix,
and drafts the next batch — without waiting.

On full hit  : committed prefix advances, pipeline stays warm.
On rejection : speculative prefix rolls back to verified point, all
               in-flight slots are flushed.

Pipeline metrics tracked (beyond what sync EdgeClient tracks):
  - pipeline_efficiency   : fraction of rounds that were full hits
  - avg_bubble_ms         : avg time drafter had to block waiting for result
  - async_speedup_ratio   : measured wall-clock speedup vs equivalent sync
  - prefetch_waste_ratio  : fraction of speculatively drafted tokens discarded
  - slot_hit_rate         : per-slot full-acceptance rate
  - overlap_time_ms       : total time where draft + verify ran concurrently
"""

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from core.protocol import CloudResponse, DraftRequest, DraftResponse, EdgeRequest, TokenInfo
from core.draft_generator import VLLMDraftGenerator
from core.model_manager import VLLMModelManager

logger = logging.getLogger(__name__)

_STOP = object()


# ---------------------------------------------------------------------------
# Slot state
# ---------------------------------------------------------------------------

class SlotState(Enum):
    PENDING  = auto()   # submitted to verifier, not yet resolved
    HIT      = auto()   # full acceptance — all K tokens accepted
    REJECTED = auto()   # partial/zero acceptance — rollback needed
    FLUSHED  = auto()   # invalidated by upstream rejection


@dataclass
class PipelineSlot:
    """One speculative batch in the async pipeline."""
    slot_id: int
    assumed_prefix: List[int]   # prefix used when drafting started
    draft_ids: List[int]        # drafted token IDs
    draft_logprobs: List[float]
    draft_time_ms: float
    state: SlotState = SlotState.PENDING

    # filled after verification
    accepted_len: int = 0
    correction_token: Optional[int] = None
    verify_time_ms: float = 0.0
    rtt_ms: float = 0.0


# ---------------------------------------------------------------------------
# Async-specific metrics
# ---------------------------------------------------------------------------

@dataclass
class AsyncRequestMetrics:
    """
    Full metrics for one async speculative decoding run.
    Superset of what sync EdgeClient.RequestMetrics tracks, plus
    pipeline-specific fields.
    """
    request_id: str
    prompt: str

    # ── Core throughput ───────────────────────────────────────────────────
    total_latency_ms: float = 0.0
    generated_tokens: int = 0
    tokens_per_second: float = 0.0

    # ── Round / token counts ─────────────────────────────────────────────
    total_rounds: int = 0               # number of draft batches submitted
    total_drafted_tokens: int = 0
    total_accepted_tokens: int = 0
    total_rollbacks: int = 0
    acceptance_ratio: float = 0.0       # accepted / drafted
    mean_k_chosen: float = 0.0

    # ── Timing breakdown ─────────────────────────────────────────────────
    total_edge_draft_time_ms: float = 0.0
    total_server_verify_time_ms: float = 0.0
    total_network_time_ms: float = 0.0
    average_rtt_ms: float = 0.0

    # ── Async-specific pipeline metrics ──────────────────────────────────
    pipeline_efficiency: float = 0.0
    """Fraction of rounds that were full hits (no rollback)."""

    avg_bubble_ms: float = 0.0
    """Average time per round the drafter blocked waiting for the verifier.
    Zero in an ideal pipeline; positive when verify is slower than draft."""

    total_bubble_ms: float = 0.0

    overlap_time_ms: float = 0.0
    """Total wall-clock time during which draft and verify ran concurrently."""

    prefetch_waste_ratio: float = 0.0
    """Fraction of speculatively pre-drafted tokens that were thrown away
    due to rollbacks.  Lower is better; 0 = never rolled back."""

    slot_hit_rate: float = 0.0
    """Same as pipeline_efficiency (alias for clarity in reports)."""

    async_speedup_vs_sync: float = 0.0
    """Theoretical speedup: (T_draft + T_verify) / max(T_draft, T_verify).
    Computed from mean per-round timings; > 1 means async helped."""

    # ── Per-slot details ─────────────────────────────────────────────────
    slot_details: List[Dict[str, Any]] = field(default_factory=list)

    # ── Network ──────────────────────────────────────────────────────────
    uplink_bytes: int = 0
    downlink_bytes: int = 0

    def compute_derived(self):
        if self.total_drafted_tokens > 0:
            self.acceptance_ratio = self.total_accepted_tokens / self.total_drafted_tokens
        if self.total_rounds > 0:
            self.mean_k_chosen = self.total_drafted_tokens / self.total_rounds
            self.avg_bubble_ms = self.total_bubble_ms / self.total_rounds
        if self.total_latency_ms > 0:
            self.tokens_per_second = 1000 * self.generated_tokens / self.total_latency_ms

        hits = sum(1 for s in self.slot_details if s.get("full_hit"))
        self.pipeline_efficiency = hits / len(self.slot_details) if self.slot_details else 0.0
        self.slot_hit_rate = self.pipeline_efficiency

        wasted = sum(s.get("wasted_tokens", 0) for s in self.slot_details)
        pre_drafted = sum(s.get("drafted", 0) for s in self.slot_details)
        self.prefetch_waste_ratio = wasted / pre_drafted if pre_drafted > 0 else 0.0

        mean_draft = self.total_edge_draft_time_ms / self.total_rounds if self.total_rounds else 0
        mean_verify = self.total_server_verify_time_ms / self.total_rounds if self.total_rounds else 0
        sync_time = mean_draft + mean_verify
        async_time = max(mean_draft, mean_verify)
        self.async_speedup_vs_sync = sync_time / async_time if async_time > 0 else 1.0


# ---------------------------------------------------------------------------
# Verifier worker payload types
# ---------------------------------------------------------------------------

@dataclass
class _VerifyJob:
    slot_id: int
    request: EdgeRequest


@dataclass
class _VerifyResult:
    slot_id: int
    cloud_response: CloudResponse
    rtt_ms: float


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AsyncEdgeClient:
    """
    Asynchronous edge client implementing the PicoSpec Parallel Drafting
    pipeline.

    Drop-in replacement for EdgeClient for benchmarking purposes.  Uses the
    same VLLMDraftGenerator and cloud_client callable so all existing
    infrastructure (HTTPCloudClient, mock server) works unchanged.

    Parameters
    ----------
    model_manager    : VLLMModelManager (needed to access tokenizer)
    draft_generator  : VLLMDraftGenerator
    cloud_client     : callable (EdgeRequest) -> CloudResponse
    max_new_tokens   : hard token budget
    temperature      : sampling temperature
    lookahead        : max in-flight verification requests (1 = PicoSpec baseline)
    eos_token_id     : stop token (auto-detected from tokenizer if None)
    """

    def __init__(
        self,
        model_manager: VLLMModelManager,
        draft_generator: VLLMDraftGenerator,
        cloud_client: Callable[[EdgeRequest], CloudResponse],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        lookahead: int = 1,
        eos_token_id: Optional[int] = None,
    ):
        self.model_manager = model_manager
        self.draft_generator = draft_generator
        self.cloud_client = cloud_client
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.lookahead = lookahead
        self.tokenizer = draft_generator.tokenizer
        self.eos_token_id = eos_token_id
        if self.eos_token_id is None and self.tokenizer:
            self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

    # ------------------------------------------------------------------
    # Public API (mirrors EdgeClient.generate signature)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        policy: Callable[[int, List[TokenInfo]], int],
        policy_name: str = "AsyncK",
    ) -> AsyncRequestMetrics:
        """
        Run async speculative decoding and return rich metrics.

        Args:
            prompt      : input text
            policy      : callable(round_id, []) -> K  (same as EdgeClient)
            policy_name : label for logging

        Returns:
            AsyncRequestMetrics
        """
        request_id = str(uuid.uuid4())
        metrics = AsyncRequestMetrics(request_id=request_id, prompt=prompt)

        prompt_ids = self.tokenizer.encode(prompt)
        committed_prefix: List[int] = list(prompt_ids)
        speculative_prefix: List[int] = list(prompt_ids)

        # {slot_id -> PipelineSlot}
        in_flight: Dict[int, PipelineSlot] = {}
        next_slot_id = 0

        draft_queue: queue.Queue = queue.Queue()
        result_queue: queue.Queue = queue.Queue()

        verifier_thread = threading.Thread(
            target=self._verifier_worker,
            args=(draft_queue, result_queue),
            daemon=True,
            name="async-verifier",
        )
        verifier_thread.start()

        wall_start = time.perf_counter()

        try:
            self._pipeline_loop(
                prompt_ids=prompt_ids,
                committed_prefix=committed_prefix,
                speculative_prefix=speculative_prefix,
                in_flight=in_flight,
                next_slot_id_ref=[next_slot_id],
                draft_queue=draft_queue,
                result_queue=result_queue,
                policy=policy,
                policy_name=policy_name,
                request_id=request_id,
                metrics=metrics,
            )
        finally:
            draft_queue.put(_STOP)
            verifier_thread.join(timeout=120)

        metrics.total_latency_ms = (time.perf_counter() - wall_start) * 1000
        metrics.generated_tokens = len(committed_prefix) - len(prompt_ids)
        metrics.compute_derived()

        logger.info(
            "Async generation done: %d tokens in %.0fms  %.2f tok/s  "
            "pipeline_eff=%.1f%%  rollbacks=%d  bubble=%.1fms",
            metrics.generated_tokens,
            metrics.total_latency_ms,
            metrics.tokens_per_second,
            100 * metrics.pipeline_efficiency,
            metrics.total_rollbacks,
            metrics.avg_bubble_ms,
        )
        return metrics

    # ------------------------------------------------------------------
    # Background verifier worker
    # ------------------------------------------------------------------

    def _verifier_worker(
        self,
        draft_queue: queue.Queue,
        result_queue: queue.Queue,
    ) -> None:
        while True:
            job = draft_queue.get()
            if job is _STOP:
                break
            assert isinstance(job, _VerifyJob)
            t0 = time.perf_counter()
            try:
                resp: CloudResponse = self.cloud_client(job.request)
            except Exception as exc:
                logger.error("Verifier error on slot %d: %s", job.slot_id, exc)
                # Synthesise a zero-acceptance response so pipeline can continue
                resp = CloudResponse(
                    request_id=job.request.request_id,
                    round_id=job.request.round_id,
                    accepted_len=0,
                    accepted_token_ids=[],
                    correction_token_id=None,
                    server_verify_time_ms=0.0,
                    server_total_time_ms=0.0,
                )
            rtt_ms = (time.perf_counter() - t0) * 1000
            result_queue.put(_VerifyResult(slot_id=job.slot_id, cloud_response=resp, rtt_ms=rtt_ms))

    # ------------------------------------------------------------------
    # Core pipeline loop
    # ------------------------------------------------------------------

    def _apply_result(
        self,
        vr: "_VerifyResult",
        slot: "PipelineSlot",
        committed_prefix: List[int],
        speculative_prefix: List[int],
        in_flight: "Dict[int, PipelineSlot]",
        metrics: "AsyncRequestMetrics",
    ) -> bool:
        """
        Apply one verification result to committed_prefix.

        Returns True if a rollback occurred (caller must stop draining
        and re-draft from the corrected committed_prefix).

        Committed-prefix advancement rule
        ----------------------------------
        A slot can only advance the committed prefix if it starts exactly
        where the committed prefix currently ends — i.e. it is the *next*
        slot in logical order.  Slots that assumed a prefix beyond the
        current committed point are downstream of a rollback and are
        discarded.
        """
        resp = vr.cloud_response
        accepted = resp.accepted_len
        drafted  = len(slot.draft_ids)
        full_hit = (accepted == drafted)

        slot.accepted_len     = accepted
        slot.correction_token = resp.correction_token_id
        slot.verify_time_ms   = resp.server_verify_time_ms
        slot.rtt_ms           = vr.rtt_ms

        metrics.total_server_verify_time_ms += resp.server_verify_time_ms
        metrics.total_network_time_ms       += max(0.0, vr.rtt_ms - resp.server_verify_time_ms)
        n = max(metrics.total_rounds, 1)
        metrics.average_rtt_ms = (metrics.average_rtt_ms * (n - 1) + vr.rtt_ms) / n
        metrics.uplink_bytes   += len(resp.accepted_token_ids) * 4 + 32
        metrics.downlink_bytes += 32

        # Only advance committed prefix if this slot is the immediate next
        slot_start = len(slot.assumed_prefix)
        committed_end = len(committed_prefix)

        if slot_start != committed_end:
            # This slot is downstream of an already-processed rollback — skip
            metrics.slot_details.append({
                "slot_id": vr.slot_id, "drafted": drafted, "accepted": 0,
                "full_hit": False, "wasted_tokens": drafted,
                "draft_ms": slot.draft_time_ms, "verify_ms": slot.verify_time_ms,
                "rtt_ms": slot.rtt_ms, "rollback": False,
                "verify_prefix_len": resp.prefix_len,
                "verify_draft_len": resp.draft_len,
                "verify_input_len": resp.input_len,
                "prompt_logprobs_requested": resp.prompt_logprobs_requested,
                "verify_batch_size": resp.verify_batch_size,
                "enable_prefix_caching": resp.enable_prefix_caching,
                "enforce_eager": resp.enforce_eager,
                "attention_backend": resp.attention_backend,
                "vllm_version": resp.vllm_version,
            })
            return False

        if full_hit:
            slot.state = SlotState.HIT
            committed_prefix.extend(slot.draft_ids)
            metrics.total_accepted_tokens += accepted
            metrics.slot_details.append({
                "slot_id": vr.slot_id, "drafted": drafted, "accepted": accepted,
                "full_hit": True, "wasted_tokens": 0,
                "draft_ms": slot.draft_time_ms, "verify_ms": slot.verify_time_ms,
                "rtt_ms": slot.rtt_ms, "rollback": False,
                "verify_prefix_len": resp.prefix_len,
                "verify_draft_len": resp.draft_len,
                "verify_input_len": resp.input_len,
                "prompt_logprobs_requested": resp.prompt_logprobs_requested,
                "verify_batch_size": resp.verify_batch_size,
                "enable_prefix_caching": resp.enable_prefix_caching,
                "enforce_eager": resp.enforce_eager,
                "attention_backend": resp.attention_backend,
                "vllm_version": resp.vllm_version,
            })
            logger.debug("Slot %d: HIT accepted=%d committed=%d",
                         vr.slot_id, accepted, len(committed_prefix))
            return False
        else:
            slot.state = SlotState.REJECTED
            new_committed = slot.assumed_prefix + slot.draft_ids[:accepted]
            if resp.correction_token_id is not None:
                new_committed.append(resp.correction_token_id)

            wasted_downstream = sum(len(s.draft_ids) for s in in_flight.values())
            metrics.total_accepted_tokens += accepted
            metrics.total_rollbacks       += 1
            metrics.slot_details.append({
                "slot_id": vr.slot_id, "drafted": drafted, "accepted": accepted,
                "full_hit": False,
                "wasted_tokens": (drafted - accepted) + wasted_downstream,
                "draft_ms": slot.draft_time_ms, "verify_ms": slot.verify_time_ms,
                "rtt_ms": slot.rtt_ms, "rollback": True,
                "verify_prefix_len": resp.prefix_len,
                "verify_draft_len": resp.draft_len,
                "verify_input_len": resp.input_len,
                "prompt_logprobs_requested": resp.prompt_logprobs_requested,
                "verify_batch_size": resp.verify_batch_size,
                "enable_prefix_caching": resp.enable_prefix_caching,
                "enforce_eager": resp.enforce_eager,
                "attention_backend": resp.attention_backend,
                "vllm_version": resp.vllm_version,
            })
            logger.debug("Slot %d: REJECT accepted=%d/%d correction=%s wasted_downstream=%d",
                         vr.slot_id, accepted, drafted, resp.correction_token_id, wasted_downstream)

            # Advance committed to rollback point
            committed_prefix.clear()
            committed_prefix.extend(new_committed)

            # Reset speculative prefix and flush all downstream slots
            for s in in_flight.values():
                s.state = SlotState.FLUSHED
            in_flight.clear()
            speculative_prefix.clear()
            speculative_prefix.extend(committed_prefix)
            return True   # rollback — caller must re-draft

    def _pipeline_loop(
        self,
        prompt_ids: List[int],
        committed_prefix: List[int],
        speculative_prefix: List[int],
        in_flight: Dict[int, PipelineSlot],
        next_slot_id_ref: List[int],
        draft_queue: queue.Queue,
        result_queue: queue.Queue,
        policy: Callable,
        policy_name: str,
        request_id: str,
        metrics: AsyncRequestMetrics,
    ) -> None:
        max_tokens = self.max_new_tokens
        eos        = self.eos_token_id

        def _done() -> bool:
            if len(committed_prefix) - len(prompt_ids) >= max_tokens:
                return True
            if eos and eos in committed_prefix[len(prompt_ids):]:
                return True
            return False

        while not _done():

            # ── Phase A: drain any already-available results (non-blocking)
            while True:
                try:
                    vr: _VerifyResult = result_queue.get_nowait()
                except queue.Empty:
                    break
                slot = in_flight.pop(vr.slot_id, None)
                if slot is None:
                    continue
                rollback = self._apply_result(
                    vr, slot, committed_prefix, speculative_prefix, in_flight, metrics
                )
                if rollback:
                    break   # speculative_prefix reset; re-draft from correct base

            if _done():
                break

            # ── Phase B: draft the next batch NOW (overlaps in-flight verify)
            # Draft runs on GPU while the previous verify call travels over
            # the network — this is where the latency hiding happens.
            slot_id = next_slot_id_ref[0]
            next_slot_id_ref[0] += 1
            K = policy(slot_id, [])

            draft_t0 = time.perf_counter()
            draft_req = DraftRequest(
                verified_prefix=list(speculative_prefix),
                num_draft_tokens=K,
            )
            draft_resp: DraftResponse = self.draft_generator.generate_draft_tokens(
                draft_req, temperature=self.temperature,
            )
            draft_ms = (time.perf_counter() - draft_t0) * 1000

            if not draft_resp.draft_token_ids:
                logger.warning("Empty draft at slot %d — stopping", slot_id)
                break

            # ── Phase C: if at max in-flight, wait for one result before
            # submitting (keeps at most `lookahead` outstanding verifies)
            if len(in_flight) >= self.lookahead:
                bubble_t0 = time.perf_counter()
                vr = result_queue.get()   # blocking — draft already done above
                metrics.total_bubble_ms += (time.perf_counter() - bubble_t0) * 1000

                slot = in_flight.pop(vr.slot_id, None)
                if slot is not None:
                    rollback = self._apply_result(
                        vr, slot, committed_prefix, speculative_prefix, in_flight, metrics
                    )
                    if rollback:
                        # Speculative prefix was wrong — discard this draft,
                        # loop back to re-draft from the corrected prefix.
                        # (The slot_id is already incremented; that's fine —
                        # slot IDs are just labels, not indices.)
                        continue

            if _done():
                break

            # ── Phase D: submit the prepared draft to the verifier ────
            slot = PipelineSlot(
                slot_id=slot_id,
                assumed_prefix=list(speculative_prefix),
                draft_ids=draft_resp.draft_token_ids,
                draft_logprobs=draft_resp.logprobs,
                draft_time_ms=draft_ms,
            )
            in_flight[slot_id] = slot

            metrics.total_rounds             += 1
            metrics.total_drafted_tokens     += len(draft_resp.draft_token_ids)
            metrics.total_edge_draft_time_ms += draft_ms

            edge_req = EdgeRequest(
                request_id=request_id,
                round_id=slot_id,
                prefix_ids=list(speculative_prefix),
                draft_ids=draft_resp.draft_token_ids,
                draft_logprobs=draft_resp.logprobs,
                edge_draft_time_ms=draft_ms,
                policy_metadata={"policy_name": policy_name, "K": K},
            )
            metrics.uplink_bytes += (len(speculative_prefix) + len(draft_resp.draft_token_ids)) * 4 + 64
            draft_queue.put(_VerifyJob(slot_id=slot_id, request=edge_req))

            # Optimistically advance speculative prefix (assume full hit)
            speculative_prefix.extend(draft_resp.draft_token_ids)

            logger.debug(
                "Slot %d drafted K=%d  spec=%d  committed=%d  in_flight=%d",
                slot_id, K, len(speculative_prefix),
                len(committed_prefix), len(in_flight),
            )

        # ── Drain any remaining in-flight slots ───────────────────────
        while in_flight:
            try:
                vr = result_queue.get(timeout=120)
            except queue.Empty:
                logger.warning("Timeout draining in-flight slots")
                break
            slot = in_flight.pop(vr.slot_id, None)
            if slot is None:
                continue
            rollback = self._apply_result(
                vr, slot, committed_prefix, speculative_prefix, in_flight, metrics
            )
            if rollback:
                break   # remaining slots already flushed by _apply_result
