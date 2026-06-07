# Performance Change Log

## 2026-06-07 — Configurable Tree Draft Recovery

- Restored optional branching-tree draft construction and target-side tree
  verification on top of the current explicit full-KV plus retrieval-working-KV
  path. Linear draft mode remains supported.
- Added `draft_mode`, `draft_tree_nodes`, and `draft_tree_max_depth` to
  `quick_benchmark_config.json` and the quick-test config flow. Defaults are
  `branching`, `32`, and `8`, matching the prior strong tree setup.
- The edge request now carries optional tree metadata (`parent_indices`,
  `draft_position_ids`, `draft_attention_mask`) while preserving linear
  `draft_ids` compatibility.
- Avoided a duplicate working-KV rebuild in ordinary non-retrieval rounds when
  the selected chunks and working indices are unchanged.
- Validation: `python -m compileall scripts/core scripts/client scripts/server scripts/experiments`,
  protocol round-trip smoke test, tree helper smoke test.

## 2026-05-29

- Added PG-19 support with `load_prompts(source="pg19")` and fixed 2048 input tokens via `--prompt-input-tokens`.
- Replaced token-by-token generation with `model.generate(..., use_cache=True)` and added cache rewind on rejection.
- Switched to SDPA attention by default; eager attention used only for retrieval.
- Added retrieval-backed draft context and async edge pipeline with configurable offsets.

## 2026-05-30 01:44

- Switched to system `miniconda3` base environment, installed missing dependencies.
- Changed scripts to use `python3` directly, separated launch scripts (`start_verify.sh`, `run_quick.sh`, `run_2x3090.sh`), changed default port to `6008`.
- Raised `RETRIEVE_EVERY_N_STEPS` default from `4` to `16`.
- Fixed sparse KV cache rebuild: draft tokens are cropped out after each decode pass, only newly accepted tokens are forwarded incrementally.
- Added stale-tail crop in `ensure_sparse_cache` when retrieved-chunk set changes.

## 2026-05-30 02:01 — Measurements (2× RTX 3090, Qwen3-14B-AWQ target @ GPU 0 / Qwen3-1.7B draft @ GPU 1)

Environment: two RTX 3090, Qwen3-14B-AWQ target (GPU 0, eager attention),
Qwen3-1.7B draft (GPU 1, SDPA), PG-19 prompt, 2048 input tokens, 256 output
tokens, nodes=32, depth=8, retrieve_every_n_steps=16.

- Direct (14B-AWQ): **18.6 s, 13.7 tok/s**
- SpecExtend: **10.6 s, 24.4 tok/s** — **1.78× faster than direct**
  - local draft time: 1.8 s (down from 18.9 s before sparse-cache fix, −90%)
  - server verify time: 5.1 s
  - rounds: 37, accepted draft tokens: 223/256, acceptance length: 6.03

### Measurements

Environment: one Tesla V100-SXM2-32GB, Qwen3-8B target, PG-19 prompt,
2048 tokenizer input tokens, 256 output tokens.

- Optimized direct 8B baseline: about 12.2s, 20.9 tok/s.
- Qwen3-0.6B draft, draft length 4, retrieval every 8 rounds:
  about 35.2s, 7.3 tok/s; acceptance length about 1.25.
- Qwen3-1.7B draft, draft length 8, retrieval every 8 rounds:
  about 21.1s, 12.2 tok/s; acceptance length about 4.0.
- Qwen3-1.7B draft, draft length 16, retrieval every 8 rounds:
  about 73.3s, 3.5 tok/s; acceptance length about 1.27.
- Qwen3-1.7B draft, draft length 8, retrieval every 8 rounds, async offsets
  `full,half`: about 43.7s, 5.9 tok/s; acceptance length about 2.39. The
  pipeline hid response wait time, but single-GPU target/draft contention and
  low candidate reuse made it slower in this environment.
- Qwen3-1.7B draft, draft length 8, real target-attention retrieval every 2
  rounds with eager verify attention, async offsets `full,half`, 64 output
  tokens: about 15.2s, 4.35 tok/s; 11 retrieval rounds, selected chunks changed
  from target attention, acceptance length about 1.75.

Conclusion: the backend is now functional and much faster than the original
full-prefix implementation, but on one V100 the optimized direct baseline is
still faster than speculative decoding. The speculative path needs separate
draft/target GPUs, higher draft-target agreement, or tensor-level retrieval KV
compaction before it can beat cached direct generation for this setup.

## 2026-05-31 20:33 — Code Change, Not Yet Benchmarked

- Replaced draft-side sparse replay cache with SpecExtend-style tensor KV cache.
- Added local Qwen3 KV model support (`modeling_qwen3_kv.py`, `qwen_kv_cache.py`).
- Changed draft backend to use local custom Qwen3ForCausalLM for cache mutability.
- Disabled async pipeline prefetch due to uncommitted speculative prefixes.
- Removed DRAFT_RECENT_TOKENS; working cache now uses chunk selection semantics.
- Validation: py_compile and tests pass. No performance benchmark yet.

## 2026-06-01 00:21 — Working Cache Rebuild Optimization + Pipeline Candidate Infrastructure

### Problem diagnosed
After the 2026-05-31 refactor to full-draft-kv + retrieval working-cache,
benchmarks showed severe regression:

- `local_draft_ms`: 18,092 ms (was 1,813 ms, ×10 slower)
- Acceptance length: 2.51/8 (was 6.03/8)
- Throughput: 8.33 tok/s (was 24.42 tok/s)

Root causes:
1. `_rebuild_working_cache` re-allocated GPU tensors every round
2. `select_working_tokens` rebuilt cache even when retrieval selection unchanged
3. `build_draft_tree` called `append_prefix` before `select_working_tokens`
4. `_generate_linear_from_working_cache` finally block called expensive rebuild

### Changes made

- **`_rebuild_working_cache`**: in-place reuse of existing buffer, no allocation
- **`select_working_tokens`**: skip rebuild when chunk selection unchanged
- **Reordered `build_draft_tree`**: `select_working_tokens` before `append_prefix`
- **`_generate_linear_from_working_cache` finally**: snapshot/restore lengths only
- **`build_draft_candidate`**: implemented for async pipeline (kept disabled)

### Measurements (2026-06-01, same hardware as 2026-05-30 02:01)

Environment: 2× RTX 3090, Qwen3-14B-AWQ target (GPU 0, eager), Qwen3-1.7B draft
(GPU 1, SDPA), PG-19, 2048 input tokens, 256 output tokens, nodes=32, depth=8,
chunk_size=64, top_k=16, retrieve_every_n_steps=16.

| | Before (broken, 06-01 00:08) | After (06-01 00:19) | Best prior (05-30) |
|---|---|---|---|
| Throughput | 8.33 tok/s | **12.84 tok/s** | 24.42 tok/s |
| Total time | 30,715 ms | **20,174 ms** | 10,648 ms |
| local_draft_ms | 18,092 ms | **11,482 ms** | 1,813 ms |
| Rounds | 73 | **49** | 37 |
| Acceptance length | 2.51 / 8 | **4.29 / 8** | 6.03 / 8 |

The working-cache rebuild fixes alone reduced draft time by 37% and improved
acceptance rate significantly. The remaining gap to the 05-30 best is due to
the working cache having ~1024 tokens of past KV per round (retrieval chunks),
making each draft attention pass materially more expensive than the 05-30
sparse replay cache which only maintained a small recent-token DynamicCache.
That trade-off is inherent to the full-draft-kv + retrieval working-cache design
and matches the sparse-kv reference architecture's cost model.

## 2026-06-01 00:40 — Async Pipeline + DynamicCache Investigation (Reverted)

### Attempts and findings

**Attempt 1: Async pipeline with parallel executors.**
Failed due to GPU contention - both executors mapped to same device, causing serialization instead of parallelism. Reverted to single-executor design.

**Attempt 2: Fixed-size working cache.**
Trimming verified-token tail broke causal attention, dropping acceptance to 1.44/8. Requires sparse attention kernel (unavailable with standard transformers).

**Attempt 3: Replace KVCache with DynamicCache.**
No performance benefit - both use SDPA with same attention path. Required cache truncation that reproduced Attempt 2's context loss.

### Root cause of remaining gap

Current architecture uses ~1024-token working cache + growing verified tokens, making each draft forward expensive (~200-250ms). The 05-30 best used sparse replay cache with only accepted tokens, keeping working cache tiny.

### Potential improvements

1. **Flash attention/sliding window**: Reduce attention cost in working cache
2. **Batched draft forward**: Amortize attention over multiple tokens
3. **Sparse attention**: Enable smaller working cache without context loss
4. **Async pipeline**: Only beneficial if verify RTT >> draft time

### Current state

- Pipeline candidates disabled (overhead exceeds benefit)
- Working cache uses KVCache with in-place operations
- Best throughput: **12.84 tok/s**

## 2026-06-01 00:05 — SpecExtend KV Semantics Cleanup, Not Yet Benchmarked

- Tightened the draft-side cache path to avoid fallback behavior that was not
  faithful to SpecExtend. Removed the inactive sparse replay cache code
  (`_sparse_cache`, `ensure_sparse_cache`, and
  `generate_token_ids_cached_sparse`) so the draft backend must use the explicit
  full draft KV cache plus retrieval-selected working KV cache.
- Fixed retrieval chunk growth semantics to match SpecExtend more closely:
  when appending verified tokens creates a new chunk, that new chunk is appended
  to the selected working chunk set. Existing selected tail chunks are still
  refreshed as their end position grows.
- Fixed draft seeding after working-cache rebuild. The first draft-token logits
  now come from the logits produced while forwarding the newly appended verified
  prefix/correction token into the draft model, matching the original
  SpecExtend flow, instead of recomputing from the last selected working-cache
  token.
- Removed the approximate `DRAFT_TREE_MODE=branching` implementation because it
  recomputed paths from token ids rather than drafting through the explicit
  working KV cache. The backend now raises if non-linear tree mode is requested
  until branching is implemented on top of the strict KV path.
- Validation so far is code-level only: `py_compile` and
  `tests/test_specextend_core.py` pass. No performance run has been executed for
  this cleanup yet.

## 2026-06-05 — Chunk-Level Working Cache Selection

### Motivation
`SpecExtendDraftKVCache` previously accepted a flat list of token indices
(`retrieval_token_indices`) from the edge client to rebuild the working KV cache.
This was different from the original sparse-kv reference, which manages chunks
internally and only accepts chunk IDs from the caller. Moving to chunk-level
selection makes the draft cache manager responsible for chunk bookkeeping,
keeps the chunk→token expansion logic in one place, and aligns the draft
backend with the sparse-kv architecture.

### Changes made

- **`SpecExtendDraftKVCache`**: added internal chunk bookkeeping fields
  (`chunk_size`, `chunks`, `selected_chunks`).

- **Replaced `select_working_tokens(indices)` with `select_chunks(chunk_ids)`**.
  The method now accepts a list of chunk IDs, looks up the corresponding
  `RetrievalChunk` objects, and expands them into `working_token_indices` via
  `_update_working_indices_from_chunks`. If `chunk_ids` is `None`, all chunks
  are selected (full cache).

- **Added `_update_chunks_from_full()`**: rebuilds the chunk list from
  `full_token_ids` after every `append_prefix`, returning `True` when new
  chunks were created.

- **Added `_refresh_selected_tail()`**: keeps the last selected chunk in sync
  with the latest full-cache tail, matching sparse-kv's behavior.

- **`append_prefix` now maintains chunk bookkeeping automatically**.
  On first call it selects all chunks; on subsequent calls it appends newly
  created tail chunks to the selected set and refreshes the tail chunk end
  position before rebuilding the working cache.

- **Updated `build_draft_tree` / `build_draft_candidate` signatures** in
  `QwenSpecExtendDraftBackend` and the `SpecExtendDraftBackend` Protocol:
  parameter renamed from `retrieval_token_indices` to `retrieval_chunk_ids`.

- **Updated `specextend_edge_client.py`**: the client now passes
  `self.retrieval.selected_chunk_ids()` instead of
  `self.retrieval.selected_token_indices()` to the draft backend.

- **`position_ids` semantics are unchanged**: `context()` still returns
  `working_token_indices` (original global positions) as `position_ids`,
  preserving the correct RoPE encoding that distinguishes this backend from
  sparse-kv's local-position approach.

- **Also fixed `quick_test.py` warmup call** to use the new parameter name.

### Validation so far

Code-level checks pass (`py_compile`, existing tests). No performance benchmark
has been run yet for this refactor.

## 2026-06-05 23:42 — Remove Branching Tree Decoding, Keep Linear-Only Draft

### Background
The draft backend (`QwenSpecExtendDraftBackend`) already only supported linear
speculative decoding in practice: `_grow_tree` checked `DRAFT_TREE_MODE` and
raised `NotImplementedError` for any non-linear mode. However, the codebase
still carried a large amount of branching-tree scaffolding, helper methods,
and experiment scripts that were never exercised and cluttered the linear path.

### Changes made

- **`scripts/experiments/proactive_tree.py` — deleted.**
  Pure tree-branch-planning logic; never imported or run by the linear pipeline.

- **`scripts/core/qwen_specextend_backend.py` — cleaned up draft backend.**
  - `build_draft_tree` now calls `_grow_linear_tree` directly; removed the
    `_grow_tree` wrapper and the `DRAFT_TREE_MODE` environment variable check.
  - Removed unused `_find_parent_index` static method.

- **`scripts/core/qwen_specextend_backend.py` — cleaned up target verifier.**
  - `verify_tree` now unconditionally calls `_verify_linear_path`; removed the
    `_is_linear_tree` branch and the full branching-tree verification fallback.
  - Removed branching-tree helper methods:
    - `_is_linear_tree`
    - `_paths_from_tree`
    - `_indices_for_path`
    - `_verify_path_against_tokens`

- **`tests/test_specextend_core.py` — removed branching-tree tests.**
  - Deleted `test_target_tree_path_helpers` (tested `_paths_from_tree` and
    `_indices_for_path`, both now removed).
  - Kept `test_tree_attention_mask_contains_ancestors`; linear draft still
    produces a degenerate tree (`parent_indices = [-1, 0, 1, …]`) and the
    `_tree_attention_mask` helper remains valid.

- **Cleaned orphaned `__pycache__/*.pyc` files** for the deleted experiment
  scripts (`proactive_tree`, `tree_async_client`).

### What is intentionally preserved

- `DraftTree`, `DraftTreeResult`, `SpecExtendTreeRequest/Response`, and the
  `_tree_attention_mask` helper are kept because the linear path is represented
  as a degenerate 1-branch tree in the SpecExtend wire protocol; removing these
  would require redesigning the protocol itself.
- `modeling_qwen3_kv.py` still supports a `tree_mask` field because the linear
  draft runs through the same batch-attention path that uses tree masks; this is
  not branching-specific.

## 2026-06-05 23:57 — Tiered KV Store Abstraction (Research Infrastructure)

### Motivation
Research trade-offs between sparse KV selection benefits and cross-tier loading latency overhead. Current implementation keeps all KV in GPU HBM, unable to quantify CPU/SSD loading impact on end-to-end latency.

### Changes made

- **Added `scripts/core/tiered_kv_store.py`**
  - `KVTier` enum: `GPU` / `CPU` / `SSD`
  - `KVLoadMetrics` dataclass: Precise latency breakdown per working cache rebuild (`gpu/cpu/ssd_chunks`, `gpu/cpu/ssd_load_ms`, `total_load_ms`, `.as_dict()`)
  - `TieredKVStore`: Manages tier labels per chunk; `evict_to_cpu` moves data to CPU pinned memory, `evict_to_ssd` serializes to SSD; `load_chunks` handles tiered data movement with timing (incl. `cuda.synchronize()`)

- **Modified `scripts/core/qwen_specextend_backend.py`**
  - `SpecExtendDraftKVCache.__init__` constructs `TieredKVStore`, added `last_load_metrics: Optional[KVLoadMetrics]`
  - `_rebuild_working_cache` dual-path: pure GPU chunks use original fast path with timing; mixed tiers route through `tiered_store.load_chunks()` with per-chunk timing
  - Added public APIs: `evict_to_cpu(chunk_id)` / `evict_to_ssd(chunk_id)` / `chunk_tier_summary()`
  - `reset()` calls `tiered_store.reset()` to clean SSD files and tier state
  - `DraftTreeResult` from `build_draft_tree` now includes `kv_load_metrics`

- **Modified `scripts/core/specextend_backend.py`**
  - `DraftTreeResult` added `kv_load_metrics: Optional[Any] = None` (backward compatible)

- **Bug fix**: `verify_tree` undefined `best_accept_len` changed to `accepted_len`

### Design principles
No inference engine changes, no paged attention. All KV data structures and attention computation unchanged; tiered store inserts controllable latency measurement point only in `_rebuild_working_cache` data movement layer. No performance impact (pure GPU uses original fast path).

## 2026-06-07 — Tail Chunk Force-Include Fix in `select_chunks`

### 问题

Cloud 每隔 `retrieve_every_n_steps` 步才触发一次 attention-based chunk 更新。
在两次更新之间，edge 每轮都接受新 token 并通过 `append_prefix` 将其写入
`full_draft_kv`，新 token 会逐渐填满当前 tail chunk 并可能创建新 chunk。
但当下一轮 `build_draft_tree` 传入上一次 cloud 返回的旧 `retrieval_chunk_ids`
时，`select_chunks(old_ids)` 仅保留 cloud 明确选中的 chunk，新生成的 tail
chunks 会被丢弃。

- **后果**：draft model 做 attention 时看不到两次 cloud 更新之间新增的 token
  的 KV；间隔越长、每轮接受 token 越多，累积的"不可见 tail"就越长。
- **极端情况**：默认 `retrieve_every_n_steps=16`，平均 acceptance length ≈ 4，
  两次更新间最多积累 ~64 个新 token，约 2 个 chunk 始终缺失于 working cache。

### 修复方案

在 `SpecExtendDraftKVCache` 中新增字段 `_retrieval_base_seq_len: int`，记录
**上次 cloud retrieval 生效时** `full_token_ids` 的长度。每次 `select_chunks`
被调用时：

- `chunk_ids is None`（全选初始化）：`_retrieval_base_seq_len = total_seq_len`
- `chunk_ids` 为外部 cloud 选择列表：在 `wanted` 集合之外，强制包含所有
  `chunk.start >= _retrieval_base_seq_len` 的 tail chunks，然后更新
  `_retrieval_base_seq_len = total_seq_len`。

这样，从上次 cloud 更新后产生的每一轮新 token 所在的 chunk，均会被加入
working cache，而不管 cloud 有没有在旧的 top-k 里选到它们。

### 受影响文件

- **`scripts/core/qwen_specextend_backend.py`**
  - `SpecExtendDraftKVCache.__init__`：新增 `self._retrieval_base_seq_len = 0`
  - `reset()`：新增 `self._retrieval_base_seq_len = 0`
  - `select_chunks()`：完整重写，加入 tail chunk force-include 逻辑和
    `_retrieval_base_seq_len` 推进

### 不变的行为

- `chunk_ids is None` 时（第一次 prefix、或无 retrieval 模式）行为不变：
  全选所有 chunks。
- `_refresh_selected_tail()` 在 `append_prefix` 中仍然负责更新最后一个 chunk
  的 `end` 边界，两者互补：`select_chunks` 保证 tail chunk 被选中，
  `_refresh_selected_tail` 保证它的边界是最新的。
- `reset()` 已清零 `_retrieval_base_seq_len`，cache 不连续时不会残留旧基线。

## 2026-06-07 — Complete Naming Cleanup: Tree → Linear

### Motivation
The draft generation and target verification had already been linear-only for
several iterations, but the codebase still used tree-flavored names everywhere
(`DraftTree`, `build_draft_tree`, `verify_tree`, `SpecExtendTreeRequest`, etc.).
This caused ongoing confusion and made the wire protocol carry unnecessary
tree-specific fields (`tree_position_ids`, `parent_indices`, `tree_attention_mask`).

### Changes made

- **`scripts/core/protocol.py`**
  - `SpecExtendTreeRequest` → `SpecExtendRequest`
  - `SpecExtendTreeResponse` → `SpecExtendResponse`
  - `accepted_tree_indices` → `accepted_indices`

- **`scripts/core/specextend_backend.py`**
  - `DraftTree` → `DraftSequence` (dropped `parent_indices` and `attention_mask`)
  - `DraftTreeResult` → `DraftResult`
  - `build_draft_tree` → `build_draft`
  - `verify_tree` → `verify`

- **`scripts/core/qwen_specextend_backend.py`**
  - `build_draft_tree` → `build_draft`
  - `build_draft_candidate` → `build_prefetch_draft`
  - `_grow_linear_tree` → `_generate_draft_sequence`
  - `verify_tree` → `verify`
  - Removed unused `_tree_attention_mask` static method
  - Updated `runtime_debug_info` to show `draft_mode: "linear"` instead of
    `specextend_tree_verify: True`

- **`scripts/client/specextend_edge_client.py`**
  - Updated all imports, method calls, and pipeline helpers to use new names.
  - `cloud_verify_tree` → `cloud_verify`
  - `DraftTree`/`DraftTreeResult` → `DraftSequence`/`DraftResult`

- **`scripts/client/http_cloud_client.py`**
  - `verify_specextend_tree` → `verify_specextend`

- **`scripts/experiments/network_conditions.py`**
  - `verify_specextend_tree` → `verify_specextend`

- **`scripts/server/verify_server.py`**
  - `verify_specextend_tree` → `verify_specextend`
  - `accepted_tree_indices` → `accepted_indices` in Pydantic response model

- **`scripts/experiments/quick_test.py`**
  - Fixed `warmup()` to construct `SpecExtendRequest` with only `draft_ids`.
  - Updated `cloud_verify` wiring.

## 2026-06-07 — Bug Fix: "Inplace update to inference tensor" Crash

### Problem
After the first successful prompt, subsequent prompts crashed with:
`Inplace update to inference tensor outside InferenceMode is not allowed`.

### Root cause
`SpecExtendDraftKVCache.reset()` calls `_allocate_kv_cache()` to create fresh
KVCache buffers. When `reset()` is triggered from inside `@torch.inference_mode()`
(via `append_prefix` → common-prefix mismatch), all tensors allocated in that
context — including GPU KV data buffers and CPU `current_length` scalars — become
**inference tensors**. Any later in-place mutation (`fill_`, `add_`, `copy_`)
outside inference mode then raises.

### Fix
- `_allocate_kv_cache()` now allocates every tensor inside
  `with torch.inference_mode(False):`
- `KVCache.copy()` and `KVCache.cat()` wrap `current_length` writes with
  `torch.inference_mode(False)`
- `_rebuild_working_cache()` and `_restore_working_lengths()` wrap all
  `current_length.fill_()` calls with `torch.inference_mode(False)`

## 2026-06-07 — Bug Fix: AWQ Model Loading Memory & Speed

### Problem
`QwenModelRuntime` loaded the AWQ verify model with `.to(self.device)`, which
first loads weights on CPU (de-quantized) then copies to GPU. Result: 16 GB VRAM
and much slower inference.

### Fix
Detect AWQ models via `quantization_config.quant_method == "awq"` and load with
`device_map="auto"` instead of `torch_dtype` + `.to()`. Memory dropped from
~16 GB to ~10 GB.

Also changed `start_verify.sh` default from `eager` to `flash_attention_2`.
(Note: Qwen3 in the current Transformers version silently falls back to
`Qwen3Attention` for both `flash_attention_2` and `sdpa`; `eager` remains the
effective implementation until upstream support is added.)

## 2026-06-07 — Retrieval-Epoch KV Suffix Fix

Replaced the fixed recent-token fallback with retrieval-epoch semantics for
draft working KV selection:

- `SpecExtendEdgeClient` now marks when the cloud returns a fresh KV selection
  (`target_attn_scores` or `selected_chunk_ids`).
- `SpecExtendDraftKVCache.select_chunks()` advances `_retrieval_base_seq_len`
  only on those fresh selection updates, not every draft round.
- Working KV is now:
  `cloud-selected sparse chunks ∪ chunks overlapping tokens generated since the last cloud selection`.
- `build_draft()` appends the verified prefix before applying chunk selection so
  newly verified tokens are present in the backend chunk table.
- Added a unit test covering the invariant that chunks generated between two
  cloud selection updates remain in the working KV.

Validation: `py_compile` passes and `tests/test_specextend_core.py` reports
5 tests OK.

## 2026-06-07 22:13
Summary
Bug fix (the crash)
Root cause: evict_to_cpu() snapshots the tail chunk when it has, say, 17 tokens. The tail chunk is still growing (it's the last chunk, _refresh_selected_tail updates it). When load_chunk_to_gpu is called later, it recomputes _chunk_slice(chunk_id, current_total_seq_len) which now returns 19 tokens, then tries to copy into a [..., 19, ...] target slice from a CPU tensor of shape [..., 17, ...] → size mismatch at dim 2.

Fix (across tiered_kv_store.py and qwen_specextend_backend.py):

load_chunk_to_gpu now uses cpu_k.shape[2] (the stored size) as length for CPU/SSD tiers, and returns it as a 4th value actual_len.
load_chunks accumulates target_pos from actual_len and stores it in metrics.tokens_written.
_rebuild_working_cache uses metrics.tokens_written (not n) to set current_length, and clips working_token_indices to match if they diverge.

## 2026-06-08 — Custom Qwen Cache, Attention Retrieval, and Tree Verify Fixes

### Problems

Real benchmark runs showed the current SpecExtend path was still much slower
than direct generation:

- direct endpoint: about 17.5-17.9s for 256 output tokens
- branching SpecExtend before this fix: 126.1s for 256 output tokens
- linear SpecExtend before this fix: 93.8s for 256 output tokens

The root causes were separate but compounding:

1. The local custom `Qwen3Model` did not create a `DynamicCache` when
   `use_cache=True` and `past_key_values=None`, so target-side cached scoring
   and retrieval-attention paths could leave `runtime._cache` as `None`.
2. The verify server default requested `flash_attention_2`, but the local Qwen3
   implementation only provides eager and SDPA layers. The instantiated layer
   and causal-mask path could disagree, causing incorrect tree verification.
3. SDPA mask elision could bypass the custom tree mask.
4. Tree mask application used incorrect broadcast/indexing for 4D masks.
5. Branching target verification had fallen back to sequential token generation
   over paths instead of a single batched tree forward.
6. Branching draft generation recomputed full paths one by one. It also lacked
   `torch.inference_mode()`, so the batched tree attempt retained autograd graph
   state and could OOM GPU 1.

### Fixes

- `modeling_qwen3_kv.py`
  - Initialize `DynamicCache()` inside custom `Qwen3Model.forward()` on first
    cached forward.
  - Preserve `cache_position` and `position_embeddings` when SDPA falls back to
    eager attention for multi-token attention output.
  - Prevent SDPA causal-mask elision when a tree mask is active.
  - Apply tree masks with explicit 4D broadcasting and support partial frontier
    rows during batched tree expansion.

- `qwen_specextend_backend.py`
  - Normalize unsupported custom-model `flash_attention_2` requests to `sdpa`
    before model construction; `/health` now reports the effective attention
    implementation.
  - Load custom Qwen config before construction and set `_attn_implementation`
    before layers are instantiated.
  - Replace path-by-path target tree verification with a batched tree forward
    using the draft tree mask.
  - Replace naive branching draft generation with layer-wise batched expansion
    inspired by `~/code/long-ecsd/specextend/shared/opt_tree.py`.
  - Wrap branching tree draft generation in `@torch.inference_mode()`.

- Runtime defaults
  - `quick_benchmark_config.json`: changed default tree depth from 8 to 4 for
    the current 2x3090/Qwen3-1.7B setup.
  - `quick.sh`: changed default `DRAFT_MAX_LEN` from 32768 to 4096 to avoid
    excessive draft KV preallocation in the quick benchmark. Larger contexts can
    still override this with `DRAFT_MAX_LEN`.

### Validation

- `py_compile` passed for the edited core/client/benchmark files.
- Direct quick result: 256 tokens, 17,878 ms, 14.32 tok/s.
- Fixed SpecExtend result: 258 tokens, 28,908 ms, 8.92 tok/s.
- Fixed SpecExtend breakdown: `local_draft_ms=15,261`,
  `server_model_ms=12,146`, 77 rounds, acceptance length 2.35.

Conclusion: the minutes-scale regression is fixed, and target retrieval now
returns real chunk selections, but this short 256-token setup still does not
beat direct generation. Remaining bottlenecks are draft tree cost and acceptance
rate.
