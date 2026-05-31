#!/bin/bash

# Quick Test Configuration Script
# ============================================
# 所有可调参数都集中在下面，直接修改即可
# ============================================

# ---------- 数据集配置（目前固定 pg19） ----------
# 可选文件: pg19_512, pg19_1K, pg19_2K, pg19_4K, pg19_8K, pg19_16K
DATASET_SPLIT="${DATASET_SPLIT:-pg19_512}"

# ---------- 生成配置 ----------
# 每次生成多少 token
MAX_TOKENS="${MAX_TOKENS:-512}"
# 加载多少条 prompt
PROMPT_COUNT="${PROMPT_COUNT:-1}"
# 输入 prompt 截断到多少 token（0 = 不截断）
PROMPT_INPUT_TOKENS="${PROMPT_INPUT_TOKENS:-0}"

# ---------- SpecExtend 树配置 ----------
NODES="${NODES:-32}"
MAX_DEPTH="${MAX_DEPTH:-8}"
THRESHOLD="${THRESHOLD:-0.7}"
RETRIEVAL_CHUNK_SIZE="${RETRIEVAL_CHUNK_SIZE:-32}"
RETRIEVE_TOP_K="${RETRIEVE_TOP_K:-32}"
RETRIEVE_EVERY_N_STEPS="${RETRIEVE_EVERY_N_STEPS:-0}"

# ---------- Draft 模型配置 ----------
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/root/code/models/Qwen3-1.7B}"
DRAFT_GPU_ID="${DRAFT_GPU_ID:-1}"
DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:${DRAFT_GPU_ID}}"
DRAFT_DTYPE="${DRAFT_DTYPE:-fp16}"
DRAFT_MAX_LEN="${DRAFT_MAX_LEN:-32768}"
DRAFT_ATTN_IMPLEMENTATION="${DRAFT_ATTN_IMPLEMENTATION:-sdpa}"
DRAFT_RECENT_TOKENS="${DRAFT_RECENT_TOKENS:-128}"

# ---------- 服务端 & 超时配置 ----------
VERIFY_SERVER_URL="${VERIFY_SERVER_URL:-http://localhost:6007}"
QUICK_TEST_TIMEOUT_SEC="${QUICK_TEST_TIMEOUT_SEC:-600}"

# ---------- 其他环境变量 ----------
SPECEXTEND_ASYNC_PIPELINE="${SPECEXTEND_ASYNC_PIPELINE:-1}"
SPECEXTEND_PIPELINE_OFFSETS="${SPECEXTEND_PIPELINE_OFFSETS:-full,half}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ============================================
# END OF CONFIGURATION
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"

# Export all env variables
export DRAFT_MODEL_PATH
export DRAFT_GPU_ID
export DRAFT_DEVICE
export DRAFT_DTYPE
export DRAFT_MAX_LEN
export DRAFT_ATTN_IMPLEMENTATION
export DRAFT_RECENT_TOKENS
export SPECEXTEND_ASYNC_PIPELINE
export SPECEXTEND_PIPELINE_OFFSETS
export TRANSFORMERS_OFFLINE
export VERIFY_SERVER_URL
export QUICK_TEST_TIMEOUT_SEC

# Build command
CMD="$PYTHON scripts/experiments/quick_test.py \
    --max-tokens ${MAX_TOKENS} \
    --prompt-count ${PROMPT_COUNT} \
    --prompt-types pg19 \
    --dataset-split ${DATASET_SPLIT} \
    --prompt-input-tokens ${PROMPT_INPUT_TOKENS} \
    --nodes ${NODES} \
    --max-depth ${MAX_DEPTH} \
    --threshold ${THRESHOLD} \
    --retrieval-chunk-size ${RETRIEVAL_CHUNK_SIZE} \
    --retrieve-top-k ${RETRIEVE_TOP_K} \
    --retrieve-every-n-steps ${RETRIEVE_EVERY_N_STEPS}"

echo "=========================================="
echo "Running Quick Test with Configuration:"
echo "=========================================="
echo "Dataset: pg19 | File: ${DATASET_SPLIT}"
echo "Max Tokens: ${MAX_TOKENS}"
echo "Prompt Count: ${PROMPT_COUNT}"
echo "Prompt Input Tokens: ${PROMPT_INPUT_TOKENS}"
echo "Nodes: ${NODES}"
echo "Max Depth: ${MAX_DEPTH}"
echo "Threshold: ${THRESHOLD}"
echo "Retrieval Chunk Size: ${RETRIEVAL_CHUNK_SIZE}"
echo "Retrieve Top K: ${RETRIEVE_TOP_K}"
echo "Retrieve Every N Steps: ${RETRIEVE_EVERY_N_STEPS}"
echo "Draft Model: ${DRAFT_MODEL_PATH}"
echo "Draft Device: ${DRAFT_DEVICE}"
echo "Draft Attention: ${DRAFT_ATTN_IMPLEMENTATION}"
echo "Draft Recent Tokens: ${DRAFT_RECENT_TOKENS}"
echo "Async Pipeline: ${SPECEXTEND_ASYNC_PIPELINE}"
echo "Pipeline Offsets: ${SPECEXTEND_PIPELINE_OFFSETS}"
echo "Verify Server URL: ${VERIFY_SERVER_URL}"
echo "Timeout: ${QUICK_TEST_TIMEOUT_SEC}s"
echo "Transformers Offline: ${TRANSFORMERS_OFFLINE}"
echo "=========================================="
echo ""
echo "Command:"
echo "$CMD"
echo ""
echo "=========================================="
echo ""

# Run the test
eval $CMD
