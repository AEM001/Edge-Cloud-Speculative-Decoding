#!/bin/bash

# Quick Test Configuration Script
# Edit the variables below to customize your test run

# ============================================
# MAX TOKENS TO GENERATE
# LongWriter prompts are intended for long-form generation.
# ============================================
MAX_TOKENS="${MAX_TOKENS:-256}"

# ============================================
# PROMPT COUNT PER TYPE
# Number of prompts to load for each prompt type
# Default: 1
# ============================================
PROMPT_COUNT="${PROMPT_COUNT:-1}"

# ============================================
# PROMPT INPUT TOKENS
# Set to 2048 for a fixed 2k-token input prompt.
# ============================================
PROMPT_INPUT_TOKENS="${PROMPT_INPUT_TOKENS:-2048}"

# ============================================
# PROMPT TYPES
# Options: gsm8k, humaneval, longwriter,
#          longwriter_single_turn:input_4k, longwriter_single_turn:input_6k,
#          longwriter_single_turn:input_8k, longwriter_single_turn:input_10k,
#          longbench_v2:short, longbench_v2:medium, longbench_v2:long,
#          longbench_v2:train, pg19, prompts_2048
# You can specify multiple (space-separated)
# ============================================
PROMPT_TYPES="${PROMPT_TYPES:-pg19}"

# ============================================
# DRAFT MODEL SETTINGS
# Defaults are set after SCRIPT_DIR is defined.
# ============================================

# ============================================
# SPECEXTEND TREE SETTINGS
# ============================================
NODES="${NODES:-32}"
MAX_DEPTH="${MAX_DEPTH:-8}"
THRESHOLD="${THRESHOLD:-0.7}"
RETRIEVAL_CHUNK_SIZE="${RETRIEVAL_CHUNK_SIZE:-32}"
RETRIEVE_TOP_K="${RETRIEVE_TOP_K:-32}"
RETRIEVE_EVERY_N_STEPS="${RETRIEVE_EVERY_N_STEPS:-8}"

# ============================================
# VERIFY SERVER SETTINGS
# ============================================
VERIFY_SERVER_URL="${VERIFY_SERVER_URL:-http://localhost:6007}"

# ============================================
# END OF CONFIGURATION
# ============================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use .venv environment
PYTHON="$SCRIPT_DIR/.venv/bin/python3"

# ============================================
# DRAFT MODEL SETTINGS (set after SCRIPT_DIR is defined)
# ============================================
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$SCRIPT_DIR/models/Qwen3-1.7B}"
DRAFT_GPU_ID="${DRAFT_GPU_ID:-0}"  # GPU 0 for single GPU setup
DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:$DRAFT_GPU_ID}"
DRAFT_DTYPE="${DRAFT_DTYPE:-fp16}"
DRAFT_MAX_LEN="${DRAFT_MAX_LEN:-32768}"
DRAFT_ATTN_IMPLEMENTATION="${DRAFT_ATTN_IMPLEMENTATION:-sdpa}"
DRAFT_RECENT_TOKENS="${DRAFT_RECENT_TOKENS:-128}"

# Export environment variables
export DRAFT_MODEL_PATH
export DRAFT_GPU_ID
export DRAFT_DEVICE
export DRAFT_DTYPE
export DRAFT_MAX_LEN
export DRAFT_ATTN_IMPLEMENTATION
export DRAFT_RECENT_TOKENS
export VERIFY_SERVER_URL

# Build command
CMD="$PYTHON scripts/experiments/quick_test.py \
    --max-tokens ${MAX_TOKENS} \
    --prompt-count ${PROMPT_COUNT} \
    --prompt-types ${PROMPT_TYPES} \
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
echo "Network: good"
echo "Methods: direct, specextend"
echo "Max Tokens: ${MAX_TOKENS}"
echo "Prompt Count (per type): ${PROMPT_COUNT}"
echo "Prompt Types: ${PROMPT_TYPES}"
echo "Prompt Input Tokens: ${PROMPT_INPUT_TOKENS}"
echo "Nodes: ${NODES}"
echo "Max Depth: ${MAX_DEPTH}"
echo "Retrieval Chunk Size: ${RETRIEVAL_CHUNK_SIZE}"
echo "Draft Model: ${DRAFT_MODEL_PATH}"
echo "Draft Device: ${DRAFT_DEVICE}"
echo "Draft Attention: ${DRAFT_ATTN_IMPLEMENTATION}"
echo "Draft Recent Tokens: ${DRAFT_RECENT_TOKENS}"
echo "Verify Server URL: ${VERIFY_SERVER_URL}"
echo "=========================================="
echo ""
echo "Command:"
echo "$CMD"
echo ""
echo "=========================================="
echo ""

# Run the test
eval $CMD
