#!/bin/bash

# Quick Test Configuration Script
# Edit the variables below to customize your test run

# ============================================
# MAX TOKENS TO GENERATE
# LongWriter prompts are intended for long-form generation.
# ============================================
MAX_TOKENS="${MAX_TOKENS:-512}"

# ============================================
# PROMPT COUNT PER TYPE
# Number of prompts to load for each prompt type
# Default: 1
# ============================================
PROMPT_COUNT="${PROMPT_COUNT:-1}"

# ============================================
# PROMPT TYPES
# Options: govreport
# ============================================
PROMPT_TYPES="${PROMPT_TYPES:-pg19}"

# ============================================
# DRAFT MODEL SETTINGS
# Defaults are set after SCRIPT_DIR is defined. This folder is Mac/MLX-only.
# ============================================

# ============================================
# SPECEXTEND TREE SETTINGS
# ============================================
NODES="${NODES:-8}"
MAX_DEPTH="${MAX_DEPTH:-4}"
THRESHOLD="${THRESHOLD:-0.7}"
RETRIEVAL_CHUNK_SIZE="${RETRIEVAL_CHUNK_SIZE:-32}"
RETRIEVE_TOP_K="${RETRIEVE_TOP_K:-32}"
RETRIEVE_EVERY_N_STEPS="${RETRIEVE_EVERY_N_STEPS:-8}"

# ============================================
# VERIFY SERVER SETTINGS
# ============================================
VERIFY_SERVER_URL="${VERIFY_SERVER_URL:-http://127.0.0.1:6007}"

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
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$SCRIPT_DIR/models/Qwen3-0.6B-4bit-AWQ}"
DRAFT_MAX_LEN="${DRAFT_MAX_LEN:-6000}"

# Export environment variables
export DRAFT_MODEL_PATH
export DRAFT_MAX_LEN
export VERIFY_SERVER_URL

# Build command
CMD="$PYTHON scripts/experiments/quick_test.py \
    --max-tokens ${MAX_TOKENS} \
    --prompt-count ${PROMPT_COUNT} \
    --prompt-types ${PROMPT_TYPES} \
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
echo "Nodes: ${NODES}"
echo "Max Depth: ${MAX_DEPTH}"
echo "Retrieval Chunk Size: ${RETRIEVAL_CHUNK_SIZE}"
echo "Draft Model: ${DRAFT_MODEL_PATH}"
echo "Draft Runtime: MLX on Apple Silicon"
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
