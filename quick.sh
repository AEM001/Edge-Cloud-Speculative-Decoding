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
# Options: gsm8k, humaneval, longwriter,
#          longwriter_single_turn:input_4k, longwriter_single_turn:input_6k,
#          longwriter_single_turn:input_8k, longwriter_single_turn:input_10k,
#          longbench_v2:short, longbench_v2:medium, longbench_v2:long,
#          longbench_v2:train, prompts_2048
# You can specify multiple (space-separated)
# ============================================
PROMPT_TYPES="${PROMPT_TYPES:-prompts_2048}"

# ============================================
# DRAFT MODEL SETTINGS
# Defaults are set after SCRIPT_DIR is defined.
# ============================================

# ============================================
# TREE BASE DRAFT SETTINGS
# ============================================
K="${K:-15}"  # Base draft length

# ============================================
# TREE ASYNC SETTINGS
# ============================================
TREE_BRANCH_WIDTH="${TREE_BRANCH_WIDTH:-4}"  # Number of tree branches
TREE_BRANCH_DRAFT_LENGTH="${TREE_BRANCH_DRAFT_LENGTH:-12}"  # Pre-draft length for each tree branch

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
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$SCRIPT_DIR/models/Qwen2.5-3B-Instruct-AWQ}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
DRAFT_GPU_ID="${DRAFT_GPU_ID:-0}"  # GPU 0 for single GPU setup
DRAFT_GPU_MEM="${DRAFT_GPU_MEM:-0.30}"
DRAFT_MAX_LEN="${DRAFT_MAX_LEN:-12000}"

# Export environment variables
export DRAFT_MODEL_PATH
export DRAFT_MODEL_NAME
export DRAFT_GPU_ID
export DRAFT_GPU_MEM
export DRAFT_MAX_LEN
export VERIFY_SERVER_URL

# Build command
CMD="$PYTHON scripts/experiments/quick_test.py \
    --max-tokens ${MAX_TOKENS} \
    --prompt-count ${PROMPT_COUNT} \
    --prompt-types ${PROMPT_TYPES} \
    --k ${K} \
    --tree-branch-width ${TREE_BRANCH_WIDTH} \
    --tree-branch-draft-length ${TREE_BRANCH_DRAFT_LENGTH}"

echo "=========================================="
echo "Running Quick Test with Configuration:"
echo "=========================================="
echo "Network: good"
echo "Methods: direct, tree_async"
echo "Max Tokens: ${MAX_TOKENS}"
echo "Prompt Count (per type): ${PROMPT_COUNT}"
echo "Prompt Types: ${PROMPT_TYPES}"
echo "K: ${K}"
echo "Tree Branch Width: ${TREE_BRANCH_WIDTH}"
echo "Tree Branch Draft Length: ${TREE_BRANCH_DRAFT_LENGTH}"
echo "Draft Model: ${DRAFT_MODEL_PATH}"
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
