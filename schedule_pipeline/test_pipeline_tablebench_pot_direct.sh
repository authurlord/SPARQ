#!/bin/bash
# Test script for POT Pipeline (Direct Answer Version)
# Usage: ./test_pipeline_tablebench_pot_direct.sh [--first_n N] [--random_sample]

# Default Paths
# LLM_PATH="../../models/Qwen3-30B-Instruct-2507-FP8"
LLM_PATH="../../models/Qwen3-4B-Instruct-2507"
LLM_NAME="qwen3-4b"
TABLEBENCH_JSONL_PATH="../datasets/TableBench/TableBench.jsonl"
API_BASE="http://localhost:8000/v1"
API_KEY="api-key-qwen3"
FIRST_N=50  # Default to 50 samples
RANDOM_SAMPLE=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --llm_path) LLM_PATH="$2"; shift ;;
        --llm_name) LLM_NAME="$2"; shift ;;
        --tablebench_jsonl_path) TABLEBENCH_JSONL_PATH="$2"; shift ;;
        --first_n) FIRST_N="$2"; shift ;;
        --random|--random_sample) RANDOM_SAMPLE="--random_sample" ;;
        --api_base) API_BASE="$2"; shift ;;
        --api_key) API_KEY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo "POT Pipeline Test (Direct Answer)"
echo "=========================================="
echo "Testing ${FIRST_N} samples"
if [ ! -z "${RANDOM_SAMPLE}" ]; then
    echo "Random sampling enabled"
fi
echo "Python code samples per question: 3"
echo "Max iterations: 3"
echo "Mode: Direct Python Answer with LLM Fallback"
echo "Timestamped output (no overwrite)"
echo ""

CMD="python run_pipeline_tablebench_pot_direct.py \
  --use_api \
  --api_base \"${API_BASE}\" \
  --api_key \"${API_KEY}\" \
  --llm_path \"${LLM_PATH}\" \
  --llm_name \"${LLM_NAME}\" \
  --dataset_name tablebench \
  --tablebench_jsonl_path \"${TABLEBENCH_JSONL_PATH}\" \
  --code_sample_num 3 \
  --max_iterations 3 \
  --temperature 0.7 \
  --top_p 0.8 \
  --n_parallel 32 \
  --llm_concurrency 32 \
  --first_n ${FIRST_N} \
  ${RANDOM_SAMPLE}"

echo "Running: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo ""
echo "Results saved with timestamp to avoid overwriting"
echo "Check the latest directory in datasets/schedule_test/"
echo ""
