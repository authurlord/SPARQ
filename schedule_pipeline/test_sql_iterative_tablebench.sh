#!/bin/bash
# Test script for SQL Iterative Pipeline
# Usage: ./test_sql_iterative_tablebench.sh [--first_n N] [--random_sample] [--sql_timeout T]

# Default Paths (same as test_pipeline_tablebench_pot_enhanced.sh)
LLM_PATH="../../models/Qwen3-30B-Instruct-2507-FP8"
LLM_NAME="qwen3-4b"
TABLEBENCH_JSONL_PATH="../datasets/TableBench/TableBench.jsonl"
API_BASE="http://localhost:8000/v1"
API_KEY="api-key-qwen3"
FIRST_N=50  # Default to 50 samples
RANDOM_SAMPLE=""
SQL_TIMEOUT=30  # Default SQL timeout in seconds

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --llm_path) LLM_PATH="$2"; shift ;;
        --llm_name) LLM_NAME="$2"; shift ;;
        --tablebench_jsonl_path) TABLEBENCH_JSONL_PATH="$2"; shift ;;
        --first_n) FIRST_N="$2"; shift ;;
        --random|--random_sample) RANDOM_SAMPLE="--random_sample" ;;
        --sql_timeout) SQL_TIMEOUT="$2"; shift ;;
        --api_base) API_BASE="$2"; shift ;;
        --api_key) API_KEY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo "SQL Iterative Pipeline Test"
echo "=========================================="
echo "Testing ${FIRST_N} samples"
if [ ! -z "${RANDOM_SAMPLE}" ]; then
    echo "Random sampling enabled"
fi
echo "SQL samples per question: 3"
echo "Max iterations: 3"
echo "SQL timeout: ${SQL_TIMEOUT}s"
echo "Timestamped output (no overwrite)"
echo ""

CMD="python run_sql_iterative_tablebench.py \
  --use_api \
  --api_base \"${API_BASE}\" \
  --api_key \"${API_KEY}\" \
  --llm_path \"${LLM_PATH}\" \
  --llm_name \"${LLM_NAME}\" \
  --dataset_name tablebench \
  --tablebench_jsonl_path \"${TABLEBENCH_JSONL_PATH}\" \
  --sql_sample_num 1 \
  --max_iterations 1 \
  --sql_timeout ${SQL_TIMEOUT} \
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
