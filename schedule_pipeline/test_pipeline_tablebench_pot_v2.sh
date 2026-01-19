#!/bin/bash
# Enhanced test script v2 with improved parsing
# Usage: ./test_pipeline_tablebench_pot_v2.sh [--first_n N]

# Default Paths
LLM_PATH="../../models/Qwen3-4B-Instruct-2507"
LLM_NAME="qwen3-4b"
TABLEBENCH_JSONL_PATH="../datasets/TableBench/TableBench_PoT.jsonl"
API_BASE="http://localhost:8000/v1"
API_KEY="api-key-qwen3"
FIRST_N=10  # Default to 10 samples for testing

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --llm_path) LLM_PATH="$2"; shift ;;
        --llm_name) LLM_NAME="$2"; shift ;;
        --tablebench_jsonl_path) TABLEBENCH_JSONL_PATH="$2"; shift ;;
        --first_n) FIRST_N="$2"; shift ;;
        --api_base) API_BASE="$2"; shift ;;
        --api_key) API_KEY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo "Enhanced TableBench PoT Pipeline Test V2"
echo "=========================================="
echo "Testing first ${FIRST_N} samples"
echo "Improved parsing logic"
echo "Timestamped output (no overwrite)"
echo ""

CMD="python run_pipeline_tablebench_pot_enhanced_v2.py \
  --use_api \
  --api_base \"${API_BASE}\" \
  --api_key \"${API_KEY}\" \
  --llm_path \"${LLM_PATH}\" \
  --llm_name \"${LLM_NAME}\" \
  --dataset_name tablebench \
  --tablebench_jsonl_path \"${TABLEBENCH_JSONL_PATH}\" \
  --code_sample_num 3 \
  --n_parallel 32 \
  --first_n ${FIRST_N}"

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


