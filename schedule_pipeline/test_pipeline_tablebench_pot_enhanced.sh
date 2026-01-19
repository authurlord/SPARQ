#!/bin/bash
# Enhanced test script for TableBench PoT pipeline with detailed logging
# Usage: ./test_pipeline_tablebench_pot_enhanced.sh [--first_n N]

# Default Paths
LLM_PATH="../../models/Qwen3-30B-A3B-Instruct-2507-FP8"
LLM_NAME="qwen3-4b"
TABLEBENCH_JSONL_PATH="../datasets/TableBench/TableBench_PoT.jsonl"
TMP_SAVE_PATH="datasets/schedule_test/tablebench_pot_enhanced"
API_BASE="http://localhost:8000/v1"
API_KEY="api-key-qwen3"
FIRST_N=50  # Default to 50 samples

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --llm_path) LLM_PATH="$2"; shift ;;
        --llm_name) LLM_NAME="$2"; shift ;;
        --tablebench_jsonl_path) TABLEBENCH_JSONL_PATH="$2"; shift ;;
        --tmp_save_path) TMP_SAVE_PATH="$2"; shift ;;
        --first_n) FIRST_N="$2"; shift ;;
        --api_base) API_BASE="$2"; shift ;;
        --api_key) API_KEY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "${TMP_SAVE_PATH}"

echo "=========================================="
echo "Enhanced TableBench PoT Pipeline Test"
echo "=========================================="
echo "Testing first ${FIRST_N} samples"
echo "Results will be saved to: ${TMP_SAVE_PATH}"
echo ""

CMD="python run_pipeline_tablebench_pot_enhanced.py \
  --use_api \
  --api_base \"${API_BASE}\" \
  --api_key \"${API_KEY}\" \
  --llm_path \"${LLM_PATH}\" \
  --llm_name \"${LLM_NAME}\" \
  --dataset_name tablebench \
  --tmp_save_path \"${TMP_SAVE_PATH}\" \
  --tablebench_jsonl_path \"${TABLEBENCH_JSONL_PATH}\" \
  --code_sample_num 3 \
  --n_parallel 32 \
  --first_n ${FIRST_N}"

echo "Running: ${CMD}"
echo ""
eval ${CMD} 2>&1 | tee "${TMP_SAVE_PATH}/test_run.log"

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo "Results saved to: ${TMP_SAVE_PATH}"
echo ""
echo "Generated files:"
echo "  - test_run.log: Full execution log"
echo "  - execution_stats.json: Execution statistics"
echo "  - execution_detailed.log: Detailed execution log"
echo "  - execution_errors.log: Error-only log"
echo "  - generated_codes/: All generated Python code"
echo "  - pot_results.csv: Final results"
echo "  - evaluation.json: Evaluation metrics"
echo ""

# Display statistics if available
if [ -f "${TMP_SAVE_PATH}/execution_stats.json" ]; then
    echo "Execution Statistics:"
    cat "${TMP_SAVE_PATH}/execution_stats.json"
    echo ""
fi

echo "To analyze failures, check:"
echo "  cat ${TMP_SAVE_PATH}/execution_errors.log | head -100"
echo "  cat ${TMP_SAVE_PATH}/execution_detailed.log | less"


