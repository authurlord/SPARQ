#!/bin/bash
# Test script for TableBench PoT pipeline
# Usage: ./test_pipeline_tablebench_pot.sh [--llm_path PATH] [--first_n N]

# Default Paths
LLM_PATH="../../models/Qwen3-4B-Instruct-2507"
TABLEBENCH_JSONL_PATH="../datasets/TableBench/TableBench_PoT.jsonl"
TMP_SAVE_PATH="datasets/schedule_test/tablebench_pot"
API_BASE="http://localhost:8000/v1"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --llm_path) LLM_PATH="$2"; shift ;;
        --tablebench_jsonl_path) TABLEBENCH_JSONL_PATH="$2"; shift ;;
        --tmp_save_path) TMP_SAVE_PATH="$2"; shift ;;
        --first_n) FIRST_N="$2"; shift ;;
        --api_base) API_BASE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "${TMP_SAVE_PATH}"

CMD="python run_pipeline_tablebench_pot.py \
  --use_api \
  --api_base \"${API_BASE}\" \
  --llm_path \"${LLM_PATH}\" \
  --dataset_name tablebench \
  --tmp_save_path \"${TMP_SAVE_PATH}\" \
  --tablebench_jsonl_path \"${TABLEBENCH_JSONL_PATH}\" \
  --code_sample_num 3 \
  --n_parallel 32"

if [ ! -z "${FIRST_N}" ]; then
  CMD="${CMD} --first_n ${FIRST_N}"
fi

echo "Running: ${CMD}"
eval ${CMD} 2>&1 | tee "${TMP_SAVE_PATH}/test_run.log"
