#!/bin/bash
# Context Length Experiment for TableBench
# Usage: ./test_context_length_experiment.sh [--first_n N] [--target_lengths "8000,16000,..."]

# Default Paths
LLM_PATH="../../models/Qwen3-4B-Instruct-2507"
LLM_NAME="qwen3-4b"
EMBEDDING_MODEL_PATH="../../models/bge-m3"
ROUTER_MODEL_PATH="../../HybridRAG/H-STAR/router/wikitq"
CHECK_MODEL_PATH="../../HybridRAG/H-STAR/check/wikitq"
TABLEBENCH_JSON_PATH="../datasets/TableBench/tablebench_math_long_98.json"
API_BASE="http://localhost:8000/v1"
API_KEY="api-key-qwen3"

# Experiment parameters
TARGET_LENGTHS="8000,16000,32000,64000,128000"
FIRST_N=50  # Default to 50 samples

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --llm_path) LLM_PATH="$2"; shift ;;
        --llm_name) LLM_NAME="$2"; shift ;;
        --embedding_model|--embedding) EMBEDDING_MODEL_PATH="$2"; shift ;;
        --router_model|--router) ROUTER_MODEL_PATH="$2"; shift ;;
        --check_model|--check) CHECK_MODEL_PATH="$2"; shift ;;
        --tablebench_json_path|--input) TABLEBENCH_JSON_PATH="$2"; shift ;;
        --target_lengths) TARGET_LENGTHS="$2"; shift ;;
        --first_n) FIRST_N="$2"; shift ;;
        --api_base) API_BASE="$2"; shift ;;
        --api_key) API_KEY="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --llm_path PATH            Path to LLM model"
            echo "  --llm_name NAME            Model name for API"
            echo "  --embedding_model PATH     Path to embedding model"
            echo "  --router_model PATH        Path to router model"
            echo "  --check_model PATH         Path to check model"
            echo "  --tablebench_json_path     Path to TableBench JSON file"
            echo "  --target_lengths LENGTHS   Comma-separated target token lengths"
            echo "  --first_n N                Only process first N samples"
            echo "  --api_base URL             vLLM API base URL"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo "Context Length Experiment for TableBench"
echo "=========================================="
echo "LLM Path:        ${LLM_PATH}"
echo "LLM Name:        ${LLM_NAME}"
echo "Router Model:    ${ROUTER_MODEL_PATH}"
echo "Check Model:     ${CHECK_MODEL_PATH}"
echo "Input File:      ${TABLEBENCH_JSON_PATH}"
echo "Target Lengths:  ${TARGET_LENGTHS}"
echo "First N:         ${FIRST_N}"
echo "API Base:        ${API_BASE}"
echo "=========================================="

CMD="python run_context_length_experiment.py \
  --use_api \
  --api_base \"${API_BASE}\" \
  --api_key \"${API_KEY}\" \
  --llm_path \"${LLM_PATH}\" \
  --llm_name \"${LLM_NAME}\" \
  --embedding_model_path \"${EMBEDDING_MODEL_PATH}\" \
  --router_model_path \"${ROUTER_MODEL_PATH}\" \
  --check_model_path \"${CHECK_MODEL_PATH}\" \
  --tablebench_jsonl_path \"${TABLEBENCH_JSON_PATH}\" \
  --target_lengths \"${TARGET_LENGTHS}\" \
  --first_n ${FIRST_N}"

echo ""
echo "Running: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "=========================================="
echo "Results saved to datasets/schedule_test/context_length_exp/"
echo ""
