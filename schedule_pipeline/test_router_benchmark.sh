#!/bin/bash
# Router Model Benchmark Pipeline
# Tests different router models while keeping Check and LLM constant
# Usage: ./test_router_benchmark.sh [--first_n N] [--router_models_dir DIR]

# Default Paths
LLM_PATH="../../models/Qwen3-4B-Instruct-2507"
LLM_NAME="qwen3-4b"
EMBEDDING_MODEL_PATH="../../models/bge-m3"
ROUTER_MODELS_DIR="../models"
ROUTER_MODEL_PATTERN="*router*"
CHECK_MODEL_PATH="../../HybridRAG/H-STAR/check/wikitq"
TABLEBENCH_JSONL_PATH="../datasets/TableBench/TableBench.jsonl"
API_BASE="http://localhost:8000/v1"
API_KEY="api-key-qwen3"
FIRST_N=50  # Default to 50 samples

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --llm_path) LLM_PATH="$2"; shift ;;
        --llm_name) LLM_NAME="$2"; shift ;;
        --embedding_model|--embedding) EMBEDDING_MODEL_PATH="$2"; shift ;;
        --router_models_dir|--routers) ROUTER_MODELS_DIR="$2"; shift ;;
        --router_model_pattern|--pattern) ROUTER_MODEL_PATTERN="$2"; shift ;;
        --check_model|--check) CHECK_MODEL_PATH="$2"; shift ;;
        --tablebench_jsonl_path|--input) TABLEBENCH_JSONL_PATH="$2"; shift ;;
        --first_n) FIRST_N="$2"; shift ;;
        --api_base) API_BASE="$2"; shift ;;
        --api_key) API_KEY="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Router Model Benchmark - Tests multiple router models"
            echo ""
            echo "Options:"
            echo "  --llm_path PATH             Path to LLM model (constant)"
            echo "  --llm_name NAME             Model name for API"
            echo "  --embedding_model PATH      Path to embedding model"
            echo "  --router_models_dir DIR     Directory containing router models to test"
            echo "  --router_model_pattern PAT  Glob pattern to filter models (default: *router*)"
            echo "  --check_model PATH          Path to check model (constant)"
            echo "  --tablebench_jsonl_path     Path to TableBench JSONL file"
            echo "  --first_n N                 Only process first N samples"
            echo "  --api_base URL              vLLM API base URL"
            echo "  -h, --help                  Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo "Router Model Benchmark Pipeline"
echo "=========================================="
echo "LLM Path:          ${LLM_PATH}"
echo "LLM Name:          ${LLM_NAME}"
echo "Router Models Dir: ${ROUTER_MODELS_DIR}"
echo "Router Pattern:    ${ROUTER_MODEL_PATTERN}"
echo "Check Model:       ${CHECK_MODEL_PATH}"
echo "Input File:        ${TABLEBENCH_JSONL_PATH}"
echo "First N:           ${FIRST_N}"
echo "API Base:          ${API_BASE}"
echo "=========================================="

# List discovered router models
echo ""
echo "Discovering router models in ${ROUTER_MODELS_DIR}..."
find ${ROUTER_MODELS_DIR} -maxdepth 1 -type d -name "${ROUTER_MODEL_PATTERN}" 2>/dev/null | head -20
echo ""

CMD="python run_router_benchmark.py \
  --use_api \
  --api_base \"${API_BASE}\" \
  --api_key \"${API_KEY}\" \
  --llm_path \"${LLM_PATH}\" \
  --llm_name \"${LLM_NAME}\" \
  --embedding_model_path \"${EMBEDDING_MODEL_PATH}\" \
  --router_models_dir \"${ROUTER_MODELS_DIR}\" \
  --router_model_pattern \"${ROUTER_MODEL_PATTERN}\" \
  --check_model_path \"${CHECK_MODEL_PATH}\" \
  --tablebench_jsonl_path \"${TABLEBENCH_JSONL_PATH}\" \
  --first_n ${FIRST_N}"

echo "Running: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
echo "Results saved to datasets/schedule_test/router_benchmark/"
echo ""
