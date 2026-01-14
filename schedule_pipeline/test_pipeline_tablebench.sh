#!/bin/bash
# Test script for TableBench pipeline
# Usage: ./test_pipeline_tablebench.sh [--router_model PATH] [--check_model PATH] [--embedding_model PATH]

# ============================================
# Default Model Paths - Override via CLI args
# ============================================
LLM_PATH="../models/Qwen3-30B-A3B-Instruct-2507-FP8"
EMBEDDING_MODEL_PATH="/data/workspace/yanmy/models/bge-m3"
ROUTER_MODEL_PATH="/data/workspace/yanmy/HybridRAG/H-STAR/router/bge-m3-finetuned/"
CHECK_MODEL_PATH="/data/workspace/yanmy/HybridRAG/H-STAR/check/output/bge-reranker-v2-m3-finetuned/"
TMP_SAVE_PATH="tmp/tablebench_test"
TABLEBENCH_JSONL_PATH="../datasets/TableBench/TableBench.jsonl"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --llm_path|--llm)
      LLM_PATH="$2"
      shift 2
      ;;
    --embedding_model|--embedding)
      EMBEDDING_MODEL_PATH="$2"
      shift 2
      ;;
    --router_model|--router)
      ROUTER_MODEL_PATH="$2"
      shift 2
      ;;
    --check_model|--check)
      CHECK_MODEL_PATH="$2"
      shift 2
      ;;
    --output|--tmp)
      TMP_SAVE_PATH="$2"
      shift 2
      ;;
    --input|--jsonl)
      TABLEBENCH_JSONL_PATH="$2"
      shift 2
      ;;
    --first_n)
      FIRST_N="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --llm_path, --llm PATH         Path to LLM model"
      echo "  --embedding_model, --embedding PATH  Path to embedding model (bge-m3)"
      echo "  --router_model, --router PATH  Path to router model"
      echo "  --check_model, --check PATH    Path to check/reranker model"
      echo "  --output, --tmp PATH           Output directory"
      echo "  --input, --jsonl PATH          Path to TableBench JSONL file"
      echo "  --first_n N                    Only process first N samples"
      echo "  -h, --help                     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Set Python path
export PYTHONPATH="$(pwd)/..:$(pwd):$PYTHONPATH"

# Create output directory
mkdir -p ${TMP_SAVE_PATH}

echo "=============================================="
echo "TableBench Pipeline Test"
echo "=============================================="
echo "LLM Path:        ${LLM_PATH}"
echo "Embedding Model: ${EMBEDDING_MODEL_PATH}"
echo "Router Model:    ${ROUTER_MODEL_PATH}"
echo "Check Model:     ${CHECK_MODEL_PATH}"
echo "Input JSONL:     ${TABLEBENCH_JSONL_PATH}"
echo "Output Path:     ${TMP_SAVE_PATH}"
if [ ! -z "${FIRST_N}" ]; then
  echo "First N:         ${FIRST_N}"
fi
echo "=============================================="

# Build command
CMD="python run_full_pipeline_tablebench.py \
  --use_api \
  --llm_path \"${LLM_PATH}\" \
  --embedding_model_path \"${EMBEDDING_MODEL_PATH}\" \
  --router_model_path \"${ROUTER_MODEL_PATH}\" \
  --check_model_path \"${CHECK_MODEL_PATH}\" \
  --dataset_name tablebench \
  --tmp_save_path \"${TMP_SAVE_PATH}\" \
  --tablebench_jsonl_path \"${TABLEBENCH_JSONL_PATH}\" \
  --n_parallel 32"

# Add first_n if specified
if [ ! -z "${FIRST_N}" ]; then
  CMD="${CMD} --first_n ${FIRST_N}"
fi

# Run TableBench pipeline
echo "Running: ${CMD}"
eval ${CMD} 2>&1 | tee "${TMP_SAVE_PATH}/test_run.log"

echo ""
echo "=============================================="
echo "Pipeline test complete."
echo "Check ${TMP_SAVE_PATH}/test_run.log for results."
echo "=============================================="
