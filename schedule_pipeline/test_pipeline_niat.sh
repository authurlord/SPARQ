#!/bin/bash
# Test script for NIAT pipeline
# Usage: ./test_pipeline_niat.sh [--router_model PATH] [--check_model PATH] [--embedding_model PATH]

# ============================================
# Default Model Paths - Override via CLI args
# ============================================
LLM_PATH="../models/Qwen3-30B-A3B-Instruct-2507-FP8"
EMBEDDING_MODEL_PATH="/data/workspace/yanmy/models/bge-m3"
ROUTER_MODEL_PATH="/data/workspace/yanmy/HybridRAG/H-STAR/router/bge-m3-finetuned/"
CHECK_MODEL_PATH="/data/workspace/yanmy/HybridRAG/H-STAR/check/output/bge-reranker-v2-m3-finetuned/"
TMP_SAVE_PATH="tmp/niat_test_30B"

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
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --llm_path, --llm PATH         Path to LLM model"
      echo "  --embedding_model, --embedding PATH  Path to embedding model (bge-m3)"
      echo "  --router_model, --router PATH  Path to router model"
      echo "  --check_model, --check PATH    Path to check/reranker model"
      echo "  --output, --tmp PATH           Output directory"
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
echo "NIAT Pipeline Test"
echo "=============================================="
echo "LLM Path:        ${LLM_PATH}"
echo "Embedding Model: ${EMBEDDING_MODEL_PATH}"
echo "Router Model:    ${ROUTER_MODEL_PATH}"
echo "Check Model:     ${CHECK_MODEL_PATH}"
echo "Output Path:     ${TMP_SAVE_PATH}"
echo "=============================================="

# Run NIAT pipeline with all model paths
python run_full_pipeline_niat.py \
  --use_api \
  --llm_path "${LLM_PATH}" \
  --embedding_model_path "${EMBEDDING_MODEL_PATH}" \
  --router_model_path "${ROUTER_MODEL_PATH}" \
  --check_model_path "${CHECK_MODEL_PATH}" \
  --dataset_name niat \
  --tmp_save_path "${TMP_SAVE_PATH}" \
  --niat_json_path ../datasets/NIAT/niat_4000_filtered.json \
  --n_parallel 32 \
  2>&1 | tee "${TMP_SAVE_PATH}/test_run.log"

echo ""
echo "=============================================="
echo "Pipeline test complete."
echo "Check ${TMP_SAVE_PATH}/test_run.log for results."
echo "=============================================="
