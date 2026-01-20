#!/bin/bash
# Training script for WikiTQ pipeline
# Usage: ./train_pipeline_wikitq.sh [--router_model PATH] [--check_model PATH] [--embedding_model PATH]

# ============================================
# Default Model Paths - Override via CLI args
# ============================================
LLM_PATH="../../models/Qwen3-4B-Instruct-2507"
EMBEDDING_MODEL_PATH="../../models/bge-m3"
ROUTER_MODEL_PATH="../../HybridRAG/H-STAR/router/wikitq"
CHECK_MODEL_PATH="../../HybridRAG/H-STAR/check/output/bge-reranker-v2-m3-finetuned"
TMP_SAVE_PATH="tmp/wikitq_train"

# API configuration
API_BASE="http://127.0.0.1:8000/v1"
API_KEY="api-key-qwen3"
MODEL_NAME="/public/Qwen3-4B-Instruct-2507"

# Pipeline parameters
DATASET_NAME="wikitq"
SPLIT="train"
TAU=0.82
CHECK_TAU=0.8
N_PARALLEL=32
SELECT_SAMPLE_NUM=2
SQL_SAMPLE_NUM=3
TEMPERATURE=0.7
TOP_P=0.8
MAX_TOKENS=2048
CONCURRENCY=512

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
    --api_base)
      API_BASE="$2"
      shift 2
      ;;
    --api_key)
      API_KEY="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --tau)
      TAU="$2"
      shift 2
      ;;
    --check_tau)
      CHECK_TAU="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top_p)
      TOP_P="$2"
      shift 2
      ;;
    --max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --n_parallel)
      N_PARALLEL="$2"
      shift 2
      ;;
    --select_sample_num)
      SELECT_SAMPLE_NUM="$2"
      shift 2
      ;;
    --sql_sample_num)
      SQL_SAMPLE_NUM="$2"
      shift 2
      ;;
    --first_n)
      FIRST_N="$2"
      shift 2
      ;;
    --save_intermediate)
      SAVE_INTERMEDIATE="--save_intermediate"
      shift
      ;;
    --use_api)
      USE_API="--use_api"
      shift
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
      echo "  --api_base URL                 API base URL (default: http://127.0.0.1:8000/v1)"
      echo "  --api_key KEY                  API key"
      echo "  --model_name NAME              Model name for API"
      echo "  --split SPLIT                  Dataset split (train/validation/test, default: train)"
      echo "  --tau VALUE                    Router threshold (default: 0.82)"
      echo "  --check_tau VALUE              Check threshold (default: 0.8)"
      echo "  --temperature VALUE            Sampling temperature (default: 0.7)"
      echo "  --top_p VALUE                  Top-p sampling (default: 0.8)"
      echo "  --max_tokens N                 Max tokens (default: 2048)"
      echo "  --concurrency N                API concurrency (default: 512)"
      echo "  --n_parallel N                 Parallel workers (default: 32)"
      echo "  --select_sample_num N          Select sample number (default: 2)"
      echo "  --sql_sample_num N             SQL sample number (default: 3)"
      echo "  --first_n N                    Only process first N samples"
      echo "  --save_intermediate            Save intermediate results"
      echo "  --use_api                      Use API mode"
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
export CUDA_VISIBLE_DEVICES=0

# Allow loading custom dataset scripts (for datasets 4.0+)
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Create output directory
mkdir -p ${TMP_SAVE_PATH}

echo "=============================================="
echo "WikiTQ Training Pipeline"
echo "=============================================="
echo "LLM Path:        ${LLM_PATH}"
echo "Embedding Model: ${EMBEDDING_MODEL_PATH}"
echo "Router Model:    ${ROUTER_MODEL_PATH}"
echo "Check Model:     ${CHECK_MODEL_PATH}"
echo "Dataset:         ${DATASET_NAME}"
echo "Split:           ${SPLIT}"
echo "Output Path:     ${TMP_SAVE_PATH}"
echo "API Base:        ${API_BASE}"
echo "Model Name:      ${MODEL_NAME}"
echo "Tau:             ${TAU}"
echo "Check Tau:       ${CHECK_TAU}"
echo "Temperature:     ${TEMPERATURE}"
echo "Top-p:           ${TOP_P}"
echo "Max Tokens:      ${MAX_TOKENS}"
echo "Concurrency:     ${CONCURRENCY}"
echo "N Parallel:      ${N_PARALLEL}"
if [ ! -z "${FIRST_N}" ]; then
  echo "First N:         ${FIRST_N}"
fi
echo "=============================================="

# Build command
CMD="python run_full_pipeline_wikitq_api.py \
  --api_base \"${API_BASE}\" \
  --api_key \"${API_KEY}\" \
  --model_name \"${MODEL_NAME}\" \
  --concurrency ${CONCURRENCY} \
  --embedding_model_path \"${EMBEDDING_MODEL_PATH}\" \
  --router_model_path \"${ROUTER_MODEL_PATH}\" \
  --check_model_path \"${CHECK_MODEL_PATH}\" \
  --dataset_name ${DATASET_NAME} \
  --split ${SPLIT} \
  --tmp_save_path \"${TMP_SAVE_PATH}\" \
  --tau ${TAU} \
  --check_tau ${CHECK_TAU} \
  --n_parallel ${N_PARALLEL} \
  --select_sample_num ${SELECT_SAMPLE_NUM} \
  --sql_sample_num ${SQL_SAMPLE_NUM} \
  --temperature ${TEMPERATURE} \
  --top_p ${TOP_P} \
  --max_tokens ${MAX_TOKENS}"

# Add optional flags
if [ ! -z "${FIRST_N}" ]; then
  CMD="${CMD} --first_n ${FIRST_N}"
fi

if [ ! -z "${SAVE_INTERMEDIATE}" ]; then
  CMD="${CMD} ${SAVE_INTERMEDIATE}"
fi

if [ ! -z "${USE_API}" ]; then
  CMD="${CMD} ${USE_API}"
fi

# Run WikiTQ training pipeline
echo "Running: ${CMD}"
eval ${CMD} 2>&1 | tee "${TMP_SAVE_PATH}/train_run.log"

echo ""
echo "=============================================="
echo "Training pipeline complete."
echo "Check ${TMP_SAVE_PATH}/train_run.log for results."
echo "=============================================="
