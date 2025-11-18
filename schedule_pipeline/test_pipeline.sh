#!/bin/bash
# Test script for optimized pipeline (direct function calls, minimal I/O)

# Set Python path
export PYTHONPATH="$(pwd)/..:$(pwd):$PYTHONPATH"

# Set multiprocessing method
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

# Specify GPUs used
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER=0

# model path (repace these with your own paths)
LLM_PATH="/data/workspace/yanmy/models/Qwen3-4B-Instruct-2507"
EMBEDDING_MODEL_PATH="/data/workspace/yanmy/models/bge-m3"
ROUTER_MODEL_PATH="/data/workspace/yanmy/HybridRAG/H-STAR/router/bge-m3-finetuned/"
CHECK_MODEL_PATH="/data/workspace/yanmy/HybridRAG/H-STAR/check/output/bge-reranker-v2-m3-finetuned/"

# Create output directory
TMP_SAVE_PATH="datasets/schedule_test/wikitq_test"
mkdir -p ${TMP_SAVE_PATH}

# Run optimized pipeline
python run_full_pipeline_wikitq.py \
  --llm_path ${LLM_PATH} \
  --embedding_model_path ${EMBEDDING_MODEL_PATH} \
  --router_model_path ${ROUTER_MODEL_PATH} \
  --check_model_path ${CHECK_MODEL_PATH} \
  --dataset_name wikitq \
  --split test \
  --tmp_save_path ${TMP_SAVE_PATH} \
  --tau 0.82 \
  --check_tau 0.8 \
  --n_parallel 32 \
  --tensor_parallel_size 1 \
  --max_model_len 23000 \
  --gpu_memory_utilization 0.7 \
  --select_sample_num 2 \
  --sql_sample_num 3 \
  --temperature 0.7 \
  --top_p 0.8 \
  --first_n 50 \
  --skip_router
  2>&1 | tee "${TMP_SAVE_PATH}/test_run.log"

echo "Pipeline test complete. Check ${TMP_SAVE_PATH}/test_run.log for results."