#!/bin/bash
# Test script for optimized pipeline (direct function calls, minimal I/O)

# Set Python path
export PYTHONPATH="$(pwd)/..:$(pwd):$PYTHONPATH"

# Set multiprocessing method
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
# Specify GPUs to use (2,3)
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_FLASHINFER=0
# Create output directory
mkdir -p datasets/schedule_test/wikitq_test

# Run optimized pipeline
python run_full_pipeline_wikitq.py \
  --llm_path /data/workspace/yanmy/models/Qwen3-4B-Instruct-2507 \
  --embedding_model_path /data/workspace/yanmy/models/bge-m3 \
  --router_model_path /data/workspace/yanmy/HybridRAG/H-STAR/router/bge-m3-finetuned/ \
  --check_model_path /data/workspace/yanmy/HybridRAG/H-STAR/check/output/bge-reranker-v2-m3-finetuned/ \
  --dataset_name wikitq \
  --split test \
  --tmp_save_path datasets/schedule_test/wikitq_test \
  --tau 0.82 \
  --check_tau 0.8 \
  --n_parallel 32 \
  --tensor_parallel_size 2 \
  --max_model_len 23000 \
  --gpu_memory_utilization 0.7 \
  --select_sample_num 2 \
  --sql_sample_num 3 \
  --temperature 0.7 \
  --top_p 0.8 \
  --first_n 1000 \
  --skip_router
  2>&1 | tee datasets/schedule_test/wikitq_test/test_run.log

echo "Pipeline test complete. Check datasets/schedule_test/wikitq_test/test_run.log for results."