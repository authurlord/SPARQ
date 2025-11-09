#!/bin/bash
# Test script for H-STAR pipeline with vLLM API

# Set environment
export PYTHONPATH="$(pwd)/..:$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1

# Create output directory
mkdir -p datasets/schedule_test/wikitq_api

# API configuration
api_base="http://127.0.0.1:8000/v1"
model_name="/public/Qwen3-4B-Instruct-2507"  # Should match the model name on your vLLM server

# Model paths (local embedding models)
embedding_model_path="/home/yanmy/sentence_transformers/bge-m3"
router_model_path="/home/yanmy/HybridRAG/H-STAR/router/bge-m3-finetuned"
check_model_path="/home/yanmy/HybridRAG/H-STAR/check/output/bge-reranker-v2-m3-finetuned"

# Run pipeline
python run_full_pipeline_wikitq_api.py \
    --api_base "$api_base" \
    --api_key "api-key-qwen3" \
    --model_name "$model_name" \
    --concurrency 512 \
    --embedding_model_path "$embedding_model_path" \
    --router_model_path "$router_model_path" \
    --check_model_path "$check_model_path" \
    --dataset_name wikitq \
    --split test \
    --tmp_save_path datasets/schedule_test/wikitq_api \
    --tau 0.82 \
    --check_tau 0.8 \
    --n_parallel 32 \
    --select_sample_num 2 \
    --sql_sample_num 3 \
    --temperature 0.7 \
    --top_p 0.8 \
    --max_tokens 2048 \
    --first_n -1 \
    --save_intermediate \
    2>&1 | tee datasets/schedule_test/wikitq_api/test_run.log

echo "Test completed. Check log at: datasets/schedule_test/wikitq_api/test_run.log"

