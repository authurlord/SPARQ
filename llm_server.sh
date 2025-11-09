#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=FLASHINFER
MODEL_PATH="/data/workspace/yanmy/models/Qwen3-4B-Instruct-2507"
MODEL_NAME="qwen3-4b"
N_GPU=2
HOST="0.0.0.0"
PORT=8000
VLLM_API_KEY="api-key-qwen3"
# 24 * 0.85 = 20.4GB
GPU_RAM=0.75
# GPU_RAM=0.70 ## for pipeline

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --tensor-parallel-size "${N_GPU}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype auto \
  --gpu-memory-utilization ${GPU_RAM} \
  --max-model-len 23000 \
  --api-key "$VLLM_API_KEY" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max_num_seqs 256 \
  --kv-cache-dtype fp8
