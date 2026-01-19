#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
N_GPU=2
export VLLM_ATTENTION_BACKEND=FLASHINFER
# 必须添加这一行，否则 /sleep 和 /wake_up 接口会报 404
# export VLLM_SERVER_DEV_MODE=1 

MODEL_PATH="../models/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_NAME="qwen3-4b"
HOST="0.0.0.0"
PORT=8000
VLLM_API_KEY="api-key-qwen3"
GPU_RAM=0.75

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
  --kv-cache-dtype fp8 \
  --async-scheduling \
  --enable-expert-parallel