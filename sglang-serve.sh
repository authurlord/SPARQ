export CUDA_VISIBLE_DEVICES=0

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /data/workspace/yanmy/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --served-model-name Qwen3-30B-A3B-FP8 \
  --trust-remote-code \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --enable-mixed-chunk \
  --reasoning-parser qwen3 \
  --kt-method AMXINT8 \
  --kt-cpuinfer 16 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 36 \
  --kt-max-deferred-experts-per-token 2
