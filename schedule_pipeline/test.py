python run_full_pipeline_wikitq.py \
  --llm_path /data/workspace/yanmy/models/Qwen3-4b-Instruct \
  --dataset_name wikitq \
  --split test \
  --tmp_save_path tmp/wikitq_qwen3_max \
  --llm_concurrency 2