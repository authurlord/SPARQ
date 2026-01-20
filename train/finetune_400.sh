#!/bin/bash

# 训练 400 样本模型
# 使用 1 个 GPU，不保存 checkpoint

torchrun --nproc_per_node 1 \
    -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
    --model_name_or_path /data/workspace/yanmy/models/bge-m3 \
    --train_data ./train_400.jsonl \
    --output_dir /data/workspace/yanmy/SPARQ/models/bge-m3-router-400 \
    --overwrite_output_dir \
    --knowledge_distillation True \
    --cache_path ./cache_data \
    --train_group_size 8 \
    --query_max_len 2048 \
    --passage_max_len 1024 \
    --pad_to_multiple_of 8 \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ./ds_config.json \
    --logging_steps 10 \
    --save_strategy no \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --unified_finetuning True \
    --use_self_distill True \
    --fix_encoder False \
    --self_distill_start_step 0

