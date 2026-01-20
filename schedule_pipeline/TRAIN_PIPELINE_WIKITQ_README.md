# WikiTQ Training Pipeline 使用说明

## 概述

`train_pipeline_wikitq.sh` 是用于 WikiTQ 数据集训练的完整 pipeline 脚本，结合了 `test_pipeline_tablebench.sh` 的结构和 `test_pipeline_api.sh` 的参数配置。

## 快速开始

### 基本用法

```bash
# 使用默认参数运行
./train_pipeline_wikitq.sh

# 指定训练集的前100个样本
./train_pipeline_wikitq.sh --first_n 100

# 使用自定义模型路径
./train_pipeline_wikitq.sh \
  --router_model /path/to/router/model \
  --check_model /path/to/check/model \
  --embedding_model /path/to/embedding/model
```

### 完整示例

```bash
./train_pipeline_wikitq.sh \
  --llm_path "../../models/Qwen3-4B-Instruct-2507" \
  --embedding_model "../../models/bge-m3" \
  --router_model "../../HybridRAG/H-STAR/router/wikitq" \
  --check_model "../../HybridRAG/H-STAR/check/wikitq" \
  --output "tmp/wikitq_train_custom" \
  --split train \
  --tau 0.82 \
  --check_tau 0.8 \
  --temperature 0.7 \
  --top_p 0.8 \
  --max_tokens 2048 \
  --concurrency 512 \
  --n_parallel 32 \
  --first_n 1000 \
  --save_intermediate
```

## 主要参数

### 模型路径
- `--llm_path, --llm`: LLM 模型路径（默认：`../../models/Qwen3-4B-Instruct-2507`）
- `--embedding_model, --embedding`: 嵌入模型路径（默认：`../../models/bge-m3`）
- `--router_model, --router`: 路由模型路径（默认：`../../HybridRAG/H-STAR/router/wikitq`）
- `--check_model, --check`: 检查/重排序模型路径（默认：`../../HybridRAG/H-STAR/check/wikitq`）

### API 配置
- `--api_base`: API 基础 URL（默认：`http://127.0.0.1:8000/v1`）
- `--api_key`: API 密钥（默认：`api-key-qwen3`）
- `--model_name`: API 模型名称（默认：`/public/Qwen3-4B-Instruct-2507`）
- `--concurrency`: API 并发数（默认：512）

### 数据集配置
- `--split`: 数据集划分（train/validation/test，默认：train）
- `--output, --tmp`: 输出目录（默认：`tmp/wikitq_train`）
- `--first_n`: 只处理前 N 个样本

### Pipeline 参数
- `--tau`: 路由阈值（默认：0.82）
- `--check_tau`: 检查阈值（默认：0.8）
- `--n_parallel`: 并行工作进程数（默认：32）
- `--select_sample_num`: 选择样本数量（默认：2）
- `--sql_sample_num`: SQL 样本数量（默认：3）

### 生成参数
- `--temperature`: 采样温度（默认：0.7）
- `--top_p`: Top-p 采样（默认：0.8）
- `--max_tokens`: 最大 token 数（默认：2048）

### 其他选项
- `--save_intermediate`: 保存中间结果
- `--use_api`: 使用 API 模式
- `-h, --help`: 显示帮助信息

## 输出

脚本会在指定的输出目录中生成：
- `train_run.log`: 完整的训练日志
- 其他中间结果文件（如果使用 `--save_intermediate`）

## 与其他脚本的对比

### vs test_pipeline_tablebench.sh
- 保留了相同的命令行参数解析结构
- 保留了清晰的输出格式和日志记录
- 适配了 WikiTQ 数据集的特定需求

### vs test_pipeline_api.sh
- 使用了相同的 API 配置参数
- 使用了相同的 pipeline 核心参数（tau, check_tau, temperature 等）
- 调用 `run_full_pipeline_wikitq_api.py` 而不是 tablebench 版本

## 注意事项

1. 确保 vLLM API 服务已启动并运行在指定的 `--api_base` 地址
2. 确保所有模型路径都正确且可访问
3. 根据 GPU 内存调整 `--concurrency` 和 `--n_parallel` 参数
4. 训练时建议使用 `--save_intermediate` 保存中间结果以便调试

## 示例场景

### 场景 1: 快速测试（前100个样本）
```bash
./train_pipeline_wikitq.sh --first_n 100 --save_intermediate
```

### 场景 2: 完整训练集
```bash
./train_pipeline_wikitq.sh --split train --save_intermediate
```

### 场景 3: 使用验证集
```bash
./train_pipeline_wikitq.sh --split validation --output tmp/wikitq_validation
```

### 场景 4: 自定义路由模型测试
```bash
./train_pipeline_wikitq.sh \
  --router_model /data/workspace/yanmy/SPARQ/models/bge-m3-router-100 \
  --output tmp/wikitq_router_100 \
  --first_n 500
```

