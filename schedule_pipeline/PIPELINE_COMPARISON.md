# Pipeline 脚本对比

## 三个脚本的关系

| 特性 | test_pipeline_tablebench.sh | test_pipeline_api.sh | train_pipeline_wikitq.sh (新) |
|------|---------------------------|---------------------|----------------------------|
| **数据集** | TableBench | WikiTQ | WikiTQ |
| **用途** | 测试 | 测试 | 训练 |
| **Python脚本** | run_full_pipeline_tablebench.py | run_full_pipeline_wikitq_api.py | run_full_pipeline_wikitq_api.py |
| **默认split** | - | test | train |
| **参数解析** | ✅ 完整CLI | ❌ 简单 | ✅ 完整CLI |
| **API支持** | ✅ | ✅ | ✅ |
| **输出日志** | test_run.log | test_run.log | train_run.log |
| **默认输出目录** | tmp/tablebench_test | datasets/schedule_test/wikitq_api | tmp/wikitq_train |

## 新脚本的特点

`train_pipeline_wikitq.sh` 结合了两个参考脚本的优点：

### 从 test_pipeline_tablebench.sh 继承
- ✅ 完整的命令行参数解析（支持 --help）
- ✅ 清晰的输出格式和日志记录
- ✅ 灵活的参数覆盖机制
- ✅ 详细的运行信息显示

### 从 test_pipeline_api.sh 继承
- ✅ WikiTQ 数据集的核心参数
- ✅ API 配置（api_base, api_key, model_name）
- ✅ Pipeline 参数（tau, check_tau, temperature, top_p 等）
- ✅ 调用 run_full_pipeline_wikitq_api.py

### 新增特性
- ✅ 默认 split 改为 "train"（适合训练场景）
- ✅ 输出日志改为 "train_run.log"
- ✅ 默认输出目录改为 "tmp/wikitq_train"
- ✅ 支持所有 API 和 Pipeline 参数的命令行覆盖

## 使用场景对比

### test_pipeline_tablebench.sh
```bash
# 测试 TableBench 数据集
./test_pipeline_tablebench.sh --first_n 100
```

### test_pipeline_api.sh
```bash
# 测试 WikiTQ 数据集（测试集）
./test_pipeline_api.sh
```

### train_pipeline_wikitq.sh (新)
```bash
# 训练 WikiTQ 数据集（训练集）
./train_pipeline_wikitq.sh --first_n 1000

# 使用不同的路由模型
./train_pipeline_wikitq.sh \
  --router_model /data/workspace/yanmy/SPARQ/models/bge-m3-router-100 \
  --output tmp/wikitq_router_100_train

# 在验证集上运行
./train_pipeline_wikitq.sh --split validation --output tmp/wikitq_validation
```

## 参数映射

| 功能 | test_pipeline_api.sh | train_pipeline_wikitq.sh |
|------|---------------------|-------------------------|
| API地址 | 硬编码 | --api_base |
| API密钥 | 硬编码 | --api_key |
| 模型名称 | 硬编码 | --model_name |
| 嵌入模型 | --embedding_model_path | --embedding_model |
| 路由模型 | --router_model_path | --router_model |
| 检查模型 | --check_model_path | --check_model |
| 输出目录 | --tmp_save_path | --output 或 --tmp |
| 数据集划分 | --split | --split |
| 温度 | --temperature | --temperature |
| Top-p | --top_p | --top_p |
| 最大tokens | --max_tokens | --max_tokens |
| 并发数 | --concurrency | --concurrency |

## 推荐工作流

1. **快速测试**（使用少量样本）
   ```bash
   ./train_pipeline_wikitq.sh --first_n 10 --save_intermediate
   ```

2. **小规模训练**（100-1000样本）
   ```bash
   ./train_pipeline_wikitq.sh --first_n 1000 --save_intermediate
   ```

3. **完整训练**
   ```bash
   ./train_pipeline_wikitq.sh --split train --save_intermediate
   ```

4. **验证评估**
   ```bash
   ./train_pipeline_wikitq.sh --split validation --output tmp/wikitq_val
   ```

5. **测试不同路由模型**
   ```bash
   for size in 20 100 400 1000; do
     ./train_pipeline_wikitq.sh \
       --router_model /data/workspace/yanmy/SPARQ/models/bge-m3-router-${size} \
       --output tmp/wikitq_router_${size} \
       --first_n 500
   done
   ```

