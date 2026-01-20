# BGE-M3 Router 模型训练

## 快速开始

本目录包含了针对不同训练样本数量的 BGE-M3 路由模型训练脚本和数据。

## 已训练模型

| 模型名称 | 样本数量 | 训练Loss | 模型路径 |
|---------|---------|---------|---------|
| bge-m3-router-20 | 20 | 2.787 | `/data/workspace/yanmy/SPARQ/models/bge-m3-router-20` |
| bge-m3-router-100 | 100 | 2.654 | `/data/workspace/yanmy/SPARQ/models/bge-m3-router-100` |
| bge-m3-router-400 | 400 | 2.582 | `/data/workspace/yanmy/SPARQ/models/bge-m3-router-400` |
| bge-m3-router-1000 | 1000 | 2.481 | `/data/workspace/yanmy/SPARQ/models/bge-m3-router-1000` |

## 训练数据

- `train_20.jsonl` - 20个训练样本
- `train_100.jsonl` - 100个训练样本
- `train_400.jsonl` - 400个训练样本
- `train_1000.jsonl` - 1000个训练样本

所有数据都经过智能采样，优先选择标签数量<5且分布多样的样本。

## 重新训练

```bash
# 激活环境
conda activate base

# 训练单个模型
bash finetune_20.sh
bash finetune_100.sh
bash finetune_400.sh
bash finetune_1000.sh

# 或批量训练所有模型
bash train_all.sh
```

## 详细信息

查看 [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md) 获取完整的训练细节和分析。

