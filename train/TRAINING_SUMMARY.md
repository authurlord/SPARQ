# 不同训练样本数量的模型训练总结

## 任务概述
测试不同训练样本数量对模型性能的影响，训练了4个不同规模的 BGE-M3 路由模型。

## 数据准备

### 数据源
- 原始数据：`/data/workspace/yanmy/HybridRAG/H-STAR/router/finetune_data.jsonl`
- 总数据量：2591 条
- 数据特点：
  - 标签数量<5的数据：1099 条
  - 标签数量>=5的数据：1492 条

### 采样策略
使用贪心算法进行采样，优先选择：
1. 标签数量小于5的数据
2. 标签分布多样性高的数据

采样使用了多样性得分计算，确保各个标签在训练集中均匀分布。

## 训练配置

### 模型参数
- 基础模型：`/data/workspace/yanmy/models/bge-m3`
- GPU数量：1 (修改自原来的2)
- 训练轮数：2 epochs
- 学习率：1e-5
- Batch size：4 per device
- 最大query长度：2048
- 最大passage长度：1024
- 保存策略：不保存checkpoint (save_strategy=no)

### 训练环境
- Conda环境：base
- 使用 DeepSpeed 加速
- FP16 混合精度训练

## 训练结果

### 1. 20样本模型 (bge-m3-router-20)
- 训练数据：`train_20.jsonl`
- 样本数量：20
- 标签分布：
  - Select_Row: 8
  - Base: 7
  - Execute_SQL: 7
  - Select_Column: 7
  - RAG_20_5: 7
  - RAG_10_3: 7
- 标签数量分布：{1: 11, 4: 6, 3: 2, 2: 1}
- 平均标签数量：2.15
- 训练时间：3.87秒
- 训练速度：2.58 samples/sec
- 最终loss：2.787
- 模型大小：1.1GB
- 输出路径：`/data/workspace/yanmy/SPARQ/models/bge-m3-router-20`

### 2. 100样本模型 (bge-m3-router-100)
- 训练数据：`train_100.jsonl`
- 样本数量：100
- 标签分布：
  - Base: 34
  - Select_Column: 34
  - Execute_SQL: 34
  - Select_Row: 34
  - RAG_20_5: 34
  - RAG_10_3: 34
- 标签数量分布：{1: 64, 4: 33, 3: 2, 2: 1}
- 平均标签数量：2.04
- 训练时间：11.86秒
- 训练速度：4.22 samples/sec
- 最终loss：2.654
- 模型大小：1.1GB
- 输出路径：`/data/workspace/yanmy/SPARQ/models/bge-m3-router-100`

### 3. 400样本模型 (bge-m3-router-400)
- 训练数据：`train_400.jsonl`
- 样本数量：400
- 标签分布：
  - Select_Column: 172
  - Base: 171
  - Select_Row: 171
  - Execute_SQL: 171
  - RAG_20_5: 171
  - RAG_10_3: 171
- 标签数量分布：{4: 164, 1: 137, 2: 63, 3: 36}
- 平均标签数量：2.57
- 训练时间：47.09秒
- 训练速度：4.25 samples/sec
- 最终loss：2.582
- 模型大小：1.1GB
- 输出路径：`/data/workspace/yanmy/SPARQ/models/bge-m3-router-400`

### 4. 1000样本模型 (bge-m3-router-1000)
- 训练数据：`train_1000.jsonl`
- 样本数量：1000
- 标签分布：
  - Select_Row: 605
  - Base: 568
  - Select_Column: 555
  - RAG_20_5: 475
  - Execute_SQL: 474
  - RAG_10_3: 262
- 标签数量分布：{4: 416, 3: 247, 2: 197, 1: 140}
- 平均标签数量：2.94
- 训练时间：117.96秒
- 训练速度：4.24 samples/sec
- 最终loss：2.481
- 模型大小：1.1GB
- 输出路径：`/data/workspace/yanmy/SPARQ/models/bge-m3-router-1000`

## 观察与分析

### Loss趋势
随着训练样本数量增加，最终loss逐渐降低：
- 20样本：2.787
- 100样本：2.654
- 400样本：2.582
- 1000样本：2.481

这表明更多的训练数据有助于模型更好地学习任务。

### 训练效率
- 所有模型的训练速度都在 4.2-4.3 samples/sec 左右（除了20样本模型稍慢）
- 使用单GPU训练，效率稳定

### 标签分布
- 20和100样本模型：标签分布非常均匀，主要包含标签数量为1的简单样本
- 400和1000样本模型：包含更多标签数量为4的复杂样本，标签分布更加多样化

## 文件结构

```
/data/workspace/yanmy/SPARQ/
├── train/
│   ├── train_20.jsonl          # 20样本训练数据
│   ├── train_100.jsonl         # 100样本训练数据
│   ├── train_400.jsonl         # 400样本训练数据
│   ├── train_1000.jsonl        # 1000样本训练数据
│   ├── finetune_20.sh          # 20样本训练脚本
│   ├── finetune_100.sh         # 100样本训练脚本
│   ├── finetune_400.sh         # 400样本训练脚本
│   ├── finetune_1000.sh        # 1000样本训练脚本
│   ├── train_all.sh            # 批量训练脚本
│   ├── sample_from_jsonl.py    # 数据采样脚本
│   ├── train_20.log            # 20样本训练日志
│   ├── train_100.log           # 100样本训练日志
│   ├── train_400.log           # 400样本训练日志
│   └── train_1000.log          # 1000样本训练日志
└── models/
    ├── bge-m3-router-20/       # 20样本模型
    ├── bge-m3-router-100/      # 100样本模型
    ├── bge-m3-router-400/      # 400样本模型
    └── bge-m3-router-1000/     # 1000样本模型
```

## 下一步

1. **模型评估**：在验证集上评估这4个模型的性能
2. **性能对比**：比较不同样本数量对模型准确率、召回率等指标的影响
3. **最优选择**：根据评估结果选择最佳的训练样本数量

## 使用方法

### 重新训练单个模型
```bash
cd /data/workspace/yanmy/SPARQ/train
conda activate base
bash finetune_20.sh    # 或其他训练脚本
```

### 批量训练所有模型
```bash
cd /data/workspace/yanmy/SPARQ/train
bash train_all.sh
```

### 加载训练好的模型
```python
from FlagEmbedding import BGEM3FlagModel

# 加载特定样本数量的模型
model = BGEM3FlagModel('/data/workspace/yanmy/SPARQ/models/bge-m3-router-100')
```

---
生成时间：2026-01-20
训练环境：base conda环境
GPU：单卡训练

