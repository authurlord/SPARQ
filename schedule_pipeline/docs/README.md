# TableBench PoT Pipeline 使用文档

## 快速开始

### 运行测试

#### 1. 启动 vLLM 服务器

在一个终端中启动 LLM 服务：

```bash
cd /home/yanmy/SPARQ
sh llm_server.sh
```

等待服务器启动完成（看到 "Application startup complete" 消息）。

#### 2. 运行测试脚本

在另一个终端中运行测试：

```bash
cd /home/yanmy/SPARQ/schedule_pipeline

# 测试前 50 个样本（默认）
./test_pipeline_tablebench_pot_enhanced.sh

# 测试前 100 个样本
./test_pipeline_tablebench_pot_enhanced.sh --first_n 100

# 测试前 10 个样本（快速测试）
./test_pipeline_tablebench_pot_enhanced.sh --first_n 10

# 测试所有样本（886个）
./test_pipeline_tablebench_pot_enhanced.sh --first_n -1
```

### 命令参数说明

```bash
./test_pipeline_tablebench_pot_enhanced.sh [OPTIONS]

选项:
  --first_n N              测试前 N 个样本（默认: 50, -1 表示全部）
  --llm_name NAME          模型名称（默认: qwen3-4b）
  --api_base URL           API 地址（默认: http://localhost:8000/v1）
  --api_key KEY            API 密钥（默认: api-key-qwen3）
  --tmp_save_path PATH     结果保存路径
```

### 示例命令

```bash
# 运行前 100 个样本
./test_pipeline_tablebench_pot_enhanced.sh --first_n 100

# 使用不同的模型
./test_pipeline_tablebench_pot_enhanced.sh --first_n 50 --llm_name qwen3-7b

# 自定义保存路径
./test_pipeline_tablebench_pot_enhanced.sh --first_n 100 --tmp_save_path datasets/test_100
```

## 测试结果

### 生成的文件

测试完成后，结果保存在 `datasets/schedule_test/tablebench_pot_enhanced/`：

```
datasets/schedule_test/tablebench_pot_enhanced/
├── execution_stats.json          # 统计摘要（JSON格式）
├── execution_detailed.log        # 详细执行日志
├── execution_errors.log          # 仅包含错误的日志
├── generated_codes/              # 所有生成的代码
│   ├── sample_0_attempt_0.py           # 提取的 Python 代码
│   ├── sample_0_attempt_0_response.txt # LLM 原始响应
│   ├── sample_0_attempt_1.py
│   ├── sample_0_attempt_1_response.txt
│   └── ... (每个样本3次尝试 × 2文件)
├── pot_results.csv               # 最终预测结果
├── evaluation.json               # ROUGE-L 评估指标
└── test_run.log                  # 完整运行日志
```

### 查看结果

```bash
# 查看统计摘要
cat datasets/schedule_test/tablebench_pot_enhanced/execution_stats.json

# 查看评估指标
cat datasets/schedule_test/tablebench_pot_enhanced/evaluation.json

# 查看前 100 行错误日志
head -100 datasets/schedule_test/tablebench_pot_enhanced/execution_errors.log

# 查看某个样本的详细日志
grep -A 20 "Sample 0" datasets/schedule_test/tablebench_pot_enhanced/execution_detailed.log

# 查看生成的代码
cat datasets/schedule_test/tablebench_pot_enhanced/generated_codes/sample_0_attempt_0.py

# 查看 LLM 原始响应
cat datasets/schedule_test/tablebench_pot_enhanced/generated_codes/sample_0_attempt_0_response.txt
```

## 测试结果分析（前50个样本）

### 执行统计

- **总样本数**: 50
- **总尝试次数**: 150 (每样本3次)
- **成功执行**: 83 (55.3%)
- **失败执行**: 67 (44.7%)
- **解析失败**: 0 (0.0%)

### 样本级别统计

- 🟢 **完全成功** (3/3成功): 25/50 (50%)
- 🟡 **部分成功** (1-2/3成功): 5/50 (10%)
- 🔴 **完全失败** (0/3成功): 20/50 (40%)

### 主要问题

#### 1. 数值转换错误 (70%的失败)

**错误示例**:
```
Could not convert string '10103119141491412' to numeric
```

**根本原因**:
- `python_executor.py` 的代码替换逻辑有问题
- Template 要求使用 `pd.read_csv('table.csv')`
- 执行器替换为 `df_input.copy()`，但变量名不匹配
- 导致 DataFrame 未正确加载到执行环境

#### 2. 列名错误 (次要问题)

**示例**:
```python
KeyError: 'Goals'  # 实际列名可能是 'goals'
```

LLM 对表格结构理解偏差导致。

## 解决方案

### 推荐修复: python_executor.py

**当前代码** (`utils/python_executor.py`):
```python
code_mod = code.replace("pd.read_csv('table.csv')", "df_input.copy()")
exec_context['df_input'] = df
```

**修复后**:
```python
# 直接提供 df 变量，无需替换代码
exec_context = {
    'pd': pd,
    'np': np,
    'scipy': scipy,
    'df': df.copy(),  # 直接提供 df
}

# 不再替换代码，直接执行
try:
    with contextlib.redirect_stdout(output_capture):
        exec(code, exec_context)
    return output_capture.getvalue()
except Exception as e:
    return f"Execution Error: {str(e)}"
```

### 预期改进效果

修复后预期:
- **成功率**: 55.3% → **85%+** (提升30%)
- **完全失败样本**: 40% → **<15%** (减少25%)

## Pipeline 架构

### 执行流程

```
1. 数据加载
   ↓
2. 生成 Python 代码 (n=3 samples)
   ↓
3. 执行 Python 代码 & 选择结果
   ↓
4. 生成最终答案 (使用执行结果作为上下文)
   ↓
5. 评估 (ROUGE-L)
```

### 关键组件

- **`run_pipeline_tablebench_pot_enhanced.py`**: 增强版主程序
  - 保存所有中间结果
  - 详细日志记录
  - 统计分析

- **`utils/async_llm.py`**: 异步 LLM 调用
  - 支持并发请求
  - 自动重试机制
  - 性能监控

- **`utils/python_executor.py`**: Python 代码执行器
  - 安全的代码执行环境
  - 捕获输出和错误
  - DataFrame 传递

- **`prompts/python_reason_tablebench.txt`**: PoT 提示模板
  - 指导 LLM 生成 Python 代码
  - 定义输出格式

## 性能优化

### 并发设置

```bash
# 默认并发设置
--n_parallel 32          # 并行工作进程数
--llm_concurrency 32     # 最大并发请求数
```

### 采样参数

```bash
--code_sample_num 3      # 每个样本生成3次代码
--temperature 0.7        # 采样温度
--top_p 0.8             # Top-p 采样
```

## 故障排查

### 问题 1: LLM 服务器未启动

**错误**: `Connection refused` 或 `401 Unauthorized`

**解决**:
```bash
# 检查服务器是否运行
curl http://localhost:8000/health

# 如果未运行，启动服务器
cd /home/yanmy/SPARQ
sh llm_server.sh
```

### 问题 2: API Key 错误

**错误**: `401 Unauthorized`

**解决**: 确保 API Key 匹配
- `llm_server.sh` 中设置: `VLLM_API_KEY="api-key-qwen3"`
- 测试脚本中使用: `--api_key "api-key-qwen3"`

### 问题 3: 内存不足

**错误**: CUDA out of memory

**解决**:
- 减少并发数: `--llm_concurrency 16`
- 减少批次大小
- 调整 GPU 内存利用率（在 `llm_server.sh` 中）

## 开发指南

### 添加新的测试

1. 复制并修改测试脚本
2. 调整参数和路径
3. 运行测试

### 修改 Prompt Template

编辑 `prompts/python_reason_tablebench.txt`：
- 修改指令
- 添加示例
- 调整输出格式

### 扩展执行器

修改 `utils/python_executor.py`：
- 添加新的库支持
- 增强错误处理
- 优化性能

## 常见问题 (FAQ)

### Q: 如何运行前 100 个样本？
```bash
./test_pipeline_tablebench_pot_enhanced.sh --first_n 100
```

### Q: 如何查看某个样本的生成代码？
```bash
cat datasets/schedule_test/tablebench_pot_enhanced/generated_codes/sample_0_attempt_0.py
```

### Q: 如何分析失败原因？
```bash
# 查看错误日志
cat datasets/schedule_test/tablebench_pot_enhanced/execution_errors.log

# 查看详细日志
cat datasets/schedule_test/tablebench_pot_enhanced/execution_detailed.log
```

### Q: 如何提高成功率？
1. 修复 `python_executor.py` 的 DataFrame 传递问题
2. 优化 Prompt Template
3. 增加代码生成次数（`--code_sample_num`）

### Q: 测试需要多长时间？
- 前 50 个样本: ~1-2 分钟
- 前 100 个样本: ~2-4 分钟
- 全部 886 个样本: ~20-30 分钟

（时间取决于 GPU 性能和并发设置）

## 相关文档

- [分析报告](../ANALYSIS_REPORT.md) - 详细的问题分析
- [总结文档](../SUMMARY.md) - 测试总结和洞察
- [原始脚本](../test_pipeline_tablebench_pot.sh) - 基础版测试脚本
- [增强脚本](../test_pipeline_tablebench_pot_enhanced.sh) - 增强版测试脚本

## 更新日志

### 2025-01-19
- ✅ 创建增强版测试脚本
- ✅ 添加详细日志记录
- ✅ 保存所有中间结果
- ✅ 完成前50个样本测试
- ✅ 分析主要问题和解决方案
- ✅ 创建完整文档

## 联系方式

如有问题或建议，请查看项目文档或联系开发团队。


