# POT Pipeline - Direct Answer Version

## 🎯 功能说明

基于 `run_pipeline_tablebench_pot_enhanced_v2.py` 创建的新版本，添加了：
1. **迭代重试机制**：Python 执行失败时，将错误信息反馈给 LLM，重新生成代码（最多 3 次迭代）
2. **Direct Answer 模式**：Python 执行成功时，直接使用执行结果作为答案（跳过 LLM QA 步骤）
3. **LLM Fallback 模式**：Python 执行失败时，使用 LLM 从原始表格生成答案

## 📊 与 SQL Direct Answer 版本的对比

| 特性 | SQL Direct | POT Direct |
|------|-----------|-----------|
| 执行内容 | SQL 查询 | Python 代码 |
| 迭代重试 | ✅ 3 次 | ✅ 3 次 |
| Direct Answer | ✅ SQL 结果 | ✅ Python 输出 |
| LLM Fallback | ✅ | ✅ |
| 答案来源标记 | `direct_sql` / `llm_fallback` | `direct_python` / `llm_fallback` |

## 🔧 核心改进

### 1. **迭代重试逻辑**

```python
for iteration in range(max_iterations):
    # 生成 Python 代码
    if iteration > 0:
        # 添加错误反馈到 prompt
        prompt += f"[Previous Error]: {error_msg}"
    
    # 执行代码
    for code in generated_codes:
        result = execute_python_code(code, df)
        if success:
            break  # 成功则停止
    
    if success:
        break  # 该样本成功，移除
```

### 2. **Direct Answer 提取**

```python
def extract_direct_answer_from_execution(execution_result: str) -> str:
    """直接使用 Python 执行结果作为答案"""
    if not execution_result or "Execution Error" in execution_result:
        return ""
    
    # 清理结果
    result = execution_result.strip()
    
    # 移除常见前缀
    for prefix in ['Answer:', 'Result:', 'Output:']:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
    
    return result[:200]  # 限制长度
```

### 3. **答案生成流程**

```
Python 成功 → 直接使用执行结果 (answer_source = 'direct_python')
     ↓
Python 失败 → LLM 从表格生成答案 (answer_source = 'llm_fallback')
```

## 🚀 使用方法

### 快速测试（10 个样本）
```bash
cd /home/yanmy/SPARQ/schedule_pipeline
./test_pipeline_tablebench_pot_direct.sh --first_n 10
```

### 完整测试（50 个样本）
```bash
./test_pipeline_tablebench_pot_direct.sh --first_n 50 --random_sample
```

### 大规模测试（100 个样本）
```bash
./test_pipeline_tablebench_pot_direct.sh --first_n 100 --random_sample
```

## 📈 输出统计

测试完成后会显示：

### 1. **执行统计**
```
Total samples: 50
Total Python Attempts: 150
Successful Python Executions: 120 (80.0%)
Sample Success Rate: 90.0%
Direct answer count: 45 (90.0%)
```

### 2. **答案来源统计**
```
Answer Sources:
  Direct Python Answer: 45 (90.0%)
  LLM Fallback: 5 (10.0%)
```

### 3. **评估结果**
```
Average ROUGE-L: 0.8234
Accuracy@0.5: 85.00%
Accuracy@0.8: 72.00%
```

## 📁 输出文件

所有结果保存在 `datasets/schedule_test/tablebench_pot_direct_YYYYMMDD_HHMMSS/`：

1. **`results.csv`** - 主要结果
   - `answer_source`: 答案来源（`direct_python` / `llm_fallback`）
   - `python_success`: Python 是否执行成功
   - `total_attempts`: 总尝试次数
   - `successful_attempts`: 成功次数
   - `iterations_used`: 使用的迭代次数

2. **`execution_stats_detailed.json`** - 详细执行统计
   - 每个样本的所有迭代记录
   - 每次尝试的代码、结果、错误信息

3. **`execution_summary.json`** - 汇总统计
   - 总体成功率
   - Direct answer 使用率
   - 迭代统计

4. **`generated_codes/`** - 生成的代码文件
   - `sample_{idx}_iter_{iteration}_attempt_{attempt}.py`
   - `sample_{idx}_iter_{iteration}_attempt_{attempt}_response.txt`

5. **`evaluation.json`** - 评估指标

## 🔍 与原版本对比

### 原版本 (v2)
- ❌ 无迭代重试
- ❌ 总是使用 LLM QA 步骤
- ✅ 详细日志

### Direct Answer 版本
- ✅ 3 次迭代重试
- ✅ 直接使用 Python 结果（跳过 LLM QA）
- ✅ LLM Fallback 机制
- ✅ 答案来源追踪
- ✅ 详细统计

## 💡 预期改进

1. **准确率提升**：
   - 迭代重试提高 Python 执行成功率
   - 直接使用 Python 结果更准确

2. **速度提升**：
   - Python 成功的样本跳过 LLM QA 步骤
   - 预计节省 50%+ 的 LLM 调用

3. **成本降低**：
   - 减少 LLM 调用次数
   - 只在必要时使用 LLM

4. **可追溯性**：
   - `answer_source` 清楚标记答案来源
   - 详细的迭代记录

## 🎯 测试建议

1. **先小规模测试**：`--first_n 10` 验证功能
2. **对比测试**：同时运行原版本和 direct 版本
3. **分析差异**：对比 `answer_source` 分布和准确率

## 📝 相关文件

- ✅ `run_pipeline_tablebench_pot_direct.py` - 主程序
- ✅ `test_pipeline_tablebench_pot_direct.sh` - 测试脚本
- ✅ `POT_DIRECT_ANSWER.md` - 本文档
