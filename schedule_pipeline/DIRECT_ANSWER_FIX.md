# Direct Answer Version - Bug Fix

## 🐛 问题描述

在之前的测试中，发现 `prediction` 列包含了不应该出现的 `row_id` 值：

```csv
prediction
"0.0, 10.6"    # ❌ 错误：包含了 row_id=0
"0, 1157"      # ❌ 错误：包含了 row_id=0
```

## 🔍 根本原因

SQL 执行结果包含 `row_id` 列，导致：
- 结果 DataFrame 形状：(1, 2) - 单行2列
- 触发了"单行多列"逻辑：返回所有列的值
- 结果：`"row_id, actual_answer"` 而不是只有 `"actual_answer"`

## ✅ 修复方案

在 `extract_direct_answer_from_sql_result()` 函数中添加逻辑：

```python
def extract_direct_answer_from_sql_result(result_df: pd.DataFrame) -> str:
    """Extract direct answer from SQL execution result."""
    if result_df is None or len(result_df) == 0:
        return ""
    
    # ✅ 关键修复：移除 row_id 列
    df = result_df.copy()
    if 'row_id' in df.columns:
        df = df.drop(columns=['row_id'])
    
    # 检查移除后是否为空
    if len(df.columns) == 0 or len(df) == 0:
        return ""
    
    # 然后再进行答案提取逻辑
    # Single cell (1x1) -> return value
    # Single row (1xN) -> return all values
    # Single column (Nx1) -> return all values
    # Multiple rows/cols -> return first cell
```

## 📊 修复效果

修复前：
```
"0.0, 10.6"  -> 包含 row_id
```

修复后：
```
"10.6"       -> 只包含实际答案
```

## 🚀 测试命令

```bash
# 快速测试（10个样本）
./test_sql_iterative_tablebench_direct.sh --first_n 10

# 完整测试（50个样本）
./test_sql_iterative_tablebench_direct.sh --first_n 50 --random_sample
```

## 📝 文件修改

- ✅ `run_sql_iterative_tablebench_direct.py` - 添加 `extract_direct_answer_from_sql_result()` 函数
- ✅ 在 SQL 成功时调用该函数提取答案
- ✅ 添加 `answer_source` 列标记答案来源（direct_sql / llm_fallback）
- ✅ 更新统计信息显示 direct answer 使用率

## 🎯 预期改进

1. **准确率提升**：移除错误的 row_id 值
2. **答案更清晰**：只包含实际答案内容
3. **可追溯性**：`answer_source` 列标记答案来源
