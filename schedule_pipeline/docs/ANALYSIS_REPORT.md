# TableBench PoT Pipeline 分析报告

## 测试概况
- **测试样本数**: 50
- **每样本尝试次数**: 3
- **总执行次数**: 150

## 执行结果统计

### 整体成功率
- ✅ **成功执行**: 83/150 (55.3%)
- ❌ **失败执行**: 67/150 (44.7%)
- 🔍 **解析失败**: 0/150 (0.0%)

### 样本级别统计
- 🟢 **完全成功** (3/3成功): 25/50 (50%)
- 🟡 **部分成功** (1-2/3成功): 5/50 (10%)
- 🔴 **完全失败** (0/3成功): 20/50 (40%)

## 主要问题分析

### 1. 数值转换错误 (最严重，占70%失败)

**错误示例**:
```
Could not convert string '10103119141491412' to numeric
```

**问题根源**:
- `python_executor.py` 将 `pd.read_csv('table.csv')` 替换为 `df_input.copy()`
- 但生成的代码使用变量名 `df`，不是 `df_input`
- 导致代码实际执行时找不到正确的 DataFrame

**受影响的样本**: 
- Sample 0, 3, 9, 11, 12 等多个样本的所有3次尝试都失败

**代码示例**:
```python
# 生成的代码
import pandas as pd
df = pd.read_csv('table.csv')  # 被替换为 df_input.copy()
mean_cyclones = df['tropical cyclones'].mean()  # df 未定义！
```

### 2. Template 与执行器不匹配

**当前 Template 指令**:
```
Ensure to load the table with command `df = pd.read_csv('table.csv')`
```

**执行器实际行为**:
- 替换 `pd.read_csv('table.csv')` → `df_input.copy()`
- 但没有创建 `df` 变量

**不匹配导致的问题**:
1. LLM 按照 template 生成 `df = pd.read_csv('table.csv')`
2. 执行器替换后变成 `df = df_input.copy()`
3. 但 `df_input` 在执行环境中不存在（或存在但后续代码用 `df`）

### 3. 列名错误 (次要问题)

**示例**:
```python
KeyError: 'Goals'  # 实际列名可能是 'goals' 或其他
```

这类错误较少，主要是 LLM 对表格结构理解偏差。

## 解决方案

### 方案 1: 修复 python_executor.py (推荐)

**修改前**:
```python
code_mod = code.replace("pd.read_csv('table.csv')", "df_input.copy()")
exec_context['df_input'] = df
```

**修改后**:
```python
# 直接提供 df 变量，不需要替换
exec_context['df'] = df.copy()
# 同时提供 mocked read_csv 以防万一
exec_context['pd'] = type('pd', (), {
    'read_csv': lambda path, *args, **kwargs: df.copy() if path == 'table.csv' else pd.read_csv(path, *args, **kwargs),
    **{k: v for k, v in pd.__dict__.items() if not k.startswith('_')}
})()
```

### 方案 2: 修改 Prompt Template

**修改 `prompts/python_reason_tablebench.txt`**:

**修改前**:
```
Ensure to load the table with command `df = pd.read_csv('table.csv')`
```

**修改后**:
```
The table data is already loaded in a DataFrame variable named `df`. 
You can directly use `df` to analyze the data without loading it.
```

### 方案 3: 组合方案 (最佳)

1. 修改 `python_executor.py` 直接提供 `df` 变量
2. 更新 template 说明 `df` 已经可用
3. 保留 `pd.read_csv('table.csv')` 的 mock 以兼容旧代码

## 预期改进

实施修复后预期结果:
- **成功率**: 55.3% → **85%+**
- **完全失败样本**: 40% → **<15%**

主要改进来源:
- 解决数值转换错误 (47个失败 → 0)
- 减少变量未定义错误

## 文件位置

生成的文件保存在:
```
datasets/schedule_test/tablebench_pot_enhanced/
├── execution_stats.json          # 统计数据
├── execution_detailed.log        # 详细日志
├── execution_errors.log          # 仅错误日志
├── generated_codes/              # 所有生成的代码
│   ├── sample_0_attempt_0.py
│   ├── sample_0_attempt_0_response.txt
│   └── ...
├── pot_results.csv               # 最终结果
└── evaluation.json               # 评估指标
```

## 下一步行动

1. ✅ 已完成: 运行前50个样本并收集详细日志
2. ✅ 已完成: 分析失败原因
3. 🔄 待执行: 修复 `python_executor.py`
4. 🔄 待执行: 更新 prompt template
5. 🔄 待执行: 重新测试验证改进效果
