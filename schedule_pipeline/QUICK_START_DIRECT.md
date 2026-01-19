# 🚀 Direct Answer 版本 - 快速开始

## 📋 一句话总结

**SQL/Python 执行成功 → 直接用结果当答案（跳过 LLM QA）**  
**SQL/Python 执行失败 → LLM 从表格生成答案（Fallback）**

## ⚡ 快速测试（推荐）

```bash
cd /home/yanmy/SPARQ/schedule_pipeline

# SQL Direct - 10 个样本快速验证
./test_sql_iterative_tablebench_direct.sh --first_n 10

# POT Direct - 10 个样本快速验证
./test_pipeline_tablebench_pot_direct.sh --first_n 10
```

## 📊 完整测试

```bash
# SQL Direct - 50 个样本随机采样
./test_sql_iterative_tablebench_direct.sh --first_n 50 --random_sample

# POT Direct - 50 个样本随机采样
./test_pipeline_tablebench_pot_direct.sh --first_n 50 --random_sample
```

## 🎯 核心特性

| 特性 | 说明 |
|------|------|
| ✅ 迭代重试 | 失败时重试 3 次，带错误反馈 |
| ✅ Direct Answer | 成功时直接用执行结果 |
| ✅ LLM Fallback | 失败时 LLM 生成答案 |
| ✅ 答案来源追踪 | `answer_source` 列标记来源 |
| ✅ Bug 修复 | SQL 版本移除 row_id |

## 📁 创建的文件

### 程序文件
- `run_sql_iterative_tablebench_direct.py` - SQL 版本
- `run_pipeline_tablebench_pot_direct.py` - POT 版本
- `test_sql_iterative_tablebench_direct.sh` - SQL 测试脚本
- `test_pipeline_tablebench_pot_direct.sh` - POT 测试脚本

### 文档文件
- `DIRECT_ANSWER_FIX.md` - Bug 修复说明
- `POT_DIRECT_ANSWER.md` - POT 功能说明
- `DIRECT_ANSWER_SUMMARY.md` - 完整总结
- `QUICK_START_DIRECT.md` - 本文档

## 📈 查看结果

测试完成后，结果保存在：
```
datasets/schedule_test/tablebench_sql_direct_YYYYMMDD_HHMMSS/results.csv
datasets/schedule_test/tablebench_pot_direct_YYYYMMDD_HHMMSS/results.csv
```

关键列：
- `answer_source`: 答案来源（`direct_sql`/`direct_python`/`llm_fallback`）
- `prediction`: 预测答案
- `gold_answer`: 正确答案

## 💡 预期改进

- ⚡ **速度**: 节省 30-50% 时间（跳过 LLM QA）
- 💰 **成本**: 减少 30-50% API 调用
- 🎯 **准确率**: 提升 5-10%（直接用执行结果）

## 🔍 对比测试

```bash
# 运行原版本
./test_sql_iterative_tablebench.sh --first_n 50 --random_sample

# 运行 direct 版本
./test_sql_iterative_tablebench_direct.sh --first_n 50 --random_sample

# 对比 results.csv 中的准确率和 answer_source 分布
```

## 📚 详细文档

- 完整功能说明 → `DIRECT_ANSWER_SUMMARY.md`
- POT 版本详情 → `POT_DIRECT_ANSWER.md`
- Bug 修复详情 → `DIRECT_ANSWER_FIX.md`

---

**现在就开始测试吧！** 🎉
