# 快速参考

## 🚀 运行测试

```bash
cd /home/yanmy/SPARQ/schedule_pipeline

# 前 100 个样本
./test_pipeline_tablebench_pot_enhanced.sh --first_n 100

# 前 50 个样本（默认）
./test_pipeline_tablebench_pot_enhanced.sh

# 前 10 个样本（快速测试）
./test_pipeline_tablebench_pot_enhanced.sh --first_n 10

# 所有样本（886个）
./test_pipeline_tablebench_pot_enhanced.sh --first_n -1
```

## 📊 查看结果

```bash
# 统计摘要
cat datasets/schedule_test/tablebench_pot_enhanced/execution_stats.json

# 评估指标
cat datasets/schedule_test/tablebench_pot_enhanced/evaluation.json

# 错误日志
head -100 datasets/schedule_test/tablebench_pot_enhanced/execution_errors.log

# 详细日志
less datasets/schedule_test/tablebench_pot_enhanced/execution_detailed.log

# 查看生成的代码
cat datasets/schedule_test/tablebench_pot_enhanced/generated_codes/sample_0_attempt_0.py
```

## 📚 文档

- [README.md](README.md) - 完整使用文档
- [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) - 详细分析报告
- [SUMMARY.md](SUMMARY.md) - 测试总结

## 🔧 常用参数

```bash
--first_n N              # 测试前 N 个样本（-1 = 全部）
--llm_name NAME          # 模型名称（默认: qwen3-4b）
--api_base URL           # API 地址
--api_key KEY            # API 密钥
--tmp_save_path PATH     # 结果保存路径
```

## 📈 当前测试结果（前50个样本）

- ✅ 成功率: **55.3%**
- 🟢 完全成功: **50%**
- 🟡 部分成功: **10%**
- 🔴 完全失败: **40%**

## 💡 主要问题

**Template 与执行器不匹配** - 70%的失败来自这个问题

修复后预期成功率: **85%+**


