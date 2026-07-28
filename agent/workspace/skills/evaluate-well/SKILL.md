---
name: evaluate-well
description: 对指定井进行 4 维度固井质量评测并生成报告
metadata:
  author: deepcement
  category: evaluation
  requires:
    - llama-index
---

# Evaluate Well Cementing Quality

## When to Use

- 用户询问某口井的固井质量
- 用户要求生成质量评测报告
- 用户需要对比两口井的数据

## Steps

1. 确认索引已加载（`python main.py --build-index` 或已有 Zilliz 连接）
2. 使用 `search_history` 检索该井的历史资料
3. 使用 `evaluate_quality` 执行 4 维度评测
4. 输出结构化报告：综合得分 + 各维度评分 + 结论 + 建议

## Evaluation Dimensions

| 维度 | 评估要点 |
|---|---|
| 水泥浆性能 | 密度(1.85-1.95 g/cm³)、失水量(<50ml)、稠化时间 |
| 施工参数 | 泵速平稳性、压力安全范围、替浆量准确性 |
| 固井效果 | 水泥返高、候凝时间、CBL 声幅测井、胶结质量 |
| 异常情况 | 漏失、窜槽、气侵、事故处理 |

## Output Format

```
=== 固井质量评测报告: {well_name} ===
综合得分: {score}分 ({grade})

--- 各维度评分 ---
  水泥浆性能: {score}分 ({grade})
  施工参数: {score}分 ({grade})
  固井效果: {score}分 ({grade})
  异常情况: {score}分 ({grade})

--- 结论 ---
{conclusion}

--- 改进建议 ---
1. {suggestion}
```

## Important Notes

- 评测依赖检索结果质量，如果该井无历史资料则无法评测
- LLM 评测结果具有随机性（temperature=0.3），同一井多次评测分数可能略有差异
- 综合得分 = 4 维度分数的算术平均值
