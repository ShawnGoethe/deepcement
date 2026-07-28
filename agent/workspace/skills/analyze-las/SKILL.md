---
name: analyze-las
description: 使用分层评估系统分析 LAS/TXT 测井数据（规则引擎 → XGBoost → LLM 兜底）
metadata:
  author: deepcement
  category: evaluation
  requires:
    - xgboost
    - numpy
---

# Analyze LAS Well Log Data

## When to Use

- 用户提供了 LAS/TXT 测井数据文件
- 用户询问某口井的水泥胶结情况
- 用户要求分析声波(AC)、井径(CAL)、电阻率(RT)、伽马(GR) 曲线

## Cascade Evaluation Flow

```
LAS 文件 → LasParser 解析
         → Layer 1: 规则引擎（置信度 > 0.85 → 直接返回）
         → Layer 2: XGBoost（置信度 > 0.75 → 直接返回）
         → Layer 3: LLM 兜底（以上都低置信度时调用）
```

## Steps

1. 确认 LAS 文件路径（支持绝对路径或 `data/raw/` 下的相对路径）
2. 使用 `evaluate_well_log` 工具执行分层评估
3. 解读评估结果：得分、等级、置信度、使用的评估层级

## Supported Curves

| Curve | Description | Feature Extraction |
|---|---|---|
| AC | 声波时差 | mean, std, min, max, median, q25, q75, trend, variability |
| CAL | 井径 | mean, std, range |
| RT | 电阻率 | mean, std, log_mean |
| GR | 伽马 | mean, std |

## Important Notes

- 无效值阈值：测井数据中 ≤ -999 表示缺失，自动跳过
- 滑动窗口大小：32 个采样点，50% 重叠
- XGBoost 未加载时自动降级为规则估算
- 分层评估结果包含 `eval_layer` 字段标识最终使用了哪一层
