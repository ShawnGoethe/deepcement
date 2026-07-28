---
name: train-model
description: 使用 LAS 测井数据重新训练 XGBoost 评估模型（需人工审批）
metadata:
  author: deepcement
  category: ml
  requires:
    - xgboost
    - scikit-learn
    - numpy
---

# Train XGBoost Model

## When to Use

- 用户要求重新训练/更新 XGBoost 模型
- 新增了大量 LAS 数据需要更新模型
- 模型精度下降需要重训

## Steps

1. **人工审批**：`updateXGBoost` 工具会触发 `interrupt_on`，需用户 approve
2. 扫描 `data/raw/` 下的 LAS/TXT 文件
3. 对每个文件：LasParser 解析 → FeatureExtractor 提取特征 → RuleEngine 生成伪标签
4. 训练 XGBClassifier（100 trees, max_depth=6, 4-class）
5. 保存模型到 `models/xgboost_cement.json`
6. 热更新运行中的 CascadeEvaluator 实例

## Training Parameters

```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=4,           # 优秀/良好/合格/不合格
    eval_metric="mlogloss",
    random_state=42,
)
```

## Important Notes

- 使用半监督方式：规则引擎生成伪标签，非人工标注
- 训练数据不足时结果可能不可靠，建议至少 5 个 LAS 文件
- 训练会覆盖 `models/xgboost_cement.json`，旧模型不会自动备份
- 训练完成后模型立即生效（热更新到内存中的 XGBoostEvaluator 实例）
- 此操作需要人工审批（`interrupt_on` 配置为 approve/reject）

## Validation

训练后使用 `analyze-las` 技能评估一个已知文件，对比训练前后的结果变化。
