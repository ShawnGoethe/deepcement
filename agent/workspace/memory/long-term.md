# Long-Term Memory

<!-- 本文件是 Agent 的长期记忆，跨会话持久保存。
     Agent 可通过 edit_file 更新本文件，积累经验知识。
     不要删除已有条目，只追加或修改。 -->

## Project Context

DeepCement 是油气田固井质量评测系统，基于 DeepAgent + LlamaIndex + Zilliz Cloud。
核心流程：文档入库 → 向量检索 → LLM/ML 评测 → 报告生成。

## Evaluation History

<!-- Agent 在此记录每次评测的关键发现，供后续评测参考 -->

### 2026-07-27: 威202H16-6 固井质量评测
- **综合得分**: 55.3分（不合格）
- **评估层级**: 规则引擎（高置信度 0.99）
- **关键发现**:
  - 水泥胶结质量差段占比 95.3%，仅 3.5% 优良
  - 全井段扩径（100%），井径变化率 212.35
  - 含气层占比 92.4%，气侵风险高
  - 历史VDL报告（18-360m浅层段）显示胶结优良率 96%，与深层段（360-2546m）差异显著
- **数据来源**: 历史固井报告 + LAS测井数据分层评估

## Learned Patterns

<!-- Agent 在此记录从评测中学到的规律和经验 -->

暂无已学习的规律。

## Model Training Records

<!-- XGBoost 模型训练历史 -->

| Date | Files | Windows | Accuracy | Notes |
|---|---|---|---|---|
| - | - | - | - | 暂无训练记录 |

## User Preferences

<!-- Agent 在此记录用户的偏好和习惯 -->

暂无记录的用户偏好。

## Known Issues

<!-- Agent 在此记录已知的数据问题和注意事项 -->

暂无已知问题。
