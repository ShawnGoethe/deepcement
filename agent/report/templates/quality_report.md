# 固井质量评测报告

> 生成时间：{{ generated_at }}

## 基本信息

| 项目 | 内容 |
|------|------|
| 井名 | {{ well_name }} |
| 施工日期 | {{ date }} |
| 井深 | {{ well_depth }}m |
| 固井井段 | {{ cement_section }} |

## 综合评分

**{{ overall_score }} 分** {{ grade_emoji }} **{{ overall_grade }}**

## 分项评测

| 维度 | 得分 | 等级 | 说明 |
|------|------|------|------|
{% for dim in dimensions -%}
| {{ dim.name }} | {{ dim.score }} | {{ dim.grade }} | {{ dim.details_short }} |
{% endfor %}

{% for dim in dimensions %}
### {{ dim.name }}

- **得分**: {{ dim.score }} 分 ({{ dim.grade }})
- **详细说明**: {{ dim.details }}
{% if dim.issues %}
- **发现的问题**:
{% for issue in dim.issues %}
  - ⚠️ {{ issue }}
{% endfor %}
{% endif %}
{% endfor %}

## 评测结论

{{ conclusion }}

## 改进建议

{% for sug in suggestions %}
{{ loop.index }}. {{ sug }}
{% endfor %}

---

*本报告由 DeepCement 固井质量评测系统自动生成*
