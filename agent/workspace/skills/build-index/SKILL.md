---
name: build-index
description: 从 data/raw/ 目录读取文档并构建 LlamaIndex 向量索引，写入 Zilliz Cloud
metadata:
  author: deepcement
  category: data
  requires:
    - llama-index
    - pymilvus
---

# Build Vector Index

## When to Use

- 用户要求构建/重建索引
- data/raw/ 中新增了文档需要入库
- 索引数据丢失需要恢复

## Steps

1. 确认 `.env` 中 Milvus/Zilliz 配置正确（`MILVUS_URI`, `MILVUS_TOKEN`）
2. 确认 `data/raw/` 下有可索引的文档（PDF/Word/Excel/CSV）
3. 运行 `python main.py --build-index`
4. 检查日志确认索引构建成功

## Important Notes

- 索引构建会遍历 `data/raw/` 下所有支持格式的文件
- 支持的文件格式：`.pdf`, `.docx`, `.doc`, `.xlsx`, `.xls`, `.csv`
- `.doc` 格式仅 Windows 可用（依赖 pywin32）
- 索引写入 Zilliz Cloud，不会覆盖本地 `data/index/` 备份
- 大量文档时 Embedding API 调用可能较慢，注意 API 配额

## Validation

```bash
# 构建后验证：查询一条记录确认索引可用
python main.py --well "测试井" --query "固井质量"
```
