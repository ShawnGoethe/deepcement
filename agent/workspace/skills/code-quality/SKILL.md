---
name: code-quality
description: 运行代码格式化和 lint 检查（black, isort, ruff）
metadata:
  author: deepcement
  category: dev
  requires:
    - black
    - isort
    - ruff
---

# Code Quality Check

## When to Use

- 用户要求格式化代码
- 用户要求检查代码质量
- 提交代码前的例行检查

## Steps

1. 格式化：`black .`
2. 排序 import：`isort .`
3. Lint 检查并修复：`ruff check . --fix`
4. 如有剩余警告，手动修复

## Configuration

工具配置在 `pyproject.toml` 中（如有）。默认配置：

- **black**: 行宽 88，Python 3.9+
- **isort**: profile = "black"
- **ruff**: 继承默认规则集

## Important Notes

- `black` 和 `isort` 是自动修复工具，会直接修改文件
- `ruff check --fix` 只修复安全的规则，部分警告需手动处理
- 代码中大量中文注释，确保编辑器使用 UTF-8 编码
- Windows 环境下注意换行符问题（git 配置 `core.autocrlf`）
