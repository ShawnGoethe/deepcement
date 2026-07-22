@echo off
REM 代码格式化脚本 (Windows)

echo === 运行 isort (import 排序) ===
isort .

echo === 运行 black (代码格式化) ===
black .

echo === 运行 ruff (lint 检查) ===
ruff check . --fix

echo === 格式化完成 ===
pause
