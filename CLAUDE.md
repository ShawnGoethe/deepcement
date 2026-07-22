# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepCement is a cementing quality evaluation system for oil wells, built on DeepAgent + LlamaIndex. It ingests cementing documents (PDF/Excel/CSV), builds vector indexes, performs semantic retrieval, and generates quality evaluation reports with scores across 4 dimensions.

**Language**: Python 3.9+ (Chinese comments and documentation throughout)

## Core Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build vector index from documents in data/raw/
python main.py --build-index

# Run single evaluation
python main.py --well "XX-1" --query "生成固井质量评测报告"

# Interactive mode (recommended for development)
python main.py --interactive

# Run tests
python tests/test_basic.py

# Code formatting (格式化代码)
black .                    # 格式化
isort .                    # 排序 import
ruff check . --fix         # lint 检查并修复
```

## Architecture

The system follows a pipeline architecture:

```
Documents (data/raw/) → Ingester → Indexer → Retriever → Evaluator → Report
                                    ↓
                              LlamaIndex VectorStoreIndex (persisted to data/index/)
```

### Core Modules (`core/`)

- **`ingester.py`** — Parses PDF/Excel/CSV into `CementDocument` objects with extracted metadata (well name, date, depth). Uses PyMuPDF for PDF, pandas for Excel/CSV.
- **`indexer.py`** — `IndexManager` wraps LlamaIndex `VectorStoreIndex`. Handles build/save/load with lazy initialization of LLM and embedding models. Uses `OpenAILike` wrappers for API-compatible models.
- **`retriever.py`** — `HistoryRetriever` provides semantic search with metadata filtering (`search()`, `query()`, `search_by_well()`).
- **`evaluator.py`** — `QualityEvaluator` runs 4-dimension evaluation (slurry, operation, effect, anomaly). Each dimension is independently scored by LLM. Overall score = dimension average.

### Agent Layer (`agent/`)

- **`orchestrator.py`** — `CementAgent` orchestrates the full flow. Uses DeepAgent when available, falls back to rule-based matching when not installed.
- **`tools.py`** — Agent tools: `search_history`, `evaluate_quality`, `compare_data`.
- **`report/`** — Markdown report generation with Jinja2 templates.

## Configuration

All config via environment variables (`.env` file, see `.env.example`):

- `LLM_*` — LLM settings (DeepSeek/Qwen compatible OpenAI API)
- `EMBED_*` — Embedding model settings
- `DATA_*` — Path overrides for raw/index/report directories

Config classes use pydantic-settings with `env_prefix` for automatic env loading.

## Key Patterns

1. **Lazy initialization**: LLM and embedding models are initialized on first use, not at import time. This keeps imports fast and allows config to load before model init.

2. **Fallback mode**: `CementAgent._fallback_run()` handles queries without DeepAgent by keyword matching ("评测" → evaluate, "对比" → compare, else → search).

3. **Document metadata**: `CementDocument.metadata` is extracted via regex from document text (well name patterns like `XX-1`, dates, depths). Metadata is used for filtering in retriever.

4. **Scoring system**: 4 dimensions (水泥浆性能, 施工参数, 固井效果, 异常情况), each 0-100. Grades: ≥90 优秀, ≥75 良好, ≥60 合格, <60 不合格.

## Testing

Tests in `tests/test_basic.py` verify config loading, ingester initialization, document structure, grade calculations, and report generation. Run directly with `python tests/test_basic.py`.

## Directory Structure

- `data/raw/` — Input documents (PDF/Excel/CSV)
- `data/index/` — Persisted LlamaIndex vector store
- `agent/report/output/` — Generated Markdown reports
- `logs/` — Application logs (loguru with rotation)
