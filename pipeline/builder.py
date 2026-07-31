"""
索引构建流水线
独立于 Agent 的数据处理流程：文档解析 → 结构化抽取 → 知识图谱 → 向量索引
"""

from pathlib import Path

from loguru import logger

from config import BASE_DIR, settings
from core.indexer import IndexManager
from core.ingester import DocumentIngester
from pipeline.extractor import CementDataExtractor
from pipeline.graph_builder import GraphBuilder


def build_index():
    """全量构建索引

    流程：
    1. 解析文档（PDF/Word/Excel/CSV）→ CementDocument
    2. LLM 抽取结构化数据 → SQLite + 知识图谱
    3. 向量索引 → Zilliz Cloud

    Returns:
        True 如果构建成功
    """
    logger.info("开始构建索引...")

    # 1. 解析文档
    ingester = DocumentIngester(settings.raw_dir)
    documents = ingester.ingest_all()
    if not documents:
        logger.warning("没有找到可索引的文档")
        return False

    # 2. 结构化抽取 + 知识图谱
    if settings.graph.enabled:
        _extract_and_build_graph(documents)

    # 3. 向量索引
    indexer = IndexManager()
    indexer.build_index(documents)
    logger.info("索引构建完成（已写入 Zilliz Cloud）")
    return True


def _extract_and_build_graph(documents: list):
    """结构化抽取 → SQLite + 知识图谱"""
    try:
        logger.info("开始抽取结构化数据...")
        extractor = CementDataExtractor()
        triples = extractor.extract_all(documents)

        builder = GraphBuilder(
            sqlite_path=str(BASE_DIR / settings.graph.sqlite_path),
            graph_dir=str(BASE_DIR / settings.graph.graph_dir),
        )
        builder.build(triples)
        builder.save()
        logger.info(f"结构化数据抽取完成: {len(triples)} 个三元组")
    except Exception as e:
        logger.error(f"结构化数据抽取失败（不影响向量索引）: {e}")
