"""
测试用例：加载 data 目录下所有 PDF，构建索引并查询

用法：
    # 直接运行
    python tests/test_pdf_query.py

    # 指定查询
    python tests/test_pdf_query.py --query "XX-1井固井质量如何"

    # 指定井名评测
    python tests/test_pdf_query.py --well "XX-1"
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | {message}")


def list_pdfs(data_dir: Path) -> list:
    """列出 data 目录下所有 PDF 文件"""
    pdfs = list(data_dir.rglob("*.pdf"))
    logger.info(f"在 {data_dir} 下找到 {len(pdfs)} 个 PDF 文件:")
    for pdf in pdfs:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        logger.info(f"  - {pdf.name} ({size_mb:.2f} MB)")
    return pdfs


def test_ingest_pdfs():
    """测试1：解析所有 PDF 文件"""
    from config import settings
    from core.ingester import DocumentIngester

    logger.info("=" * 50)
    logger.info("测试1：解析 PDF 文件")
    logger.info("=" * 50)

    pdfs = list_pdfs(settings.raw_dir)
    if not pdfs:
        logger.warning("data/raw/ 目录下没有 PDF 文件，请先放入固井资料")
        return []

    ingester = DocumentIngester(settings.raw_dir)
    documents = []

    for pdf_path in pdfs:
        logger.info(f"\n解析: {pdf_path.name}")
        docs = ingester.ingest_file(pdf_path)
        documents.extend(docs)

        for doc in docs:
            logger.info(f"  类型: {doc.doc_type}")
            logger.info(f"  元数据: {doc.metadata}")
            preview = doc.content[:200].replace("\n", " ")
            logger.info(f"  内容预览: {preview}...")

    logger.info(f"\n共解析 {len(documents)} 个文档片段")
    return documents


def test_build_index(documents):
    """测试2：构建向量索引"""
    from core.indexer import IndexManager

    logger.info("\n" + "=" * 50)
    logger.info("测试2：构建向量索引")
    logger.info("=" * 50)

    indexer = IndexManager()
    indexer.build_index(documents)
    indexer.save_index()

    logger.info("索引构建并保存完成")
    return indexer


def test_search(indexer, query: str = "固井质量 水泥浆 密度"):
    """测试3：语义检索"""
    from core.retriever import HistoryRetriever

    logger.info("\n" + "=" * 50)
    logger.info(f"测试3：语义检索")
    logger.info(f"查询: {query}")
    logger.info("=" * 50)

    retriever = HistoryRetriever(indexer)
    results = retriever.search(query=query, top_k=5)

    if not results:
        logger.warning("未检索到相关结果")
        return results

    for i, r in enumerate(results, 1):
        logger.info(f"\n--- 结果{i} (score: {r.score:.4f}) ---")
        logger.info(f"来源: {r.source}")
        logger.info(f"元数据: {r.metadata}")
        preview = r.content[:300].replace("\n", " ")
        logger.info(f"内容: {preview}...")

    return results


def test_llm_query(indexer, query: str = "这份固井资料的主要内容是什么？"):
    """测试4：LLM 问答"""
    from core.retriever import HistoryRetriever

    logger.info("\n" + "=" * 50)
    logger.info(f"测试4：LLM 问答")
    logger.info(f"问题: {query}")
    logger.info("=" * 50)

    retriever = HistoryRetriever(indexer)
    answer = retriever.query(query)

    logger.info(f"\n回答:\n{answer}")
    return answer


def test_evaluate(indexer, well_name: str):
    """测试5：质量评测"""
    from core.retriever import HistoryRetriever
    from core.evaluator import QualityEvaluator
    from agent.report.generator import ReportGenerator

    logger.info("\n" + "=" * 50)
    logger.info(f"测试5：质量评测 - {well_name}")
    logger.info("=" * 50)

    retriever = HistoryRetriever(indexer)
    evaluator = QualityEvaluator(retriever)

    report = evaluator.evaluate(well_name=well_name)

    # 打印评测结果
    logger.info(f"\n综合得分: {report.overall_score:.1f} ({report.overall_grade.value})")
    for dim in report.dimensions:
        logger.info(f"  {dim.name}: {dim.score:.1f} ({dim.grade.value})")
    logger.info(f"\n结论: {report.conclusion}")
    logger.info(f"\n建议:")
    for i, sug in enumerate(report.suggestions, 1):
        logger.info(f"  {i}. {sug}")

    # 生成报告文件
    generator = ReportGenerator()
    md = generator.generate(report, save=True)
    logger.info(f"\n报告已保存")

    return report


def main():
    parser = argparse.ArgumentParser(description="PDF 加载与查询测试")
    parser.add_argument("--query", type=str, default="固井质量 水泥浆 密度 施工", help="检索查询文本")
    parser.add_argument("--well", type=str, default=None, help="井名（用于质量评测）")
    parser.add_argument("--skip-eval", action="store_true", help="跳过质量评测（节省 API 调用）")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    logger.info("DeepCement PDF 加载与查询测试")
    logger.info("=" * 50)

    # Step 1: 解析 PDF
    documents = test_ingest_pdfs()
    if not documents:
        logger.error("没有文档可处理，退出")
        return

    # Step 2: 构建索引
    indexer = test_build_index(documents)

    # Step 3: 语义检索
    test_search(indexer, query=args.query)

    # Step 4: LLM 问答
    test_llm_query(indexer)

    # Step 5: 质量评测（可选）
    if not args.skip_eval:
        well_name = args.well
        if not well_name:
            # 从文档元数据中提取第一个井名
            for doc in documents:
                if doc.metadata.get("well_name"):
                    well_name = doc.metadata["well_name"]
                    break

        if well_name:
            test_evaluate(indexer, well_name)
        else:
            logger.warning("未指定井名且文档中未提取到井名，跳过质量评测")
            logger.info("提示: 使用 --well '井名' 指定评测井")

    logger.info("\n" + "=" * 50)
    logger.info("全部测试完成！")


if __name__ == "__main__":
    main()
