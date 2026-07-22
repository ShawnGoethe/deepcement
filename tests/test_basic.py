"""
基础测试
"""

import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config():
    """测试配置加载"""
    from config import settings

    assert settings.llm is not None
    assert settings.embed is not None
    assert settings.paths is not None
    print("✓ 配置加载正常")


def test_ingester_init():
    """测试摄入器初始化"""
    from core.ingester import DocumentIngester
    from config import settings

    ingester = DocumentIngester(settings.raw_dir)
    assert ingester.raw_dir.exists()
    print("✓ 摄入器初始化正常")


def test_cement_document():
    """测试文档结构"""
    from core.ingester import CementDocument

    doc = CementDocument(
        content="测试内容",
        source="test.pdf",
        doc_type="report",
        metadata={"well_name": "XX-1"},
    )
    assert doc.content == "测试内容"
    assert doc.metadata["well_name"] == "XX-1"
    print("✓ 文档结构正常")


def test_evaluator_grades():
    """测试评分等级"""
    from core.evaluator import QualityEvaluator, QualityGrade

    assert QualityEvaluator._score_to_grade(95) == QualityGrade.EXCELLENT
    assert QualityEvaluator._score_to_grade(80) == QualityGrade.GOOD
    assert QualityEvaluator._score_to_grade(65) == QualityGrade.QUALIFIED
    assert QualityEvaluator._score_to_grade(50) == QualityGrade.POOR
    print("✓ 评分等级正常")


def test_report_generator():
    """测试报告生成器"""
    from agent.report.generator import ReportGenerator
    from core.evaluator import (
        QualityReport, QualityDimension, QualityGrade,
    )

    report = QualityReport(
        well_name="测试井-1",
        overall_score=85.0,
        overall_grade=QualityGrade.GOOD,
        dimensions=[
            QualityDimension(
                name="水泥浆性能",
                score=90.0,
                grade=QualityGrade.EXCELLENT,
                details="密度合格，失水量达标",
            ),
            QualityDimension(
                name="施工参数",
                score=80.0,
                grade=QualityGrade.GOOD,
                details="泵速平稳，压力正常",
            ),
        ],
        conclusion="固井质量良好，各项指标基本达标。",
        suggestions=[
            "建议优化水泥浆配方，降低失水量",
            "加强施工过程监控",
        ],
    )

    generator = ReportGenerator()
    md = generator.generate(report, save=False)

    assert "测试井-1" in md
    assert "85.0" in md
    assert "水泥浆性能" in md
    assert "改进建议" in md
    print("✓ 报告生成正常")
    print(f"\n--- 报告预览 ---\n{md[:500]}...")


if __name__ == "__main__":
    print("=== DeepCement 基础测试 ===\n")

    test_config()
    test_ingester_init()
    test_cement_document()
    test_evaluator_grades()
    test_report_generator()

    print("\n=== 全部测试通过 ===")
