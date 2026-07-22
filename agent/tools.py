"""
Agent 工具定义
为 DeepAgent 提供固井质量评测相关的工具函数
"""

from typing import Optional

from langchain_core.tools import tool

from config import settings
from core.retriever import HistoryRetriever
from core.evaluator import QualityEvaluator


def create_cement_tools(
    retriever: HistoryRetriever,
    evaluator: QualityEvaluator,
) -> list:
    """创建固井评测工具集

    Returns:
        工具列表（LangChain @tool 装饰的函数）
    """

    @tool
    def search_history(
        query: str,
        well_name: Optional[str] = None,
        top_k: int = settings.retriever.top_k,
    ) -> str:
        """检索固井历史资料

        Args:
            query: 查询内容（自然语言描述）
            well_name: 井名过滤（可选）
            top_k: 返回结果数量，默认从配置读取
        """
        results = retriever.search(
            query=query,
            top_k=top_k,
            well_name=well_name,
        )

        if not results:
            return "未找到相关固井资料。"

        output_parts = []
        for i, r in enumerate(results, 1):
            meta_info = f"来源: {r.source}"
            if r.metadata.get("well_name"):
                meta_info += f" | 井名: {r.metadata['well_name']}"
            if r.metadata.get("date"):
                meta_info += f" | 日期: {r.metadata['date']}"
            output_parts.append(
                f"【结果{i}】{meta_info}\n{r.content[:500]}..."
            )

        return "\n\n".join(output_parts)

    @tool
    def evaluate_quality(
        well_name: str,
        query: Optional[str] = None,
    ) -> str:
        """对指定井进行固井质量评测

        Args:
            well_name: 井名（必填）
            query: 额外查询条件（可选）
        """
        report = evaluator.evaluate(
            well_name=well_name,
            query=query,
        )

        # 格式化输出
        parts = [
            f"=== 固井质量评测报告: {report.well_name} ===",
            f"综合得分: {report.overall_score:.1f}分 ({report.overall_grade.value})",
            "",
            "--- 各维度评分 ---",
        ]

        for dim in report.dimensions:
            parts.append(f"  {dim.name}: {dim.score:.1f}分 ({dim.grade.value})")
            if dim.details:
                parts.append(f"    说明: {dim.details}")
            if dim.issues:
                parts.append(f"    问题: {'; '.join(dim.issues)}")

        parts.extend([
            "",
            "--- 结论 ---",
            report.conclusion,
            "",
            "--- 改进建议 ---",
        ])

        for i, sug in enumerate(report.suggestions, 1):
            parts.append(f"  {i}. {sug}")

        return "\n".join(parts)

    @tool
    def compare_data(
        well_name: str,
        target_well: str,
    ) -> str:
        """对比两口井的固井数据

        Args:
            well_name: 当前井名
            target_well: 对比井名
        """
        # 检索两口井的资料
        results_a = retriever.search_by_well(well_name, top_k=settings.retriever.top_k)
        results_b = retriever.search_by_well(target_well, top_k=settings.retriever.top_k)

        if not results_a:
            return f"未找到 {well_name} 的固井资料。"
        if not results_b:
            return f"未找到 {target_well} 的固井资料。"

        output = f"=== 固井数据对比: {well_name} vs {target_well} ===\n\n"

        output += f"【{well_name} 相关资料】\n"
        for r in results_a:
            output += f"- {r.source}: {r.content[:200]}...\n"

        output += f"\n【{target_well} 相关资料】\n"
        for r in results_b:
            output += f"- {r.source}: {r.content[:200]}...\n"

        output += "\n注：详细对比分析请使用 evaluate_quality 分别评测后参考。"
        return output

    return [search_history, evaluate_quality, compare_data]
