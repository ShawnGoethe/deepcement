"""
Agent 工具定义
为 DeepAgent 提供固井质量评测相关的工具函数
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from loguru import logger

from config import settings
from core.evaluator import QualityEvaluator
from core.retriever import HistoryRetriever


def create_cement_tools(
    retriever: HistoryRetriever,
    evaluator: QualityEvaluator,
) -> list:
    """创建固井评测工具集

    Args:
        retriever: 历史资料检索器
        evaluator: LLM 评估器

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
            output_parts.append(f"【结果{i}】{meta_info}\n{r.content[:500]}...")

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

        parts.extend(
            [
                "",
                "--- 结论 ---",
                report.conclusion,
                "",
                "--- 改进建议 ---",
            ]
        )

        for i, sug in enumerate(report.suggestions, 1):
            parts.append(f"  {i}. {sug}")

        return "\n".join(parts)

    @tool
    def archive_file(file_path: str) -> str:
        """将已处理的文件归档到 data/raw/loaded/

        将 data/raw/ 下的文件移动到 data/raw/loaded/，标记为已处理。
        不会真正删除文件，可随时从 loaded/ 目录恢复。

        Args:
            file_path: 要归档的文件名或路径（相对于 data/raw/ 或绝对路径）
        """
        target = Path(file_path)

        # 如果是相对路径，在 data/raw 下查找
        if not target.is_absolute():
            target = settings.raw_dir / target

        # 安全检查：只允许移动 data/raw 目录下的文件
        try:
            target = target.resolve()
            raw_dir_resolved = settings.raw_dir.resolve()
        except Exception:
            return f"路径解析失败: {file_path}"

        if not str(target).startswith(str(raw_dir_resolved)):
            return f"安全拒绝：只能归档 data/raw/ 目录下的文件，不允许操作 {file_path}"

        if not target.exists():
            return f"文件不存在: {target}"

        if target.is_dir():
            return f"安全拒绝：不支持归档目录，请指定具体文件"

        # 创建 loaded 目录
        loaded_dir = settings.raw_dir / "loaded"
        loaded_dir.mkdir(parents=True, exist_ok=True)

        # 目标路径（处理重名）
        dest = loaded_dir / target.name
        if dest.exists():
            stem = target.stem
            suffix = target.suffix
            counter = 1
            while dest.exists():
                dest = loaded_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        # 移动文件
        try:
            import shutil
            shutil.move(str(target), str(dest))
            file_size = dest.stat().st_size
            size_str = (
                f"{file_size / 1024:.1f} KB"
                if file_size < 1024 * 1024
                else f"{file_size / (1024 * 1024):.1f} MB"
            )
            logger.info(f"文件已归档: {target} → {dest}")
            return (
                f"✅ 文件已归档\n"
                f"  原路径: {target}\n"
                f"  新路径: {dest}\n"
                f"  大小: {size_str}"
            )
        except Exception as e:
            logger.error(f"文件归档失败: {e}")
            return f"❌ 文件归档失败: {e}"

    return [
        search_history,
        evaluate_quality,
        archive_file,
    ]
