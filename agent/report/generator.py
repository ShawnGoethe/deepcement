"""
报告生成器
基于质量评测结果生成 Markdown 格式的固井质量评测报告
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

from loguru import logger

from config import settings
from core.evaluator import QualityReport, QualityGrade


class ReportGenerator:
    """固井质量评测报告生成器

    输出格式：Markdown（可扩展 HTML/PDF）
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else settings.report_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, report: QualityReport, save: bool = True) -> str:
        """生成评测报告

        Args:
            report: 质量评测报告对象
            save: 是否保存到文件

        Returns:
            Markdown 格式的报告文本
        """
        md = self._render_markdown(report)

        if save:
            filename = self._generate_filename(report)
            filepath = self.output_dir / filename
            filepath.write_text(md, encoding="utf-8")
            logger.info(f"报告已保存: {filepath}")

        return md

    def _render_markdown(self, report: QualityReport) -> str:
        """渲染 Markdown 报告"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        grade_emoji = self._grade_emoji(report.overall_grade)

        parts = [
            f"# 固井质量评测报告",
            f"",
            f"> 生成时间：{now}",
            f"",
            f"## 基本信息",
            f"",
            f"| 项目 | 内容 |",
            f"|------|------|",
            f"| 井名 | {report.well_name} |",
        ]

        # 添加井信息
        if report.well_info:
            info_map = {
                "date": "施工日期",
                "well_depth": "井深(m)",
                "cement_section": "固井井段",
            }
            for key, label in info_map.items():
                if key in report.well_info:
                    parts.append(f"| {label} | {report.well_info[key]} |")

        parts.extend([
            f"",
            f"## 综合评分",
            f"",
            f"**{report.overall_score:.1f} 分** {grade_emoji} **{report.overall_grade.value}**",
            f"",
        ])

        # 评分进度条
        bar = self._score_bar(report.overall_score)
        parts.append(f"`{bar}` {report.overall_score:.0f}/100")
        parts.append("")

        # 各维度评分
        parts.extend([
            f"## 分项评测",
            f"",
            f"| 维度 | 得分 | 等级 | 说明 |",
            f"|------|------|------|------|",
        ])

        for dim in report.dimensions:
            dim_emoji = self._grade_emoji(dim.grade)
            detail_short = dim.details[:50] + "..." if len(dim.details) > 50 else dim.details
            parts.append(
                f"| {dim.name} | {dim.score:.1f} | {dim_emoji} {dim.grade.value} | {detail_short} |"
            )

        parts.append("")

        # 各维度详细信息
        for dim in report.dimensions:
            parts.extend([
                f"### {dim.name}",
                f"",
                f"- **得分**: {dim.score:.1f} 分 ({dim.grade.value})",
                f"- **详细说明**: {dim.details or '无'}",
            ])
            if dim.issues:
                parts.append("- **发现的问题**:")
                for issue in dim.issues:
                    parts.append(f"  - ⚠️ {issue}")
            parts.append("")

        # 结论
        parts.extend([
            f"## 评测结论",
            f"",
            report.conclusion or "暂无结论。",
            f"",
        ])

        # 改进建议
        parts.extend([
            f"## 改进建议",
            f"",
        ])

        if report.suggestions:
            for i, sug in enumerate(report.suggestions, 1):
                parts.append(f"{i}. {sug}")
        else:
            parts.append("暂无改进建议。")

        parts.extend([
            f"",
            f"---",
            f"",
            f"*本报告由 DeepCement 固井质量评测系统自动生成*",
        ])

        return "\n".join(parts)

    def _generate_filename(self, report: QualityReport) -> str:
        """生成报告文件名"""
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        well_name = report.well_name.replace("/", "-").replace("\\", "-")
        return f"report_{well_name}_{date_str}.md"

    @staticmethod
    def _grade_emoji(grade: QualityGrade) -> str:
        """等级对应的 emoji"""
        return {
            QualityGrade.EXCELLENT: "🟢",
            QualityGrade.GOOD: "🔵",
            QualityGrade.QUALIFIED: "🟡",
            QualityGrade.POOR: "🔴",
            QualityGrade.UNKNOWN: "⚪",
        }.get(grade, "⚪")

    @staticmethod
    def _score_bar(score: float, length: int = 20) -> str:
        """生成分数进度条"""
        filled = int(score / 100 * length)
        return "█" * filled + "░" * (length - filled)
