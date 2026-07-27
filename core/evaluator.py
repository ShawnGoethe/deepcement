"""
质量评测模块
基于检索结果 + LLM 生成固井质量评测
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from config import settings
from core.retriever import HistoryRetriever, RetrievalResult


class QualityGrade(str, Enum):
    """质量等级"""

    EXCELLENT = "优秀"
    GOOD = "良好"
    QUALIFIED = "合格"
    POOR = "不合格"
    UNKNOWN = "待定"


@dataclass
class QualityDimension:
    """质量评测维度"""

    name: str  # 维度名称
    score: float = 0.0  # 得分（0-100）
    grade: QualityGrade = QualityGrade.UNKNOWN
    details: str = ""  # 详细说明
    issues: List[str] = field(default_factory=list)  # 发现的问题


@dataclass
class QualityReport:
    """质量评测报告"""

    well_name: str = ""  # 井名
    well_info: Dict[str, Any] = field(default_factory=dict)  # 基本信息
    overall_score: float = 0.0  # 综合得分
    overall_grade: QualityGrade = QualityGrade.UNKNOWN
    dimensions: List[QualityDimension] = field(default_factory=list)
    conclusion: str = ""  # 结论
    suggestions: List[str] = field(default_factory=list)  # 改进建议
    raw_data: List[RetrievalResult] = field(default_factory=list)  # 原始检索数据


class QualityEvaluator:
    """固井质量评测器

    评测维度：
    1. 水泥浆性能（密度、失水量、稠化时间）
    2. 施工参数（泵速、压力、替量）
    3. 固井效果（返高、候凝、胶结质量）
    4. 异常情况（漏失、窜槽、气侵）
    """

    # 评测维度定义
    DIMENSIONS = {
        "slurry": {
            "name": "水泥浆性能",
            "keywords": ["水泥浆", "密度", "失水", "稠化", "浆体", "配方"],
            "prompt": """请根据以下固井资料，评测水泥浆性能质量。
评估要点：
- 水泥浆密度是否在合理范围（通常1.85-1.95 g/cm³）
- 失水量是否达标（API标准通常<50ml）
- 稠化时间是否合理（施工安全时间）
- 浆体配方是否规范

请给出：得分(0-100)、等级(优秀/良好/合格/不合格)、详细说明、发现的问题。""",
        },
        "operation": {
            "name": "施工参数",
            "keywords": ["泵速", "压力", "替量", "排量", "施工", "注入"],
            "prompt": """请根据以下固井资料，评测施工参数质量。
评估要点：
- 泵速/排量是否平稳
- 施工压力是否在安全范围
- 替浆量计算是否准确
- 施工过程是否连续

请给出：得分(0-100)、等级(优秀/良好/合格/不合格)、详细说明、发现的问题。""",
        },
        "effect": {
            "name": "固井效果",
            "keywords": ["返高", "候凝", "胶结", "声幅", "固井质量", "CBL"],
            "prompt": """请根据以下固井资料，评测固井效果。
评估要点：
- 水泥返高是否达到设计要求
- 候凝时间是否充足
- 声幅测井(CBL)结果
- 胶结质量评价

请给出：得分(0-100)、等级(优秀/良好/合格/不合格)、详细说明、发现的问题。""",
        },
        "anomaly": {
            "name": "异常情况",
            "keywords": ["漏失", "窜槽", "气侵", "异常", "事故", "复杂"],
            "prompt": """请根据以下固井资料，评估异常情况。
评估要点：
- 是否发生漏失
- 是否存在窜槽风险
- 是否有气侵现象
- 其他异常及处理情况

请给出：得分(0-100)、等级(优秀/良好/合格/不合格)、详细说明、发现的问题。""",
        },
    }

    def __init__(self, retriever: HistoryRetriever):
        self.retriever = retriever
        self._llm = None

    def _init_llm(self):
        """初始化 LLM"""
        if self._llm is not None:
            return

        from llama_index.llms.openai_like import OpenAILike

        self._llm = OpenAILike(
            model=settings.llm.model,
            api_base=settings.llm.base_url,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

    def evaluate(
        self,
        well_name: str,
        query: Optional[str] = None,
        top_k: int = settings.retriever.top_k,
    ) -> QualityReport:
        """执行完整的质量评测

        Args:
            well_name: 井名
            query: 额外查询条件（可选）
            top_k: 检索结果数量（默认从配置读取）

        Returns:
            QualityReport 评测报告
        """
        logger.info(f"开始评测: {well_name}")

        # 1. 检索相关资料
        search_query = query or f"{well_name} 固井 质量 评测"
        raw_data = self.retriever.search(
            query=search_query,
            top_k=top_k,
            well_name=well_name,
        )

        if not raw_data:
            logger.warning(f"未找到 {well_name} 的相关资料")
            return QualityReport(
                well_name=well_name,
                conclusion="未找到该井的固井资料，无法进行评测。",
            )

        # 2. 构建上下文
        context = self._build_context(raw_data)

        # 3. 逐维度评测
        dimensions = []
        for dim_key, dim_def in self.DIMENSIONS.items():
            dim_result = self._evaluate_dimension(dim_key, dim_def, context)
            dimensions.append(dim_result)
            logger.info(f"  {dim_def['name']}: {dim_result.score}分 ({dim_result.grade.value})")

        # 4. 计算综合得分
        overall_score = sum(d.score for d in dimensions) / len(dimensions) if dimensions else 0
        overall_grade = self._score_to_grade(overall_score)

        # 5. 生成结论和建议
        conclusion = self._generate_conclusion(well_name, context, dimensions)
        suggestions = self._generate_suggestions(well_name, context, dimensions)

        # 6. 提取井基本信息
        well_info = self._extract_well_info(raw_data)

        report = QualityReport(
            well_name=well_name,
            well_info=well_info,
            overall_score=overall_score,
            overall_grade=overall_grade,
            dimensions=dimensions,
            conclusion=conclusion,
            suggestions=suggestions,
            raw_data=raw_data,
        )

        logger.info(f"评测完成: {well_name} → {overall_score:.1f}分 ({overall_grade.value})")
        return report

    def _build_context(self, data: List[RetrievalResult]) -> str:
        """将检索结果构建为 LLM 上下文"""
        parts = []
        for i, item in enumerate(data, 1):
            parts.append(f"【资料{i}】(来源: {item.source})\n{item.content}")
        return "\n\n".join(parts)

    def _extract_response_text(self, response) -> str:
        """从 LLM ChatResponse 中提取文本内容"""
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content
        return str(response)

    def _evaluate_dimension(
        self,
        dim_key: str,
        dim_def: dict,
        context: str,
    ) -> QualityDimension:
        """评测单个维度"""
        self._init_llm()

        prompt = f"""你是一位固井质量评测专家。请根据以下固井资料，评测【{dim_def['name']}】维度的质量。

{dim_def['prompt']}

===== 固井资料 =====
{context}
===== 资料结束 =====

请按以下格式输出：
得分：[0-100的数字]
等级：[优秀/良好/合格/不合格]
说明：[详细说明]
问题：[发现的问题，用 | 分隔多个问题，无问题则写"无"]
"""

        try:
            from llama_index.core import Settings

            Settings.llm = self._llm
            response = self._llm.chat(prompt)
            result_text = self._extract_response_text(response)
            logger.debug(f"维度 {dim_key} LLM 返回: {result_text[:200]}...")
        except Exception as e:
            logger.error(f"维度 {dim_key} 评测失败: {e}")
            return QualityDimension(name=dim_def["name"])

        # 解析结果
        return self._parse_dimension_result(dim_def["name"], result_text)

    def _parse_dimension_result(self, name: str, text: str) -> QualityDimension:
        """解析 LLM 输出的维度评测结果"""
        import re

        dim = QualityDimension(name=name)

        # 提取得分
        score_match = re.search(r"得分[：:]\s*(\d+(?:\.\d+)?)", text)
        if score_match:
            dim.score = float(score_match.group(1))

        # 提取等级
        grade_match = re.search(r"等级[：:]\s*(优秀|良好|合格|不合格)", text)
        if grade_match:
            dim.grade = QualityGrade(grade_match.group(1))
        else:
            dim.grade = self._score_to_grade(dim.score)

        # 提取说明
        detail_match = re.search(r"说明[：:]\s*(.+?)(?=问题[：:]|$)", text, re.DOTALL)
        if detail_match:
            dim.details = detail_match.group(1).strip()

        # 提取问题
        issue_match = re.search(r"问题[：:]\s*(.+?)$", text, re.DOTALL)
        if issue_match:
            issue_text = issue_match.group(1).strip()
            if issue_text and issue_text != "无":
                dim.issues = [i.strip() for i in issue_text.split("|") if i.strip()]

        return dim

    def _generate_conclusion(
        self,
        well_name: str,
        context: str,
        dimensions: List[QualityDimension],
    ) -> str:
        """生成综合结论"""
        self._init_llm()

        dim_summary = "\n".join(
            [f"- {d.name}: {d.score}分({d.grade.value})。{d.details}" for d in dimensions]
        )

        prompt = f"""你是一位固井质量评测专家。请根据以下评测结果，为井【{well_name}】撰写固井质量综合结论。

===== 各维度评测结果 =====
{dim_summary}
===== 评测结束 =====

要求：
1. 总结固井质量整体情况
2. 指出主要优点和不足
3. 结论简洁明了（200字以内）

请直接输出结论文本，不要加标题或前缀。
"""

        try:
            from llama_index.core import Settings

            Settings.llm = self._llm
            response = self._llm.chat(prompt)
            text = self._extract_response_text(response)
            return text.strip()
        except Exception as e:
            logger.error(f"生成结论失败: {e}")
            return "结论生成失败，请查看各维度详细评测结果。"

    def _generate_suggestions(
        self,
        well_name: str,
        context: str,
        dimensions: List[QualityDimension],
    ) -> List[str]:
        """生成改进建议"""
        self._init_llm()

        # 收集所有问题
        all_issues = []
        for d in dimensions:
            if d.issues:
                all_issues.extend(d.issues)
        issues_text = "\n".join([f"- {i}" for i in all_issues]) if all_issues else "无明显问题"

        prompt = f"""你是一位固井质量评测专家。请根据以下评测结果，为井【{well_name}】提出改进建议。

===== 发现的问题 =====
{issues_text}
===== 问题结束 =====

要求：
1. 针对每个问题提出具体可行的改进建议
2. 建议应具有可操作性
3. 每条建议一行，以序号开头（如"1. xxx"）
4. 提出3-5条建议

请直接输出建议列表，不要加标题或前缀。
"""

        try:
            from llama_index.core import Settings

            Settings.llm = self._llm
            response = self._llm.chat(prompt)
            text = self._extract_response_text(response)
            text = text.strip()

            # 解析建议列表
            suggestions = []
            for line in text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                    # 去掉序号前缀
                    import re

                    cleaned = re.sub(r"^[\d.]+\s*", "", line).strip()
                    if cleaned:
                        suggestions.append(cleaned)

            return suggestions if suggestions else [text]
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            return ["建议生成失败，请查看各维度详细评测结果。"]

    def _extract_well_info(self, data: List[RetrievalResult]) -> Dict[str, Any]:
        """从检索结果中提取井基本信息"""
        info = {}
        for item in data:
            meta = item.metadata
            for key in ["well_name", "date", "well_depth", "cement_section"]:
                if key in meta and key not in info:
                    info[key] = meta[key]
        return info

    @staticmethod
    def _score_to_grade(score: float) -> QualityGrade:
        """根据分数确定等级"""
        if score >= 90:
            return QualityGrade.EXCELLENT
        elif score >= 75:
            return QualityGrade.GOOD
        elif score >= 60:
            return QualityGrade.QUALIFIED
        else:
            return QualityGrade.POOR
