"""
分层评估协调器
实现 规则引擎 → XGBoost → LLM 兜底 的分层评估
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from core.las_parser import LasParser, LasData
from core.rule_engine import RuleEngine, RuleResult
from core.ml_models import XGBoostEvaluator, MLResult


@dataclass
class CascadeResult:
    """分层评估结果"""
    well_name: str = ""
    overall_score: float = 0.0
    overall_grade: str = "待定"
    confidence: float = 0.0
    eval_layer: str = ""                        # 最终使用的评估层

    # 各层结果
    rule_result: Optional[RuleResult] = None
    xgboost_result: Optional[MLResult] = None
    llm_result: Optional[Any] = None

    # 汇总信息
    summary: str = ""
    details: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    # 原始数据
    depth_range: tuple = (0.0, 0.0)
    curves_used: List[str] = field(default_factory=list)


class CascadeEvaluator:
    """分层评估协调器

    评估流程：
    1. 规则引擎（置信度 > 0.85 → 直接返回）
    2. XGBoost（置信度 > 0.75 → 直接返回）
    3. LLM 兜底（以上都低置信度时调用）

    设计原则：
    - 快速路径：规则引擎对标准数据置信度高，毫秒级返回
    - 精度提升：XGBoost 处理复杂模式
    - 兜底保障：LLM 处理边界情况和异常数据
    """

    def __init__(
        self,
        rules_path: Optional[str] = None,
        xgboost_path: Optional[str] = None,
        llm_evaluator=None,
        retriever=None,
    ):
        """初始化分层评估器

        Args:
            rules_path: 规则配置文件路径
            xgboost_path: XGBoost 模型路径
            llm_evaluator: LLM 评估器实例（可选）
            retriever: 检索器实例（可选，用于复用检索结果）
        """
        self.las_parser = LasParser()
        self.rule_engine = RuleEngine(rules_path)
        self.xgboost = XGBoostEvaluator(xgboost_path)
        self.llm_evaluator = llm_evaluator
        self.retriever = retriever

        # 置信度阈值
        self.thresholds = {
            "rule_engine": 0.85,
            "xgboost": 0.75,
        }

        logger.info("分层评估器初始化完成")

    def evaluate_las_file(self, file_path: str) -> CascadeResult:
        """评估 LAS 文件

        Args:
            file_path: LAS 文件路径

        Returns:
            CascadeResult 评估结果
        """
        logger.info(f"开始分层评估: {file_path}")

        # 1. 解析 LAS 文件
        try:
            las_data = self.las_parser.parse(file_path)
        except Exception as e:
            logger.error(f"LAS 文件解析失败: {e}")
            return CascadeResult(
                well_name=Path(file_path).stem,
                summary=f"LAS 文件解析失败: {e}",
            )

        return self.evaluate(las_data)

    def evaluate(self, las_data: LasData) -> CascadeResult:
        """评估 LAS 数据

        Args:
            las_data: LAS 解析数据

        Returns:
            CascadeResult 评估结果
        """
        logger.info(f"开始分层评估: {las_data.well_name}")

        result = CascadeResult(
            well_name=las_data.well_name,
            depth_range=(las_data.depth_start, las_data.depth_end),
            curves_used=list(las_data.curves.keys()),
        )

        # ========== Layer 1: 规则引擎 ==========
        logger.info("Layer 1: 规则引擎评估...")
        try:
            rule_result = self.rule_engine.evaluate(las_data)
            result.rule_result = rule_result

            if rule_result.overall_confidence >= self.thresholds["rule_engine"]:
                logger.info(f"规则引擎置信度 {rule_result.overall_confidence:.2f} >= {self.thresholds['rule_engine']}，直接返回")
                return self._build_result_from_rule(result, rule_result)
            else:
                logger.info(f"规则引擎置信度 {rule_result.overall_confidence:.2f} < {self.thresholds['rule_engine']}，继续下一层")
        except Exception as e:
            logger.error(f"规则引擎评估失败: {e}")

        # ========== Layer 2: XGBoost ==========
        logger.info("Layer 2: XGBoost 评估...")
        try:
            xgb_result = self.xgboost.predict(las_data.curves)
            result.xgboost_result = xgb_result

            if xgb_result.confidence >= self.thresholds["xgboost"]:
                logger.info(f"XGBoost 置信度 {xgb_result.confidence:.2f} >= {self.thresholds['xgboost']}，直接返回")
                return self._build_result_from_ml(result, xgb_result, "xgboost")
            else:
                logger.info(f"XGBoost 置信度 {xgb_result.confidence:.2f} < {self.thresholds['xgboost']}，继续 LLM 兜底")
        except Exception as e:
            logger.error(f"XGBoost 评估失败: {e}")

        # ========== Layer 3: LLM 兜底 ==========
        logger.info("Layer 3: LLM 兜底评估...")
        if self.llm_evaluator is not None:
            try:
                llm_result = self.llm_evaluator.evaluate(well_name=las_data.well_name)
                result.llm_result = llm_result
                return self._build_result_from_llm(result, llm_result)
            except Exception as e:
                logger.error(f"LLM 评估失败: {e}")

        # 所有层都失败，返回综合结果
        return self._build_fallback_result(result)

    def _build_result_from_rule(self, result: CascadeResult, rule_result: RuleResult) -> CascadeResult:
        """从规则引擎结果构建最终结果"""
        result.overall_score = rule_result.overall_score
        result.overall_grade = rule_result.grade
        result.confidence = rule_result.overall_confidence
        result.eval_layer = "rule_engine"
        result.summary = rule_result.summary

        for dim in rule_result.dimensions:
            result.details.append(f"{dim.name}: {dim.score:.1f}分")
            result.issues.extend(dim.issues)

        return result

    def _build_result_from_ml(self, result: CascadeResult, ml_result: MLResult, layer: str) -> CascadeResult:
        """从 ML 模型结果构建最终结果"""
        result.overall_score = ml_result.score
        result.overall_grade = ml_result.grade
        result.confidence = ml_result.confidence
        result.eval_layer = layer
        result.summary = f"{ml_result.details}"

        # 如果有规则引擎结果，合并信息
        if result.rule_result:
            for dim in result.rule_result.dimensions:
                result.details.append(f"{dim.name}: {dim.score:.1f}分 (规则)")
                result.issues.extend(dim.issues)

        result.details.append(f"ML 评估: {ml_result.score:.1f}分 ({ml_result.grade})")

        return result

    def _build_result_from_llm(self, result: CascadeResult, llm_result) -> CascadeResult:
        """从 LLM 结果构建最终结果"""
        result.overall_score = llm_result.overall_score
        result.overall_grade = llm_result.overall_grade.value
        result.confidence = 0.6  # LLM 置信度中等
        result.eval_layer = "llm"
        result.summary = llm_result.conclusion

        for dim in llm_result.dimensions:
            result.details.append(f"{dim.name}: {dim.score:.1f}分 (LLM)")
            result.issues.extend(dim.issues)

        return result

    def _build_fallback_result(self, result: CascadeResult) -> CascadeResult:
        """构建降级结果（所有层都失败时）"""
        # 综合各层已有结果
        scores = []
        confidences = []

        if result.rule_result:
            scores.append(result.rule_result.overall_score)
            confidences.append(result.rule_result.overall_confidence)

        if result.xgboost_result:
            scores.append(result.xgboost_result.score)
            confidences.append(result.xgboost_result.confidence)

        if scores:
            # 加权平均
            weights = np.array(confidences) / sum(confidences)
            result.overall_score = np.average(scores, weights=weights)
            result.confidence = np.mean(confidences)
        else:
            result.overall_score = 0.0
            result.confidence = 0.0

        result.overall_grade = self._score_to_grade(result.overall_score)
        result.eval_layer = "fallback"
        result.summary = f"综合评估结果: {result.overall_score:.1f}分 ({result.overall_grade})"

        return result

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 90:
            return "优秀"
        elif score >= 75:
            return "良好"
        elif score >= 60:
            return "合格"
        else:
            return "不合格"

    def get_evaluation_report(self, result: CascadeResult) -> str:
        """生成评估报告文本

        Args:
            result: CascadeResult 评估结果

        Returns:
            格式化的报告文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"固井质量分层评估报告")
        lines.append("=" * 60)
        lines.append(f"井名: {result.well_name}")
        lines.append(f"深度范围: {result.depth_range[0]:.1f} - {result.depth_range[1]:.1f} m")
        lines.append(f"使用曲线: {', '.join(result.curves_used)}")
        lines.append("")
        lines.append(f"综合得分: {result.overall_score:.1f} 分")
        lines.append(f"质量等级: {result.overall_grade}")
        lines.append(f"评估置信度: {result.confidence:.2f}")
        lines.append(f"评估层级: {result.eval_layer}")
        lines.append("")

        if result.summary:
            lines.append("评估摘要:")
            lines.append(result.summary)
            lines.append("")

        if result.details:
            lines.append("详细信息:")
            for detail in result.details:
                lines.append(f"  - {detail}")
            lines.append("")

        if result.issues:
            lines.append("发现的问题:")
            for issue in result.issues:
                lines.append(f"  ⚠️ {issue}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)
