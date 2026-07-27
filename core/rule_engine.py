"""
规则引擎模块
基于石油行业标准的固井质量快速评估
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import yaml
from loguru import logger

from core.las_parser import LasData


@dataclass
class DimensionResult:
    """单维度评估结果"""
    name: str                           # 维度名称
    score: float = 0.0                  # 得分 (0-100)
    confidence: float = 0.0            # 置信度 (0-1)
    details: str = ""                   # 详细说明
    issues: List[str] = field(default_factory=list)
    data_points: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleResult:
    """规则引擎评估结果"""
    overall_score: float = 0.0          # 综合得分
    overall_confidence: float = 0.0     # 综合置信度
    grade: str = "待定"                 # 质量等级
    dimensions: List[DimensionResult] = field(default_factory=list)
    layer: str = "rule_engine"          # 评估层级
    well_name: str = ""
    depth_range: tuple = (0.0, 0.0)
    summary: str = ""


class RuleEngine:
    """规则引擎 - 第一层评估

    基于行业标准阈值进行快速评估，置信度高
    """

    def __init__(self, rules_path: Optional[str] = None):
        """初始化规则引擎

        Args:
            rules_path: 规则配置文件路径，默认为 config/rules.yaml
        """
        if rules_path is None:
            rules_path = str(Path(__file__).parent.parent / "config" / "rules.yaml")

        self.rules = self._load_rules(rules_path)
        logger.info("规则引擎初始化完成")

    def _load_rules(self, rules_path: str) -> dict:
        """加载规则配置"""
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                rules = yaml.safe_load(f)
            logger.info(f"加载规则配置: {rules_path}")
            return rules
        except Exception as e:
            logger.error(f"加载规则配置失败: {e}")
            return {}

    def evaluate(self, las_data: LasData) -> RuleResult:
        """执行规则评估

        Args:
            las_data: LAS 解析数据

        Returns:
            RuleResult 评估结果
        """
        logger.info(f"开始规则评估: {las_data.well_name}")

        dimensions = []

        # 1. 评估水泥胶结质量 (AC 曲线)
        if "AC" in las_data.curves:
            ac_result = self._evaluate_ac(las_data.curves["AC"].values)
            dimensions.append(ac_result)
        else:
            logger.warning("缺少 AC 曲线，跳过水泥胶结评估")

        # 2. 评估井眼状况 (CAL 曲线)
        if "CAL" in las_data.curves:
            cal_result = self._evaluate_cal(las_data.curves["CAL"].values)
            dimensions.append(cal_result)
        else:
            logger.warning("缺少 CAL 曲线，跳过井眼状况评估")

        # 3. 评估地层流体 (RT 曲线)
        if "RT" in las_data.curves:
            rt_result = self._evaluate_rt(las_data.curves["RT"].values)
            dimensions.append(rt_result)
        else:
            logger.warning("缺少 RT 曲线，跳过地层流体评估")

        # 4. 评估伽马特征 (GR 曲线)
        if "GR" in las_data.curves:
            gr_result = self._evaluate_gr(las_data.curves["GR"].values)
            dimensions.append(gr_result)

        # 计算综合得分
        overall_score, overall_confidence = self._calculate_overall(dimensions)
        grade = self._score_to_grade(overall_score)

        # 生成摘要
        summary = self._generate_summary(las_data.well_name, dimensions, overall_score, grade)

        result = RuleResult(
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            grade=grade,
            dimensions=dimensions,
            layer="rule_engine",
            well_name=las_data.well_name,
            depth_range=(las_data.depth_start, las_data.depth_end),
            summary=summary,
        )

        logger.info(f"规则评估完成: {overall_score:.1f}分 ({grade}), 置信度 {overall_confidence:.2f}")

        return result

    def _evaluate_ac(self, ac_values: np.ndarray) -> DimensionResult:
        """评估声波时差曲线

        Args:
            ac_values: AC 值数组

        Returns:
            DimensionResult 评估结果
        """
        result = DimensionResult(name="水泥胶结质量")

        # 过滤缺失值
        valid_values = ac_values[~np.isnan(ac_values)]
        if len(valid_values) == 0:
            result.details = "无有效 AC 数据"
            result.confidence = 0.0
            return result

        # 获取阈值配置
        thresholds = self.rules.get("cement_bond", {}).get("ac_thresholds", {})

        # 统计各等级占比
        excellent_max = thresholds.get("excellent", {}).get("max", 50)
        good_max = thresholds.get("good", {}).get("max", 70)
        qualified_max = thresholds.get("qualified", {}).get("max", 90)

        total = len(valid_values)
        excellent_count = np.sum(valid_values <= excellent_max)
        good_count = np.sum((valid_values > excellent_max) & (valid_values <= good_max))
        qualified_count = np.sum((valid_values > good_max) & (valid_values <= qualified_max))
        poor_count = np.sum(valid_values > qualified_max)

        # 计算加权得分
        excellent_ratio = excellent_count / total
        good_ratio = good_count / total
        qualified_ratio = qualified_count / total
        poor_ratio = poor_count / total

        score = (excellent_ratio * 95 +
                 good_ratio * 80 +
                 qualified_ratio * 65 +
                 poor_ratio * 40)

        # 检测异常
        issues = []
        anomaly_rules = self.rules.get("cement_bond", {}).get("ac_anomaly", {})

        # AC 值突变检测
        spike_threshold = anomaly_rules.get("spike_threshold", 30)
        spikes = np.where(np.abs(np.diff(valid_values)) > spike_threshold)[0]
        if len(spikes) > 0:
            issues.append(f"检测到 {len(spikes)} 处 AC 值突变")

        # 连续低值段检测
        low_segment_min = anomaly_rules.get("low_segment_min_length", 10)
        low_mask = valid_values < excellent_max
        consecutive_low = self._find_consecutive_segments(low_mask)
        long_low_segments = [seg for seg in consecutive_low if seg[1] - seg[0] >= low_segment_min]
        if long_low_segments:
            issues.append(f"检测到 {len(long_low_segments)} 段连续优质胶结")

        # 置信度（基于数据完整性）
        null_count = np.sum(np.isnan(ac_values))
        confidence = 1.0 - (null_count / len(ac_values)) * 0.5

        result.score = score
        result.confidence = confidence
        result.details = f"优良胶结 {excellent_ratio*100:.1f}%, 良好 {good_ratio*100:.1f}%, " \
                        f"合格 {qualified_ratio*100:.1f}%, 差 {poor_ratio*100:.1f}%"
        result.issues = issues
        result.data_points = {
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "total_points": total,
            "null_points": int(null_count),
        }

        return result

    def _evaluate_cal(self, cal_values: np.ndarray) -> DimensionResult:
        """评估井径曲线"""
        result = DimensionResult(name="井眼状况")

        valid_values = cal_values[~np.isnan(cal_values)]
        if len(valid_values) == 0:
            result.details = "无有效 CAL 数据"
            result.confidence = 0.0
            return result

        # 获取配置
        cal_rules = self.rules.get("caliper_rules", {})
        bit_diameter = cal_rules.get("bit_diameter", 8.5)
        washout_threshold = cal_rules.get("washout", {}).get("threshold", 1.2)
        tight_threshold = cal_rules.get("tight_hole", {}).get("threshold", 0.9)

        # 计算扩径/缩径比例
        total = len(valid_values)
        washout_count = np.sum(valid_values > bit_diameter * washout_threshold)
        tight_count = np.sum(valid_values < bit_diameter * tight_threshold)
        normal_count = total - washout_count - tight_count

        washout_ratio = washout_count / total
        tight_ratio = tight_count / total
        normal_ratio = normal_count / total

        # 计算得分
        score = normal_ratio * 90 + tight_ratio * 70 + washout_ratio * 50

        # 检测问题
        issues = []
        if washout_ratio > 0.1:
            issues.append(f"扩径段占比 {washout_ratio*100:.1f}%")
        if tight_ratio > 0.1:
            issues.append(f"缩径段占比 {tight_ratio*100:.1f}%")

        # 井径变化率检测
        change_rate_max = cal_rules.get("cal_change_rate", {}).get("max", 0.5)
        if len(valid_values) > 1:
            changes = np.abs(np.diff(valid_values)) / 0.125  # 每100ft变化
            max_change = np.max(changes)
            if max_change > change_rate_max:
                issues.append(f"井径变化率过大: {max_change:.2f}")

        confidence = 1.0 - (np.sum(np.isnan(cal_values)) / len(cal_values)) * 0.3

        result.score = score
        result.confidence = confidence
        result.details = f"正常井段 {normal_ratio*100:.1f}%, 扩径 {washout_ratio*100:.1f}%, " \
                        f"缩径 {tight_ratio*100:.1f}%"
        result.issues = issues
        result.data_points = {
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
            "bit_diameter": bit_diameter,
        }

        return result

    def _evaluate_rt(self, rt_values: np.ndarray) -> DimensionResult:
        """评估地层电阻率曲线"""
        result = DimensionResult(name="地层流体特征")

        valid_values = rt_values[~np.isnan(rt_values)]
        if len(valid_values) == 0:
            result.details = "无有效 RT 数据"
            result.confidence = 0.0
            return result

        # 获取配置
        rt_rules = self.rules.get("resistivity_rules", {}).get("rt_thresholds", {})
        gas_min = rt_rules.get("gas_zone", {}).get("min", 50)
        water_max = rt_rules.get("water_zone", {}).get("max", 20)

        # 统计各流体类型占比
        total = len(valid_values)
        gas_count = np.sum(valid_values >= gas_min)
        oil_count = np.sum((valid_values >= water_max) & (valid_values < gas_min))
        water_count = np.sum(valid_values < water_max)

        gas_ratio = gas_count / total
        oil_ratio = oil_count / total
        water_ratio = water_count / total

        # 评估气侵风险
        issues = []
        gas_rules = self.rules.get("resistivity_rules", {}).get("gas_invasion", {})
        if gas_ratio > 0.3:
            issues.append(f"含气层段占比 {gas_ratio*100:.1f}%，注意气侵风险")

        # 评分（基于流体分布的合理性）
        score = 75  # 基础分
        if gas_ratio > 0.5:
            score -= 10  # 含气过多扣分
        if water_ratio > 0.7:
            score -= 5   # 含水过多

        confidence = 1.0 - (np.sum(np.isnan(rt_values)) / len(rt_values)) * 0.3

        result.score = min(max(score, 0), 100)
        result.confidence = confidence
        result.details = f"含气层 {gas_ratio*100:.1f}%, 含油层 {oil_ratio*100:.1f}%, " \
                        f"含水层 {water_ratio*100:.1f}%"
        result.issues = issues
        result.data_points = {
            "mean": float(np.mean(valid_values)),
            "gas_ratio": gas_ratio,
            "oil_ratio": oil_ratio,
            "water_ratio": water_ratio,
        }

        return result

    def _evaluate_gr(self, gr_values: np.ndarray) -> DimensionResult:
        """评估自然伽马曲线"""
        result = DimensionResult(name="地层岩性特征")

        valid_values = gr_values[~np.isnan(gr_values)]
        if len(valid_values) == 0:
            result.details = "无有效 GR 数据"
            result.confidence = 0.0
            return result

        # 获取配置
        gr_rules = self.rules.get("gamma_ray_rules", {})
        shale_threshold = gr_rules.get("shale_threshold", 80)
        sand_threshold = gr_rules.get("sand_threshold", 30)

        # 统计岩性
        total = len(valid_values)
        shale_count = np.sum(valid_values >= shale_threshold)
        sand_count = np.sum(valid_values <= sand_threshold)
        mixed_count = total - shale_count - sand_count

        shale_ratio = shale_count / total
        sand_ratio = sand_count / total

        # 检测水泥返高位置（GR突变）
        issues = []
        gr_change_threshold = gr_rules.get("cement_top", {}).get("gr_change_threshold", 20)
        if len(valid_values) > 1:
            changes = np.abs(np.diff(valid_values))
            large_changes = np.where(changes > gr_change_threshold)[0]
            if len(large_changes) > 0:
                issues.append(f"检测到 {len(large_changes)} 处 GR 突变，可能指示水泥面位置")

        # 评分
        score = 80  # 基础分
        confidence = 1.0 - (np.sum(np.isnan(gr_values)) / len(gr_values)) * 0.2

        result.score = score
        result.confidence = confidence
        result.details = f"泥岩段 {shale_ratio*100:.1f}%, 砂岩段 {sand_ratio*100:.1f}%"
        result.issues = issues
        result.data_points = {
            "mean": float(np.mean(valid_values)),
            "shale_ratio": shale_ratio,
            "sand_ratio": sand_ratio,
        }

        return result

    def _calculate_overall(self, dimensions: List[DimensionResult]) -> tuple:
        """计算综合得分和置信度

        Returns:
            (overall_score, overall_confidence)
        """
        if not dimensions:
            return 0.0, 0.0

        weights = self.rules.get("scoring_weights", {})

        # 默认权重
        weight_map = {
            "水泥胶结质量": weights.get("ac_score", 0.35),
            "井眼状况": weights.get("cal_score", 0.20),
            "地层流体特征": weights.get("rt_score", 0.20),
            "地层岩性特征": weights.get("gr_score", 0.15),
        }

        total_score = 0.0
        total_weight = 0.0
        total_confidence = 0.0

        for dim in dimensions:
            weight = weight_map.get(dim.name, 0.10)
            total_score += dim.score * weight
            total_weight += weight
            total_confidence += dim.confidence * weight

        if total_weight > 0:
            overall_score = total_score / total_weight
            overall_confidence = total_confidence / total_weight
        else:
            overall_score = 0.0
            overall_confidence = 0.0

        return overall_score, overall_confidence

    def _score_to_grade(self, score: float) -> str:
        """根据分数确定等级"""
        grades = self.rules.get("quality_grades", {})

        if score >= grades.get("excellent", {}).get("min_score", 90):
            return "优秀"
        elif score >= grades.get("good", {}).get("min_score", 75):
            return "良好"
        elif score >= grades.get("qualified", {}).get("min_score", 60):
            return "合格"
        else:
            return "不合格"

    def _find_consecutive_segments(self, mask: np.ndarray) -> List[tuple]:
        """找到连续 True 段的起止索引"""
        segments = []
        start = None

        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                segments.append((start, i))
                start = None

        if start is not None:
            segments.append((start, len(mask)))

        return segments

    def _generate_summary(
        self,
        well_name: str,
        dimensions: List[DimensionResult],
        overall_score: float,
        grade: str,
    ) -> str:
        """生成评估摘要"""
        summary_parts = [f"井 {well_name} 固井质量评估结果："]
        summary_parts.append(f"综合得分：{overall_score:.1f}分，等级：{grade}")
        summary_parts.append("")

        for dim in dimensions:
            summary_parts.append(f"- {dim.name}：{dim.score:.1f}分")
            if dim.details:
                summary_parts.append(f"  {dim.details}")
            if dim.issues:
                for issue in dim.issues:
                    summary_parts.append(f"  ⚠️ {issue}")

        return "\n".join(summary_parts)
