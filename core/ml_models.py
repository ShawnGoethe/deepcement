"""
机器学习模型模块
实现 XGBoost 评估器
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class MLResult:
    """ML 模型评估结果"""
    score: float = 0.0                  # 得分 (0-100)
    confidence: float = 0.0            # 置信度 (0-1)
    grade: str = "待定"                 # 质量等级
    probabilities: Dict[str, float] = field(default_factory=dict)  # 各等级概率
    features_importance: Dict[str, float] = field(default_factory=dict)
    layer: str = "ml_model"            # 评估层级
    details: str = ""


class FeatureExtractor:
    """测井曲线特征提取器

    从原始测井曲线中提取用于 ML 模型的特征
    """

    def extract_features(
        self,
        curves: Dict[str, np.ndarray],
        window_size: int = 32,
    ) -> np.ndarray:
        """提取特征向量

        Args:
            curves: 曲线数据字典 {"AC": array, "CAL": array, ...}
            window_size: 滑动窗口大小

        Returns:
            特征向量 (n_windows, n_features)
        """
        features_list = []

        ac_values = curves.get("AC", np.array([]))
        cal_values = curves.get("CAL", np.array([]))
        rt_values = curves.get("RT", np.array([]))
        gr_values = curves.get("GR", np.array([]))

        # 无效值阈值：测井数据中 -999 及以下表示缺失/无效
        INVALID_THRESHOLD = -999

        n_points = max(len(ac_values), len(cal_values), len(rt_values), len(gr_values))
        if n_points == 0:
            return np.array([])

        for i in range(0, n_points - window_size + 1, window_size // 2):
            # 检查窗口内任意曲线是否存在无效值，整行跳过
            has_invalid = False
            for arr in (ac_values, cal_values, rt_values, gr_values):
                if len(arr) > i + window_size:
                    window = arr[i:i + window_size]
                    if np.any(window < INVALID_THRESHOLD):
                        has_invalid = True
                        break
            if has_invalid:
                continue

            window_features = []

            # AC 特征
            if len(ac_values) > i + window_size:
                ac_window = ac_values[i:i + window_size]
                ac_window = ac_window[~np.isnan(ac_window)]
                ac_window = ac_window[ac_window >= INVALID_THRESHOLD]
                if len(ac_window) > 0:
                    window_features.extend([
                        np.mean(ac_window), np.std(ac_window),
                        np.min(ac_window), np.max(ac_window),
                        np.median(ac_window),
                        np.percentile(ac_window, 25), np.percentile(ac_window, 75),
                        self._calc_trend(ac_window), self._calc_variability(ac_window),
                    ])
                else:
                    window_features.extend([np.nan] * 9)
            else:
                window_features.extend([np.nan] * 9)

            # CAL 特征
            if len(cal_values) > i + window_size:
                cal_window = cal_values[i:i + window_size]
                cal_window = cal_window[~np.isnan(cal_window)]
                cal_window = cal_window[cal_window >= INVALID_THRESHOLD]
                if len(cal_window) > 0:
                    window_features.extend([
                        np.mean(cal_window), np.std(cal_window),
                        np.max(cal_window) - np.min(cal_window),
                    ])
                else:
                    window_features.extend([np.nan] * 3)
            else:
                window_features.extend([np.nan] * 3)

            # RT 特征
            if len(rt_values) > i + window_size:
                rt_window = rt_values[i:i + window_size]
                rt_window = rt_window[~np.isnan(rt_window)]
                rt_window = rt_window[rt_window >= INVALID_THRESHOLD]
                if len(rt_window) > 0:
                    window_features.extend([
                        np.mean(rt_window), np.std(rt_window),
                        np.log1p(np.mean(rt_window)),
                    ])
                else:
                    window_features.extend([np.nan] * 3)
            else:
                window_features.extend([np.nan] * 3)

            # GR 特征
            if len(gr_values) > i + window_size:
                gr_window = gr_values[i:i + window_size]
                gr_window = gr_window[~np.isnan(gr_window)]
                gr_window = gr_window[gr_window >= INVALID_THRESHOLD]
                if len(gr_window) > 0:
                    window_features.extend([np.mean(gr_window), np.std(gr_window)])
                else:
                    window_features.extend([np.nan] * 2)
            else:
                window_features.extend([np.nan] * 2)

            features_list.append(window_features)

        return np.array(features_list) if features_list else np.array([])

    def _calc_trend(self, values: np.ndarray) -> float:
        """计算趋势（线性回归斜率）"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        try:
            return np.polyfit(x, values, 1)[0]
        except:
            return 0.0

    def _calc_variability(self, values: np.ndarray) -> float:
        """计算变异系数"""
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        return np.std(values) / abs(mean)

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return [
            "AC_mean", "AC_std", "AC_min", "AC_max", "AC_median",
            "AC_q25", "AC_q75", "AC_trend", "AC_variability",
            "CAL_mean", "CAL_std", "CAL_range",
            "RT_mean", "RT_std", "RT_log_mean",
            "GR_mean", "GR_std",
        ]


class XGBoostEvaluator:
    """XGBoost 快速评估器"""

    GRADE_MAP = {0: "优秀", 1: "良好", 2: "合格", 3: "不合格"}
    SCORE_MAP = {0: 95, 1: 80, 2: 65, 3: 40}

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self._model_loaded = False

        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "models" / "xgboost_cement.json")
        self._model_path = model_path
        self._try_load_model()

    def _try_load_model(self):
        try:
            import xgboost as xgb
            if Path(self._model_path).exists():
                self.model = xgb.XGBClassifier()
                self.model.load_model(self._model_path)
                self._model_loaded = True
                logger.info(f"XGBoost 模型加载成功: {self._model_path}")
            else:
                logger.warning(f"XGBoost 模型文件不存在: {self._model_path}，将使用规则估算")
        except ImportError:
            logger.warning("未安装 xgboost，将使用规则估算")
        except Exception as e:
            logger.error(f"XGBoost 模型加载失败: {e}")

    def predict(self, curves: Dict[str, np.ndarray]) -> MLResult:
        """预测固井质量"""
        features = self.feature_extractor.extract_features(curves)
        if features.size == 0:
            return MLResult(score=0.0, confidence=0.0, details="无法提取有效特征", layer="xgboost")

        features = np.nan_to_num(features, nan=0.0)

        if self._model_loaded:
            return self._predict_with_model(features)
        else:
            return self._predict_with_rules(features, curves)

    def _predict_with_model(self, features: np.ndarray) -> MLResult:
        """使用 XGBoost 模型预测"""
        try:
            probabilities = self.model.predict_proba(features)
            avg_probs = np.mean(probabilities, axis=0)
            pred_class = np.argmax(avg_probs)
            confidence = float(avg_probs[pred_class])
            score = sum(avg_probs[i] * self.SCORE_MAP[i] for i in range(len(avg_probs)))

            prob_dict = {self.GRADE_MAP[i]: float(avg_probs[i]) for i in range(len(avg_probs))}

            importance = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_names = self.feature_extractor.get_feature_names()
                for name, imp in zip(feature_names, self.model.feature_importances_):
                    importance[name] = float(imp)

            return MLResult(
                score=float(score), confidence=confidence, grade=self.GRADE_MAP[pred_class],
                probabilities=prob_dict, features_importance=importance, layer="xgboost",
                details=f"XGBoost 预测: {self.GRADE_MAP[pred_class]}，置信度 {confidence:.2f}",
            )
        except Exception as e:
            logger.error(f"XGBoost 预测失败: {e}")
            return MLResult(score=0.0, confidence=0.0, details=f"预测失败: {e}", layer="xgboost")

    def _predict_with_rules(self, features: np.ndarray, curves: Dict[str, np.ndarray]) -> MLResult:
        """使用规则估算（模型未加载时的降级方案）"""
        ac_mean_idx = 0
        if features.shape[1] > ac_mean_idx:
            ac_means = features[:, ac_mean_idx]
            valid_mask = ac_means > 0
            if np.any(valid_mask):
                avg_ac = np.mean(ac_means[valid_mask])
                if avg_ac < 50:
                    return MLResult(score=90, confidence=0.7, grade="优秀", layer="xgboost", details=f"规则估算: AC均值={avg_ac:.1f}")
                elif avg_ac < 70:
                    return MLResult(score=78, confidence=0.6, grade="良好", layer="xgboost", details=f"规则估算: AC均值={avg_ac:.1f}")
                elif avg_ac < 90:
                    return MLResult(score=62, confidence=0.5, grade="合格", layer="xgboost", details=f"规则估算: AC均值={avg_ac:.1f}")
                else:
                    return MLResult(score=45, confidence=0.4, grade="不合格", layer="xgboost", details=f"规则估算: AC均值={avg_ac:.1f}")

        return MLResult(score=50.0, confidence=0.3, grade="待定", layer="xgboost", details="特征不足")
