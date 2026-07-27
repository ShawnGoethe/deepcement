"""
LAS 测井数据解析模块
支持标准 LAS 1.2/2.0 格式测井曲线数据
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class LasCurve:
    """测井曲线定义"""
    name: str                 # 曲线名称 (如 AC, CAL, GR)
    unit: str = ""            # 单位
    description: str = ""     # 描述
    values: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class LasData:
    """LAS 文件解析结果"""
    well_name: str = ""                   # 井名
    depth_start: float = 0.0              # 起始深度
    depth_end: float = 0.0                # 结束深度
    step: float = 0.125                   # 采样间隔
    curves: Dict[str, LasCurve] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def depth(self) -> np.ndarray:
        """获取深度数组"""
        if "DEPT" in self.curves:
            return self.curves["DEPT"].values
        elif "DEPTH" in self.curves:
            return self.curves["DEPTH"].values
        else:
            # 根据起止深度和步长生成
            return np.arange(self.depth_start, self.depth_end + self.step, self.step)

    @property
    def curve_names(self) -> List[str]:
        """获取所有曲线名称"""
        return list(self.curves.keys())


class LasParser:
    """LAS 测井数据解析器

    支持格式：
    - LOGEXPRESSTOP_TEXT_FORMAT（威德福格式）
    - 标准 LAS 1.2/2.0 格式
    """

    # 常见曲线名称映射（标准化）
    CURVE_ALIASES = {
        "AC": ["AC", "DT", "DTC", "SONIC"],
        "CAL": ["CAL", "CALI", "CALIPER", "CALS"],
        "GR": ["GR", "GAMMA", "GAMMARAY"],
        "RT": ["RT", "RES", "RESIST", "ILD", "LLD"],
        "RXO": ["RXO", "Rxo", "RXOZ", "LLS"],
        "DEV": ["DEV", "DEVI", "INCL", "INCLINATION"],
        "AZ": ["AZ", "AZI", "AZIM", "AZIMUTH"],
        "DEPT": ["DEPT", "DEPTH", "DEP"],
        "CCL": ["CCL", "CCLP"],  # 套管接箍测井
        "VDL": ["VDL"],          # 变密度测井
    }

    # 缺失值标记
    NULL_VALUES = [-999.250, -9999.000, -999.0, -9999.0]

    def __init__(self):
        self._current_file = None

    def parse(self, file_path: str) -> LasData:
        """解析 LAS 文件

        Args:
            file_path: LAS 文件路径

        Returns:
            LasData 解析结果
        """
        file_path = Path(file_path)
        self._current_file = file_path.name

        logger.info(f"解析 LAS 文件: {file_path.name}")

        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # 解析头部信息
            header_lines, data_start = self._find_data_section(lines)

            # 解析头部
            well_info = self._parse_header(header_lines)

            # 解析数据体
            curves = self._parse_data(lines[data_start:], well_info.get("curves", []))

            result = LasData(
                well_name=well_info.get("well_name", file_path.stem),
                depth_start=well_info.get("depth_start", 0.0),
                depth_end=well_info.get("depth_end", 0.0),
                step=well_info.get("step", 0.125),
                curves=curves,
                metadata=well_info,
            )

            logger.info(f"解析完成: {len(curves)} 条曲线, "
                       f"深度 {result.depth_start:.1f}-{result.depth_end:.1f}m, "
                       f"{len(result.depth)} 个采样点")

            return result

        except Exception as e:
            logger.error(f"LAS 解析失败: {e}")
            raise

    def _find_data_section(self, lines: List[str]) -> Tuple[List[str], int]:
        """找到数据段的起始位置

        Returns:
            (header_lines, data_start_index)
        """
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 数据段通常以数字开头
            if stripped and not stripped.startswith('#') and not stripped.startswith('~'):
                # 检查是否是数据行（第一个字段是数字）
                parts = stripped.split()
                if parts:
                    try:
                        float(parts[0])
                        return lines[:i], i
                    except ValueError:
                        continue

        # 如果没找到明确的数据段，尝试按行分析
        logger.warning("未找到明确的数据段分隔，尝试自动检测")
        return lines, 0

    def _parse_header(self, header_lines: List[str]) -> dict:
        """解析头部信息"""
        info = {
            "well_name": "",
            "depth_start": 0.0,
            "depth_end": 0.0,
            "step": 0.125,
            "curves": [],
            "file_format": "unknown",
        }

        for line in header_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # LOGEXPRESSTOP 格式
            if 'WellName' in line or 'WELL' in line.upper():
                if '=' in line:
                    value = line.split('=', 1)[1].strip()
                    info["well_name"] = value

            if 'STDEP' in line or 'STRT' in line:
                if '=' in line:
                    try:
                        info["depth_start"] = float(line.split('=', 1)[1].strip())
                    except ValueError:
                        pass

            if 'ENDEP' in line or 'STOP' in line:
                if '=' in line:
                    try:
                        info["depth_end"] = float(line.split('=', 1)[1].strip())
                    except ValueError:
                        pass

            if 'RLEV' in line or 'STEP' in line:
                if '=' in line:
                    try:
                        info["step"] = float(line.split('=', 1)[1].strip())
                    except ValueError:
                        pass

            if 'CURVENAME' in line or 'CURVE' in line.upper():
                if '=' in line:
                    curves_str = line.split('=', 1)[1].strip()
                    info["curves"] = [c.strip() for c in curves_str.split(',')]

        return info

    def _parse_data(self, data_lines: List[str], curve_names: List[str]) -> Dict[str, LasCurve]:
        """解析数据体"""
        # 收集所有数据行
        data_rows = []
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('~'):
                continue

            parts = line.split()
            if parts:
                try:
                    row = [float(p) for p in parts]
                    data_rows.append(row)
                except ValueError:
                    continue

        if not data_rows:
            logger.warning("未找到有效的数据行")
            return {}

        # 转为 numpy 数组
        data_array = np.array(data_rows)

        # 确定列数
        n_cols = data_array.shape[1]

        # 如果没有提供曲线名称，使用默认名称
        if not curve_names:
            if n_cols >= 2:
                curve_names = ["DEPT"] + [f"CURVE_{i}" for i in range(1, n_cols)]
            else:
                curve_names = [f"CURVE_{i}" for i in range(n_cols)]

        # 确保曲线名称数量匹配
        if len(curve_names) < n_cols:
            curve_names.extend([f"CURVE_{i}" for i in range(len(curve_names), n_cols)])
        elif len(curve_names) > n_cols:
            curve_names = curve_names[:n_cols]

        # 构建曲线字典
        curves = {}
        for i, name in enumerate(curve_names):
            name = name.strip().upper()

            # 标准化曲线名称
            std_name = self._standardize_curve_name(name)

            values = data_array[:, i]

            # 处理缺失值
            values = self._handle_null_values(values)

            curves[std_name] = LasCurve(
                name=std_name,
                unit=self._get_curve_unit(std_name),
                description=self._get_curve_description(std_name),
                values=values,
            )

        return curves

    def _standardize_curve_name(self, name: str) -> str:
        """标准化曲线名称"""
        for std_name, aliases in self.CURVE_ALIASES.items():
            if name in aliases:
                return std_name
        return name

    def _handle_null_values(self, values: np.ndarray) -> np.ndarray:
        """处理缺失值，替换为 NaN"""
        result = values.copy()
        for null_val in self.NULL_VALUES:
            mask = np.isclose(result, null_val, atol=0.01)
            result[mask] = np.nan
        return result

    def _get_curve_unit(self, curve_name: str) -> str:
        """获取曲线单位"""
        units = {
            "AC": "μs/ft",
            "CAL": "in",
            "GR": "API",
            "RT": "Ω·m",
            "RXO": "Ω·m",
            "DEV": "°",
            "AZ": "°",
            "DEPT": "m",
        }
        return units.get(curve_name, "")

    def _get_curve_description(self, curve_name: str) -> str:
        """获取曲线描述"""
        descriptions = {
            "AC": "声波时差",
            "CAL": "井径",
            "GR": "自然伽马",
            "RT": "地层电阻率",
            "RXO": "冲洗带电阻率",
            "DEV": "井斜角",
            "AZ": "方位角",
            "DEPT": "测量深度",
        }
        return descriptions.get(curve_name, curve_name)

    def to_dataframe(self, las_data: LasData) -> pd.DataFrame:
        """将 LasData 转为 DataFrame

        Args:
            las_data: LAS 解析结果

        Returns:
            pandas DataFrame
        """
        data = {}
        for name, curve in las_data.curves.items():
            data[name] = curve.values

        df = pd.DataFrame(data)
        df.index.name = "INDEX"

        return df

    def get_statistics(self, las_data: LasData) -> dict:
        """获取曲线统计信息

        Args:
            las_data: LAS 解析结果

        Returns:
            统计信息字典
        """
        stats = {}
        for name, curve in las_data.curves.items():
            valid_values = curve.values[~np.isnan(curve.values)]
            if len(valid_values) > 0:
                stats[name] = {
                    "count": len(valid_values),
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "median": float(np.median(valid_values)),
                    "null_count": int(np.sum(np.isnan(curve.values))),
                }
        return stats


# 便捷函数
def parse_las_file(file_path: str) -> LasData:
    """解析 LAS 文件的便捷函数"""
    parser = LasParser()
    return parser.parse(file_path)


def las_to_dataframe(file_path: str) -> pd.DataFrame:
    """将 LAS 文件转为 DataFrame 的便捷函数"""
    parser = LasParser()
    las_data = parser.parse(file_path)
    return parser.to_dataframe(las_data)
