"""
分层评估系统测试脚本
测试 LAS 解析、规则引擎和分层评估
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cascade_evaluator import CascadeEvaluator
from core.las_parser import LasParser
from core.rule_engine import RuleEngine


def test_las_parser():
    """测试 LAS 解析器"""
    print("=" * 60)
    print("测试 LAS 解析器")
    print("=" * 60)

    parser = LasParser()

    # 查找 LAS 文件
    las_files = list(Path("data/raw").glob("*.TXT")) + list(Path("data/raw").glob("*.las"))
    if not las_files:
        print("未找到 LAS 文件，请将测井数据放入 data/raw/ 目录")
        return None

    las_file = las_files[0]
    print(f"解析文件: {las_file.name}")

    try:
        las_data = parser.parse(str(las_file))
        print(f"井名: {las_data.well_name}")
        print(f"深度范围: {las_data.depth_start:.1f} - {las_data.depth_end:.1f} m")
        print(f"采样点数: {len(las_data.depth)}")
        print(f"曲线列表: {las_data.curve_names}")

        # 统计信息
        stats = parser.get_statistics(las_data)
        print("\n曲线统计:")
        for name, stat in stats.items():
            print(
                f"  {name}: 均值={stat['mean']:.2f}, 标准差={stat['std']:.2f}, "
                f"范围=[{stat['min']:.2f}, {stat['max']:.2f}]"
            )

        return las_data

    except Exception as e:
        print(f"解析失败: {e}")
        return None


def test_rule_engine(las_data):
    """测试规则引擎"""
    print("\n" + "=" * 60)
    print("测试规则引擎")
    print("=" * 60)

    engine = RuleEngine()

    try:
        result = engine.evaluate(las_data)
        print(f"综合得分: {result.overall_score:.1f} 分")
        print(f"质量等级: {result.grade}")
        print(f"置信度: {result.overall_confidence:.2f}")

        print("\n各维度评估:")
        for dim in result.dimensions:
            print(f"  {dim.name}: {dim.score:.1f}分 (置信度: {dim.confidence:.2f})")
            if dim.details:
                print(f"    详情: {dim.details}")
            if dim.issues:
                for issue in dim.issues:
                    print(f"    ⚠️ {issue}")

        return result

    except Exception as e:
        print(f"规则评估失败: {e}")
        return None


def test_cascade_evaluator(las_data):
    """测试分层评估器"""
    print("\n" + "=" * 60)
    print("测试分层评估器")
    print("=" * 60)

    evaluator = CascadeEvaluator()

    try:
        result = evaluator.evaluate(las_data)
        print(evaluator.get_evaluation_report(result))
        return result

    except Exception as e:
        print(f"分层评估失败: {e}")
        return None


def main():
    """主测试流程"""
    print("DeepCement 分层评估系统测试")
    print("=" * 60)

    # 1. 测试 LAS 解析
    las_data = test_las_parser()
    if las_data is None:
        print("\nLAS 解析失败，无法继续测试")
        return

    # 2. 测试规则引擎
    rule_result = test_rule_engine(las_data)

    # 3. 测试分层评估
    cascade_result = test_cascade_evaluator(las_data)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
