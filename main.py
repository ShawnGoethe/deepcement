"""
DeepCement - 固井质量评测报告系统
主入口文件

用法：
    # 构建索引
    python main.py --build-index

    # 查询评测（单次）
    python main.py --well "XX-1" --query "生成固井质量评测报告"

    # 交互模式
    pytxiang
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | {message}")
logger.add("logs/deepcement.log", rotation="10 MB", retention="7 days", level="DEBUG")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DeepCement 固井质量评测报告系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py --build-index                    # 构建索引
  python main.py --interactive                    # 交互模式
  python main.py --well "XX-1"                    # 评测指定井
  python main.py --well "XX-1" --query "质量评测"  # 带查询评测
        """,
    )

    parser.add_argument(
        "--build-index",
        action="store_true",
        help="构建向量索引（从 data/raw/ 读取文档）",
    )
    parser.add_argument(
        "--well",
        type=str,
        help="指定井名进行质量评测",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="查询文本（与 --well 配合使用）",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="进入交互模式",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存报告到文件",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    from agent.orchestrator import CementAgent
    from agent.report.generator import ReportGenerator

    agent = CementAgent()

    # 模式1：构建索引
    if args.build_index:
        logger.info("=== 构建索引 ===")
        success = agent.build_index()
        if success:
            logger.info("索引构建完成！")
        else:
            logger.error("索引构建失败，请检查 data/raw/ 目录是否有文档")
        return

    # 模式2：交互模式
    if args.interactive:
        agent.interactive()
        return

    # 模式3：单次查询
    if args.well:
        logger.info(f"=== 评测井: {args.well} ===")

        # 确保索引已加载
        if not agent.load_index():
            logger.error("索引未找到，请先运行: python main.py --build-index")
            return

        query = args.query or f"{args.well} 固井质量评测"
        result = agent.run(query)
        print("\n" + result)

        # 尝试生成报告文件
        try:
            report = agent.evaluator.evaluate(well_name=args.well, query=args.query)
            generator = ReportGenerator()
            md = generator.generate(report, save=not args.no_save)
            if not args.no_save:
                logger.info("报告文件已生成")
        except Exception as e:
            logger.warning(f"报告文件生成失败: {e}")

        return

    # 默认：显示帮助
    logger.info("欢迎使用 DeepCement 固井质量评测系统！")
    logger.info("请使用 --help 查看可用命令，或使用 --interactive 进入交互模式。")


if __name__ == "__main__":
    main()
