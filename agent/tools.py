"""
Agent 工具定义
为 DeepAgent 提供固井质量评测相关的工具函数
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from loguru import logger

from config import BASE_DIR, settings
from core.cascade_evaluator import CascadeEvaluator
from core.evaluator import QualityEvaluator
from core.retriever import HistoryRetriever


def create_cement_tools(
    retriever: HistoryRetriever,
    evaluator: QualityEvaluator,
    cascade_evaluator: CascadeEvaluator,
) -> list:
    """创建固井评测工具集

    Args:
        retriever: 历史资料检索器
        evaluator: LLM 评估器
        cascade_evaluator: 分层评估器（可选）

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

    @tool
    def evaluate_well_log(
        las_file: str,
        well_name: Optional[str] = None,
    ) -> str:
        """使用分层评估系统评估测井数据（LAS 文件）

        评估流程：规则引擎 → XGBoost → LLM 兜底
        置信度高的层级直接返回，低置信度才交给下一层。

        Args:
            las_file: LAS 文件路径（支持 .las 和 .TXT 格式）
            well_name: 井名（可选，默认从文件解析）
        """
        if cascade_evaluator is None:
            return "分层评估器未初始化，请先加载模型。"

        las_path = Path(las_file)
        if not las_path.exists():
            # 尝试在 data/raw 目录查找
            las_path = settings.raw_dir / las_file
            if not las_path.exists():
                return f"LAS 文件不存在: {las_file}"

        try:
            result = cascade_evaluator.evaluate_las_file(str(las_path))
            return cascade_evaluator.get_evaluation_report(result)
        except Exception as e:
            return f"测井数据评估失败: {e}"

    @tool
    def remove_file(file_path: str) -> str:
        """删除数据目录中的文件（危险操作，需人工确认）

        删除 data/raw/ 下的指定文件。此操作不可逆，执行前会要求人工审批。

        Args:
            file_path: 要删除的文件名或路径（相对于 data/raw/ 或绝对路径）
        """
        target = Path(file_path)

        # 如果是相对路径，在 data/raw 下查找
        if not target.is_absolute():
            target = settings.raw_dir / target

        # 安全检查：只允许删除 data/raw 目录下的文件
        try:
            target = target.resolve()
            raw_dir_resolved = settings.raw_dir.resolve()
        except Exception:
            return f"路径解析失败: {file_path}"

        if not str(target).startswith(str(raw_dir_resolved)):
            return f"安全拒绝：只能删除 data/raw/ 目录下的文件，不允许删除 {file_path}"

        if not target.exists():
            return f"文件不存在: {target}"

        if target.is_dir():
            return f"安全拒绝：不支持删除目录，请指定具体文件"

        # 获取文件信息供人工审核
        file_size = target.stat().st_size
        size_str = (
            f"{file_size / 1024:.1f} KB"
            if file_size < 1024 * 1024
            else f"{file_size / (1024 * 1024):.1f} MB"
        )

        # 执行删除（interrupt_on 已在 agent 层拦截，到这里说明已获批准）
        try:
            target.unlink()
            logger.info(f"文件已删除: {target}")
            return (
                f"✅ 文件已删除\n"
                f"  路径: {target}\n"
                f"  大小: {size_str}\n"
                f"  注意：此操作不可逆，文件已永久删除"
            )
        except Exception as e:
            logger.error(f"文件删除失败: {e}")
            return f"❌ 文件删除失败: {e}"

    @tool
    def updateXGBoost(data_dir: Optional[str] = None) -> str:
        """重新训练 XGBoost 评估模型（耗时操作，需人工确认）

        使用 data/raw/ 下的 LAS 测井数据重新训练 XGBoost 模型。
        训练完成后模型保存到 models/xgboost_cement.json。

        Args:
            data_dir: 训练数据目录（可选，默认 data/raw/）
        """
        import time

        data_path = Path(data_dir) if data_dir else settings.raw_dir
        if not data_path.exists():
            return f"训练数据目录不存在: {data_path}"

        # 收集 LAS 文件
        las_files = (
            list(data_path.glob("*.las"))
            + list(data_path.glob("*.LAS"))
            + list(data_path.glob("*.TXT"))
        )
        if not las_files:
            return f"在 {data_path} 中未找到 LAS/TXT 测井数据文件"

        logger.info(f"开始 XGBoost 模型训练，数据目录: {data_path}，文件数: {len(las_files)}")

        try:
            import numpy as np

            from core.las_parser import LasParser
            from core.ml_models import FeatureExtractor

            parser = LasParser()
            extractor = FeatureExtractor()

            all_features = []
            all_labels = []

            for las_file in las_files:
                try:
                    las_data = parser.parse(str(las_file))
                    features = extractor.extract_features(las_data.curves)
                    if features.size == 0:
                        continue

                    # 用规则引擎生成伪标签（半监督方式）
                    from core.rule_engine import RuleEngine

                    rule_engine = RuleEngine()
                    rule_result = rule_engine.evaluate(las_data)

                    # 映射等级到标签
                    grade_to_label = {"优秀": 0, "良好": 1, "合格": 2, "不合格": 3}
                    label = grade_to_label.get(rule_result.grade, 2)

                    n_windows = features.shape[0]
                    all_features.append(features)
                    all_labels.extend([label] * n_windows)

                    logger.info(f"  {las_file.name}: {n_windows} 个窗口, 标签={rule_result.grade}")
                except Exception as e:
                    logger.warning(f"  跳过 {las_file.name}: {e}")
                    continue

            if not all_features:
                return "未能从数据中提取有效特征，训练中止"

            X = np.vstack(all_features)
            y = np.array(all_labels)

            # 处理 NaN
            X = np.nan_to_num(X, nan=0.0)

            # 训练 XGBoost
            import xgboost as xgb

            start_time = time.time()

            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=4,
                eval_metric="mlogloss",
                random_state=42,
            )
            model.fit(X, y)
            train_time = time.time() - start_time

            # 保存模型
            model_dir = BASE_DIR / "models"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / "xgboost_cement.json"
            model.save_model(str(model_path))

            # 更新运行中的模型实例
            if cascade_evaluator is not None:
                cascade_evaluator.xgboost.model = model
                cascade_evaluator.xgboost._model_loaded = True

            logger.info(f"XGBoost 模型训练完成，保存至: {model_path}")

            # 统计信息
            unique, counts = np.unique(y, return_counts=True)
            grade_map = {0: "优秀", 1: "良好", 2: "合格", 3: "不合格"}
            dist = {grade_map.get(int(u), str(u)): int(c) for u, c in zip(unique, counts)}

            return (
                f"✅ XGBoost 模型训练完成\n"
                f"  训练文件数: {len(las_files)}\n"
                f"  特征窗口数: {X.shape[0]}\n"
                f"  特征维度: {X.shape[1]}\n"
                f"  标签分布: {dist}\n"
                f"  训练耗时: {train_time:.1f} 秒\n"
                f"  模型保存: {model_path}"
            )

        except ImportError as e:
            return f"缺少依赖库: {e}。请安装: pip install xgboost"
        except Exception as e:
            logger.error(f"XGBoost 训练失败: {e}")
            return f"❌ XGBoost 模型训练失败: {e}"

    return [
        search_history,
        evaluate_quality,
        compare_data,
        evaluate_well_log,
        remove_file,
        updateXGBoost,
    ]
