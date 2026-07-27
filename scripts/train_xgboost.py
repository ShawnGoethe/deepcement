"""
XGBoost 模型训练脚本
使用 data/raw/ 目录下的 LAS/TXT 测井数据文件训练模型

由于没有人工标注数据，使用规则引擎自动生成训练标签
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.las_parser import LasData, LasParser
from core.ml_models import FeatureExtractor
from core.rule_engine import RuleEngine


def find_las_files(data_dir: str = "data/raw") -> List[Path]:
    """查找所有 LAS/TXT 测井数据文件"""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return []

    # 查找所有可能的测井数据文件（Windows 不区分大小写，需要去重）
    las_files = set()
    for ext in ["*.las", "*.LAS", "*.TXT", "*.txt"]:
        las_files.update(data_path.glob(ext))

    las_files = sorted(las_files)
    logger.info(f"找到 {len(las_files)} 个测井数据文件")
    return las_files


def extract_features_from_file(
    file_path: Path, parser: LasParser, extractor: FeatureExtractor
) -> Tuple[np.ndarray, LasData]:
    """从单个文件提取特征

    Args:
        file_path: LAS 文件路径
        parser: LAS 解析器
        extractor: 特征提取器

    Returns:
        (特征矩阵, LAS 数据)
    """
    try:
        las_data = parser.parse(str(file_path))

        # 将 LasCurve 对象转换为 numpy 数组字典
        curves_dict = {}
        for name, curve in las_data.curves.items():
            curves_dict[name] = curve.values

        features = extractor.extract_features(curves_dict)
        return features, las_data
    except Exception as e:
        logger.error(f"处理文件失败 {file_path.name}: {e}")
        return np.array([]), None


def generate_labels_from_rules(
    las_data: LasData,
    rule_engine: RuleEngine,
    window_size: int = 32,
) -> np.ndarray:
    """使用规则引擎生成训练标签

    Args:
        las_data: LAS 数据
        rule_engine: 规则引擎
        window_size: 窗口大小（需要与特征提取一致）

    Returns:
        标签数组 (0=优秀, 1=良好, 2=合格, 3=不合格)
    """
    # 获取 AC 曲线（LasCurve 对象需要取 .values）
    ac_curve = las_data.curves.get("AC")
    ac_values = ac_curve.values if ac_curve is not None else np.array([])
    if len(ac_values) == 0:
        return np.array([])

    # 获取规则阈值
    thresholds = rule_engine.rules.get("cement_bond", {}).get("ac_thresholds", {})
    excellent_max = thresholds.get("excellent", {}).get("max", 50)
    good_max = thresholds.get("good", {}).get("max", 70)
    qualified_max = thresholds.get("qualified", {}).get("max", 90)

    # 对每个窗口生成标签
    labels = []
    n_points = len(ac_values)

    for i in range(0, n_points - window_size + 1, window_size // 2):
        window = ac_values[i : i + window_size]
        window = window[~np.isnan(window)]

        if len(window) == 0:
            # 无有效数据，标记为不合格
            labels.append(3)
            continue

        # 计算窗口内各等级占比
        excellent_ratio = np.sum(window <= excellent_max) / len(window)
        good_ratio = np.sum((window > excellent_max) & (window <= good_max)) / len(window)
        qualified_ratio = np.sum((window > good_max) & (window <= qualified_max)) / len(window)
        poor_ratio = np.sum(window > qualified_max) / len(window)

        # 根据占比确定标签
        # 如果优良占比 > 60%，标记为优秀
        if excellent_ratio > 0.6:
            labels.append(0)  # 优秀
        # 如果良好及以上占比 > 60%，标记为良好
        elif excellent_ratio + good_ratio > 0.6:
            labels.append(1)  # 良好
        # 如果合格及以上占比 > 60%，标记为合格
        elif excellent_ratio + good_ratio + qualified_ratio > 0.6:
            labels.append(2)  # 合格
        else:
            labels.append(3)  # 不合格

    return np.array(labels)


def prepare_training_data(
    data_dir: str = "data/raw",
    window_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """准备训练数据

    Args:
        data_dir: 数据目录
        window_size: 窗口大小

    Returns:
        (特征矩阵, 标签数组, 文件名列表)
    """
    parser = LasParser()
    extractor = FeatureExtractor()
    rule_engine = RuleEngine()

    # 查找文件
    las_files = find_las_files(data_dir)
    if not las_files:
        logger.error("未找到测井数据文件")
        return np.array([]), np.array([]), []

    all_features = []
    all_labels = []
    file_names = []

    for file_path in las_files:
        logger.info(f"处理文件: {file_path.name}")

        # 提取特征
        features, las_data = extract_features_from_file(file_path, parser, extractor)
        if features.size == 0 or las_data is None:
            continue

        # 生成标签
        labels = generate_labels_from_rules(las_data, rule_engine, window_size)
        if labels.size == 0:
            continue

        # 确保特征和标签数量匹配
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]

        all_features.append(features)
        all_labels.append(labels)
        file_names.extend([file_path.name] * min_len)

        logger.info(f"  提取 {min_len} 个样本")

    if not all_features:
        logger.error("未能提取任何有效特征")
        return np.array([]), np.array([]), []

    # 合并所有数据
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    # 处理缺失值
    X = np.nan_to_num(X, nan=0.0)

    logger.info(f"训练数据准备完成: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    logger.info(
        f"标签分布: 优秀={np.sum(y==0)}, 良好={np.sum(y==1)}, 合格={np.sum(y==2)}, 不合格={np.sum(y==3)}"
    )

    return X, y, file_names


def train_xgboost_model(
    X: np.ndarray,
    y: np.ndarray,
    output_path: str = "models/xgboost_cement.json",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """训练 XGBoost 模型

    Args:
        X: 特征矩阵
        y: 标签数组
        output_path: 模型保存路径
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        训练结果字典
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        logger.error("请安装: pip install xgboost scikit-learn")
        return {}

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"训练集: {X_train.shape[0]} 个样本")
    logger.info(f"测试集: {X_test.shape[0]} 个样本")

    # 计算类别权重（处理不平衡）
    class_counts = np.bincount(y_train.astype(int))
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    sample_weights = class_weights[y_train.astype(int)]

    # 创建 XGBoost 模型
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )

    # 训练模型
    logger.info("开始训练 XGBoost 模型...")
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["优秀", "良好", "合格", "不合格"],
        output_dict=True,
    )
    conf_matrix = confusion_matrix(y_test, y_pred)

    logger.info(f"模型准确率: {accuracy:.4f}")
    logger.info("\n分类报告:")
    logger.info(
        classification_report(
            y_test,
            y_pred,
            target_names=["优秀", "良好", "合格", "不合格"],
        )
    )
    logger.info(f"\n混淆矩阵:\n{conf_matrix}")

    # 保存模型
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(output_path)
    logger.info(f"模型已保存到: {output_path}")

    # 特征重要性
    feature_names = FeatureExtractor().get_feature_names()
    importance = dict(zip(feature_names, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    logger.info("\n特征重要性 (Top 10):")
    for name, imp in list(importance.items())[:10]:
        logger.info(f"  {name}: {imp:.4f}")

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": conf_matrix,
        "feature_importance": importance,
        "model_path": output_path,
    }


def load_existing_model(model_path: str = "models/xgboost_cement.json"):
    """加载已有的 XGBoost 模型

    Args:
        model_path: 模型文件路径

    Returns:
        加载的模型，如果不存在则返回 None
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("未安装 xgboost")
        return None

    path = Path(model_path)
    if not path.exists():
        logger.warning(f"模型文件不存在: {model_path}")
        return None

    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info(f"加载已有模型: {model_path}")
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None


def incremental_train(
    X_new: np.ndarray,
    y_new: np.ndarray,
    model_path: str = "models/xgboost_cement.json",
    output_path: str = None,
    n增量_trees: int = 50,
    learning_rate: float = 0.05,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """增量学习：在已有模型基础上继续训练

    Args:
        X_new: 新数据特征矩阵
        y_new: 新数据标签数组
        model_path: 已有模型路径
        output_path: 输出模型路径（默认覆盖原模型）
        n增量_trees: 增量训练的树数量
        learning_rate: 增量学习率（通常比初始训练小）
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        训练结果字典
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        return {}

    if output_path is None:
        output_path = model_path

    # 加载已有模型
    base_model = load_existing_model(model_path)
    if base_model is None:
        logger.warning("未找到已有模型，将进行全量训练")
        return train_xgboost_model(X_new, y_new, output_path, test_size, random_state)

    logger.info(f"增量训练: 新数据 {X_new.shape[0]} 个样本")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y_new, test_size=test_size, random_state=random_state, stratify=y_new
    )

    # 计算类别权重
    class_counts = np.bincount(y_train.astype(int))
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    sample_weights = class_weights[y_train.astype(int)]

    # 增量训练：使用 xgb_model 参数加载已有模型继续训练
    logger.info(f"开始增量训练 (新增 {n增量_trees} 棵树, lr={learning_rate})...")

    # XGBoost 增量训练需要使用原生 API
    try:
        # 加载已有模型为 Booster
        booster = xgb.Booster()
        booster.load_model(model_path)

        # 创建 DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # 增量训练参数
        params = {
            "max_depth": 6,
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "multi:softprob",
            "num_class": 4,
            "eval_metric": "mlogloss",
        }

        # 增量训练
        updated_booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n增量_trees,
            xgb_model=booster,
            evals=[(dtest, "test")],
            verbose_eval=False,
        )

        # 保存更新后的模型
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        updated_booster.save_model(output_path)

        # 用 sklearn 包装以便评估
        model = xgb.XGBClassifier()
        model.load_model(output_path)

    except Exception as e:
        logger.error(f"增量训练失败: {e}")
        logger.info("回退到全量训练...")
        return train_xgboost_model(X_new, y_new, output_path, test_size, random_state)

    # 评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["优秀", "良好", "合格", "不合格"],
        output_dict=True,
    )
    conf_matrix = confusion_matrix(y_test, y_pred)

    logger.info(f"增量训练后准确率: {accuracy:.4f}")
    logger.info("\n分类报告:")
    logger.info(classification_report(
        y_test, y_pred,
        target_names=["优秀", "良好", "合格", "不合格"],
    ))

    # 特征重要性
    feature_names = FeatureExtractor().get_feature_names()
    importance = dict(zip(feature_names, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": conf_matrix,
        "feature_importance": importance,
        "model_path": output_path,
        "mode": "incremental",
    }


def prepare_single_file_data(
    file_path: str,
    window_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """从单个文件准备训练数据

    Args:
        file_path: LAS 文件路径
        window_size: 窗口大小

    Returns:
        (特征矩阵, 标签数组)
    """
    parser = LasParser()
    extractor = FeatureExtractor()
    rule_engine = RuleEngine()

    path = Path(file_path)
    if not path.exists():
        logger.error(f"文件不存在: {file_path}")
        return np.array([]), np.array([])

    logger.info(f"处理文件: {path.name}")

    # 提取特征
    features, las_data = extract_features_from_file(path, parser, extractor)
    if features.size == 0 or las_data is None:
        return np.array([]), np.array([])

    # 生成标签
    labels = generate_labels_from_rules(las_data, rule_engine, window_size)
    if labels.size == 0:
        return np.array([]), np.array([])

    # 确保数量匹配
    min_len = min(len(features), len(labels))
    features = features[:min_len]
    labels = labels[:min_len]

    # 处理缺失值
    features = np.nan_to_num(features, nan=0.0)

    logger.info(f"提取 {min_len} 个样本")
    return features, labels


def main():
    """主训练流程"""
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost 固井质量评估模型训练")
    parser.add_argument("--mode", choices=["train", "incremental"], default="train",
                       help="训练模式: train=全量训练, incremental=增量学习")
    parser.add_argument("--data-dir", default="data/raw",
                       help="数据目录路径")
    parser.add_argument("--file", default=None,
                       help="增量学习时指定单个文件路径")
    parser.add_argument("--model-path", default="models/xgboost_cement.json",
                       help="模型保存/加载路径")
    parser.add_argument("--增量-trees", type=int, default=50,
                       help="增量训练的树数量")
    parser.add_argument("--learning-rate", type=float, default=0.05,
                       help="增量学习率")

    args = parser.parse_args()

    print("=" * 60)
    print("XGBoost 固井质量评估模型训练")
    print("=" * 60)
    print(f"模式: {args.mode}")

    if args.mode == "train":
        # ========== 全量训练 ==========
        print("\n[1/3] 准备训练数据...")
        X, y, file_names = prepare_training_data(
            data_dir=args.data_dir,
            window_size=32,
        )

        if X.size == 0:
            print("错误: 无法准备训练数据")
            return

        print(f"训练数据: {X.shape[0]} 个样本, {X.shape[1]} 个特征")

        print("\n[2/3] 训练 XGBoost 模型...")
        result = train_xgboost_model(
            X, y,
            output_path=args.model_path,
            test_size=0.2,
            random_state=42,
        )

    elif args.mode == "incremental":
        # ========== 增量学习 ==========
        if args.file:
            # 从单个文件增量学习
            print(f"\n[1/2] 从文件准备增量数据: {args.file}")
            X_new, y_new = prepare_single_file_data(args.file)
        else:
            # 从目录增量学习
            print(f"\n[1/2] 从目录准备增量数据: {args.data_dir}")
            X_new, y_new, _ = prepare_training_data(
                data_dir=args.data_dir,
                window_size=32,
            )

        if X_new.size == 0:
            print("错误: 无法准备增量数据")
            return

        print(f"增量数据: {X_new.shape[0]} 个样本, {X_new.shape[1]} 个特征")

        print("\n[2/2] 增量训练...")
        result = incremental_train(
            X_new, y_new,
            model_path=args.model_path,
            n增量_trees=args.增量_trees,
            learning_rate=args.learning_rate,
        )

    if not result:
        print("错误: 训练失败")
        return

    # 输出结果
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"模型准确率: {result['accuracy']:.4f}")
    print(f"模型保存路径: {result['model_path']}")
    if result.get("mode") == "incremental":
        print("训练模式: 增量学习")

    print("\n特征重要性 (Top 5):")
    for name, imp in list(result["feature_importance"].items())[:5]:
        print(f"  {name}: {imp:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
