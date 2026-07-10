"""
模型评估模块

提供模型性能评估和验证功能
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)

from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """
    模型评估结果

    Attributes:
        model_name: 模型名称
        accuracy: 准确率（分类模型）
        f1_score: F1分数（分类模型）
        auc: AUC值（分类模型）
        rmse: 均方根误差（回归模型）
        mae: 平均绝对误差（回归模型）
        r2: R²分数（回归模型）
        confusion_matrix: 混淆矩阵
        classification_report: 分类报告
        feature_importances: 特征重要性
        evaluated_at: 评估时间
    """
    model_name: str = ""
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    feature_importances: Optional[List[float]] = None
    evaluated_at: Optional[datetime] = None


class ModelEvaluator(AnalyzerBase):
    """
    模型评估器

    评估训练好的机器学习模型的性能

    Attributes:
        config: 配置实例
        model_dir: 模型目录
        training_data_dir: 训练数据目录

    Example:
        >>> evaluator = ModelEvaluator()
        >>> result = evaluator.evaluate("return_5d_gt_5pct")
        >>> print(f"准确率: {result.accuracy}")
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        training_data_dir: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        初始化模型评估器

        Args:
            model_dir: 模型目录路径
            training_data_dir: 训练数据目录路径
            config: 配置实例
        """
        super().__init__(config)

        if model_dir is None:
            model_dir = self.config.data.models_dir
        if training_data_dir is None:
            training_data_dir = "training_data"

        self.model_dir = model_dir
        self.training_data_dir = training_data_dir

    def evaluate(
        self,
        target_label: str,
        test_size: float = 0.2,
        model_name: Optional[str] = None,
    ) -> Optional[EvaluationResult]:
        """
        评估模型性能

        Args:
            target_label: 目标标签名称
            test_size: 测试集比例
            model_name: 模型名称，None则使用最新的

        Returns:
            评估结果
        """
        self.logger.info(f"开始评估模型: {target_label}")

        # 加载模型
        if model_name is None:
            model_name = self._find_latest_model(target_label)

        if model_name is None:
            self.logger.error(f"未找到模型: {target_label}")
            return None

        # 加载模型和scaler
        model, scaler = self._load_model(model_name)
        if model is None:
            return None

        # 加载数据
        X, y = self._load_evaluation_data(target_label)
        if X is None or y is None:
            return None

        # 预处理数据
        X_processed = self._preprocess_data(X, scaler, fit=False)

        # 预测
        y_pred = model.predict(X_processed)

        # 构建评估结果
        result = EvaluationResult(
            model_name=model_name,
            evaluated_at=datetime.now()
        )

        # 判断模型类型并计算相应指标
        if self._is_classification_model(target_label):
            result = self._evaluate_classification(
                y, y_pred, result, X_processed, model
            )
        else:
            result = self._evaluate_regression(
                y, y_pred, result
            )

        self.logger.info(
            f"评估完成: 准确率={result.accuracy}, "
            f"F1={result.f1_score}, R²={result.r2}"
        )

        return result

    def _find_latest_model(self, target_label: str) -> Optional[str]:
        """查找最新的模型"""
        if not os.path.exists(self.model_dir):
            return None

        model_files = [
            f for f in os.listdir(self.model_dir)
            if f.startswith(f"random_forest_{target_label}") and f.endswith(".pkl")
            and not f.startswith("scaler_")
        ]

        if not model_files:
            return None

        # 按日期排序，返回最新的
        return sorted(model_files)[-1]

    def _load_model(self, model_name: str):
        """加载模型和scaler"""
        model_path = os.path.join(self.model_dir, model_name)
        scaler_name = f"scaler_{model_name}"
        scaler_path = os.path.join(self.model_dir, scaler_name)

        if not os.path.exists(model_path):
            self.logger.error(f"模型文件不存在: {model_path}")
            return None, None

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            return model, scaler
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            return None, None

    def _load_evaluation_data(self, target_label: str):
        """加载评估数据"""
        # 查找最新的特征文件
        feature_files = [
            f for f in os.listdir(self.training_data_dir)
            if f.startswith("features_") and f.endswith(".npy")
        ]

        if not feature_files:
            self.logger.error("未找到特征数据文件")
            return None, None

        latest_feature_file = sorted(feature_files)[-1]
        timestamp = latest_feature_file.split("_")[-1].replace(".npy", "")

        # 加载特征和标签
        X = np.load(os.path.join(self.training_data_dir, latest_feature_file))
        label_file = f"labels_{target_label}_{timestamp}.npy"
        label_path = os.path.join(self.training_data_dir, label_file)

        if not os.path.exists(label_path):
            self.logger.error(f"标签文件不存在: {label_file}")
            return None, None

        y = np.load(label_path)

        return X, y

    def _preprocess_data(self, X: np.ndarray, scaler, fit: bool = True):
        """预处理数据"""
        # 处理缺失值
        X_processed = X.copy()
        nan_mask = np.isnan(X_processed)
        if nan_mask.any():
            for col in range(X_processed.shape[1]):
                col_mask = nan_mask[:, col]
                if col_mask.any():
                    col_median = np.nanmedian(X_processed[:, col])
                    X_processed[col_mask, col] = col_median

        # 处理无穷值
        inf_mask = np.isinf(X_processed)
        if inf_mask.any():
            for col in range(X_processed.shape[1]):
                col_data = X_processed[:, col]
                finite_data = col_data[np.isfinite(col_data)]
                if len(finite_data) > 0:
                    X_processed[np.isposinf(col_data), col] = np.max(finite_data)
                    X_processed[np.isneginf(col_data), col] = np.min(finite_data)

        # 标准化
        if scaler is not None:
            if fit:
                X_processed = scaler.fit_transform(X_processed)
            else:
                X_processed = scaler.transform(X_processed)

        return X_processed

    def _is_classification_model(self, target_label: str) -> bool:
        """判断是否为分类模型"""
        return not target_label.endswith("_continuous")

    def _evaluate_classification(
        self, y_true, y_pred, result: EvaluationResult, X, model
    ) -> EvaluationResult:
        """评估分类模型"""
        result.accuracy = accuracy_score(y_true, y_pred)
        result.f1_score = f1_score(y_true, y_pred, average="weighted")
        result.confusion_matrix = confusion_matrix(y_true, y_pred)
        result.classification_report = classification_report(
            y_true, y_pred, output_dict=True
        )

        # 尝试计算AUC
        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X)
                if y_pred_proba.shape[1] == 2:
                    result.auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    result.auc = roc_auc_score(
                        y_true, y_pred_proba, multi_class="ovr", average="weighted"
                    )
        except Exception:
            pass

        # 特征重要性
        if hasattr(model, "feature_importances_"):
            result.feature_importances = model.feature_importances_.tolist()

        return result

    def _evaluate_regression(
        self, y_true, y_pred, result: EvaluationResult
    ) -> EvaluationResult:
        """评估回归模型"""
        result.rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        result.mae = mean_absolute_error(y_true, y_pred)
        result.r2 = r2_score(y_true, y_pred)

        return result

    def compare_models(
        self, target_label: str, top_n: int = 5
    ) -> pd.DataFrame:
        """
        比较多个模型的性能

        Args:
            target_label: 目标标签名称
            top_n: 返回前N个模型

        Returns:
            模型比较结果DataFrame
        """
        model_files = [
            f for f in os.listdir(self.model_dir)
            if f.startswith(f"random_forest_{target_label}") and f.endswith(".pkl")
            and not f.startswith("scaler_")
        ]

        if not model_files:
            return pd.DataFrame()

        results = []

        for model_file in sorted(model_files)[-top_n:]:
            model_name = model_file.replace(".pkl", "")
            eval_result = self.evaluate(target_label, model_name=model_name)

            if eval_result:
                results.append({
                    "model": model_name,
                    "accuracy": eval_result.accuracy,
                    "f1_score": eval_result.f1_score,
                    "auc": eval_result.auc,
                    "rmse": eval_result.rmse,
                    "r2": eval_result.r2,
                })

        return pd.DataFrame(results)

    def generate_report(
        self, target_label: str, output_file: Optional[str] = None
    ) -> str:
        """
        生成评估报告

        Args:
            target_label: 目标标签名称
            output_file: 输出文件路径

        Returns:
            报告文本
        """
        result = self.evaluate(target_label)

        if result is None:
            return "评估失败"

        lines = [
            "=" * 70,
            f"模型评估报告: {result.model_name}",
            f"评估时间: {result.evaluated_at}",
            "=" * 70,
        ]

        if result.accuracy is not None:
            lines.extend([
                f"\n分类指标:",
                f"  准确率 (Accuracy): {result.accuracy:.4f}",
                f"  F1分数: {result.f1_score:.4f}",
                f"  AUC: {result.auc if result.auc else 'N/A'}",
            ])

        if result.rmse is not None:
            lines.extend([
                f"\n回归指标:",
                f"  均方根误差 (RMSE): {result.rmse:.4f}",
                f"  平均绝对误差 (MAE): {result.mae:.4f}",
                f"  R²分数: {result.r2:.4f}",
            ])

        report = "\n".join(lines)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            self.logger.info(f"报告已保存: {output_file}")

        return report
