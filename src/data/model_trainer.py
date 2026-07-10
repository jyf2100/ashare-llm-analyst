"""
模型训练器模块

提供机器学习模型的训练和预测功能
"""

import os
import pickle
import shutil
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.core.base import AnalyzerBase, BaseModel
from src.core.config import Config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)


class ModelTrainerBase(AnalyzerBase):
    """
    模型训练器抽象基类

    定义模型训练的通用接口

    Attributes:
        training_data_dir: 训练数据目录
        model_save_dir: 模型保存目录
        model: 训练好的模型
        scaler: 特征标准化器

    Example:
        >>> trainer = MyModelTrainer()
        >>> results = trainer.train_models(target_label="return_5d_gt_5pct")
    """

    def __init__(
        self,
        training_data_dir: Optional[str] = None,
        model_save_dir: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        初始化模型训练器

        Args:
            training_data_dir: 训练数据目录
            model_save_dir: 模型保存目录
            config: 配置实例
        """
        super().__init__(config)

        # 确定目录路径
        if training_data_dir is None:
            training_data_dir = "training_data"
        if model_save_dir is None:
            model_save_dir = self.config.data.models_dir

        self.training_data_dir = training_data_dir
        self.model_save_dir = model_save_dir

        # 创建模型保存目录
        os.makedirs(self.model_save_dir, exist_ok=True)

        # 模型和标准化器
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            X: 特征数据
            y: 标签数据
            **kwargs: 其他参数

        Returns:
            训练结果字典
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 特征数据

        Returns:
            预测结果
        """
        pass

    def load_training_data(self, target_label: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """
        加载训练数据

        Args:
            target_label: 目标标签名称

        Returns:
            (特征X, 标签y, 实际标签名) 元组
        """
        # 查找最新的特征文件
        feature_files = [
            f for f in os.listdir(self.training_data_dir) if f.startswith("features_") and f.endswith(".npy")
        ]

        if not feature_files:
            self.logger.error("未找到特征数据文件")
            return None, None, None

        latest_feature_file = sorted(feature_files)[-1]
        timestamp = latest_feature_file.split("_")[-1].replace(".npy", "")

        # 加载特征
        X = np.load(os.path.join(self.training_data_dir, latest_feature_file))
        self.logger.info(f"加载特征数据: {latest_feature_file}, 形状: {X.shape}")

        # 加载标签
        label_file = f"labels_{target_label}_{timestamp}.npy"
        label_path = os.path.join(self.training_data_dir, label_file)

        if not os.path.exists(label_path):
            self.logger.error(f"标签文件不存在: {label_file}")
            return X, None, None

        y = np.load(label_path)
        self.logger.info(f"加载标签数据: {label_file}, 形状: {y.shape}")

        return X, y, target_label

    def preprocess_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, use_robust: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        数据预处理

        Args:
            X: 特征数据
            y: 标签数据(可选)
            use_robust: 是否使用RobustScaler(对异常值更鲁棒)

        Returns:
            (预处理后的X, 预处理后的y) 元组
        """
        self.logger.info("开始数据预处理...")

        # 处理缺失值
        nan_mask = np.isnan(X)
        if nan_mask.any():
            nan_count = nan_mask.sum()
            self.logger.warning(f"发现 {nan_count} 个缺失值，使用中位数填充")
            for col in range(X.shape[1]):
                col_mask = nan_mask[:, col]
                if col_mask.any():
                    col_median = np.nanmedian(X[:, col])
                    X[col_mask, col] = col_median

        # 处理无穷值
        inf_mask = np.isinf(X)
        if inf_mask.any():
            self.logger.warning(f"发现 {inf_mask.sum()} 个无穷值，进行处理")
            for col in range(X.shape[1]):
                col_data = X[:, col]
                finite_data = col_data[np.isfinite(col_data)]
                if len(finite_data) > 0:
                    col_max = np.max(finite_data)
                    col_min = np.min(finite_data)
                    X[np.isposinf(col_data), col] = col_max
                    X[np.isneginf(col_data), col] = col_min

        # 标准化
        if use_robust:
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        X_scaled = self.scaler.fit_transform(X)

        self.logger.info("数据预处理完成")

        if y is not None:
            return X_scaled, y
        return X_scaled, None

    def save_model(self, model_name: str) -> bool:
        """
        保存模型和标准化器

        Args:
            model_name: 模型名称

        Returns:
            是否成功保存
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d")

            # 保存模型
            model_path = os.path.join(self.model_save_dir, f"{model_name}_{timestamp}.pkl")
            joblib.dump(self.model, model_path)
            self.logger.info(f"模型已保存: {model_path}")

            # 保存标准化器
            if self.scaler is not None:
                scaler_path = os.path.join(self.model_save_dir, f"scaler_{model_name}_{timestamp}.pkl")
                joblib.dump(self.scaler, scaler_path)
                self.logger.info(f"标准化器已保存: {scaler_path}")

            return True

        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
            return False

    def load_model(self, model_path: str, scaler_path: str) -> bool:
        """
        加载模型和标准化器

        Args:
            model_path: 模型文件路径
            scaler_path: 标准化器文件路径

        Returns:
            是否成功加载
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.logger.info(f"模型加载成功: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False


class RandomForestTrainer(ModelTrainerBase):
    """
    随机森林模型训练器

    支持分类和回归任务的随机森林训练

    Attributes:
        model_params: 模型参数配置

    Example:
        >>> trainer = RandomForestTrainer()
        >>> results = trainer.train_models("return_5d_gt_5pct")
    """

    def __init__(
        self,
        training_data_dir: Optional[str] = None,
        model_save_dir: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        初始化随机森林训练器

        Args:
            training_data_dir: 训练数据目录
            model_save_dir: 模型保存目录
            config: 配置实例
        """
        super().__init__(training_data_dir, model_save_dir, config)

        # 随机森林参数配置
        self.model_params = {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
            "bootstrap": True,
        }

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "classifier",
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        训练随机森林模型

        Args:
            X: 特征数据
            y: 标签数据
            model_type: 模型类型 ("classifier" 或 "regressor")
            test_size: 测试集比例

        Returns:
            训练结果字典
        """
        # 预处理数据
        X_processed, y_processed = self.preprocess_data(X, y)

        # 分割数据集
        if model_type == "regressor":
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=test_size, random_state=42
            )
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=42, stratify=y_processed
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=42
                )

        # 创建模型
        if model_type == "regressor":
            model = RandomForestRegressor(**self.model_params)
        else:
            model = RandomForestClassifier(
                **self.model_params, class_weight="balanced"
            )

        # 训练模型
        self.logger.info(f"开始训练{model_type}模型...")
        model.fit(X_train, y_train)
        self.model = model

        # 预测和评估
        y_pred = model.predict(X_test)

        results = {
            "model_type": model_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        if model_type == "regressor":
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            self.logger.info(f"回归评估: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")

            results.update({"rmse": rmse, "r2": r2, "mae": mae})

        else:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            try:
                y_pred_proba = model.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
            except Exception:
                auc = None

            self.logger.info(f"分类评估: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc}")

            results.update({"accuracy": accuracy, "f1": f1, "auc": auc})

        # 特征重要性
        if hasattr(model, "feature_importances_"):
            results["feature_importances"] = model.feature_importances_.tolist()

        return results

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测

        Args:
            X: 特征数据

        Returns:
            (预测结果, 概率) 元组
        """
        if self.model is None:
            self.logger.error("模型未训练")
            return None, None

        if self.scaler is None:
            self.logger.error("标准化器未训练")
            return None, None

        # 预处理
        X_processed, _ = self.preprocess_data(X)

        # 预测
        predictions = self.model.predict(X_processed)

        # 概率预测
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            try:
                probabilities = self.model.predict_proba(X_processed)
            except Exception:
                pass

        return predictions, probabilities

    def train_models(self, target_label: str, **kwargs) -> Optional[Dict[str, Dict]]:
        """
        训练模型(便捷方法)

        Args:
            target_label: 目标标签
            **kwargs: 训练参数

        Returns:
            训练结果字典
        """
        # 加载数据
        X, y, actual_label = self.load_training_data(target_label)

        if X is None or y is None:
            self.logger.error("数据加载失败")
            return None

        # 确定模型类型
        if actual_label.endswith("_continuous"):
            model_type = "regressor"
        else:
            model_type = "classifier"

        # 训练模型
        results = self.train(X, y, model_type=model_type, **kwargs)

        # 保存模型
        model_name = f"random_forest_{actual_label}"
        self.save_model(model_name)

        return {model_name: results}
