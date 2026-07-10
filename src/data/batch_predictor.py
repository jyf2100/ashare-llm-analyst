"""
批量预测模块

提供批量预测功能
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.logger import get_logger
from src.data.feature_engineering import FeatureCalculator

logger = get_logger(__name__)


class BatchPredictor(AnalyzerBase):
    """
    批量预测器

    使用训练好的模型对多只股票进行批量预测

    Attributes:
        config: 配置实例
        model_dir: 模型目录
        data_dir: 数据目录
        feature_calculator: 特征计算器

    Example:
        >>> predictor = BatchPredictor()
        >>> predictions = predictor.predict_batch(["sh.600000", "sz.000001"])
        >>> print(predictions.head())
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        初始化批量预测器

        Args:
            model_dir: 模型目录路径
            data_dir: 数据目录路径
            config: 配置实例
        """
        super().__init__(config)

        if model_dir is None:
            model_dir = self.config.data.models_dir
        if data_dir is None:
            data_dir = self.config.data.data_dir

        self.model_dir = model_dir
        self.data_dir = data_dir
        self.feature_calculator = FeatureCalculator(self.config)

    def predict_batch(
        self,
        stock_codes: Optional[List[str]] = None,
        target_label: str = "return_5d_gt_5pct",
        output_file: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        批量预测多只股票

        Args:
            stock_codes: 股票代码列表，None则预测所有
            target_label: 目标标签名称
            output_file: 输出文件路径

        Returns:
            预测结果DataFrame
        """
        self.logger.info(f"开始批量预测: {target_label}")

        # 获取股票列表
        if stock_codes is None:
            stock_codes = self._get_available_stocks()

        if not stock_codes:
            self.logger.error("没有可预测的股票")
            return None

        # 加载模型
        model, scaler = self._load_model(target_label)
        if model is None:
            return None

        # 预测每只股票
        all_predictions = []

        for stock_code in stock_codes:
            try:
                pred = self._predict_single_stock(
                    stock_code, model, scaler, target_label
                )
                if pred is not None:
                    all_predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"预测 {stock_code} 失败: {e}")

        if not all_predictions:
            return None

        # 合并结果
        result_df = pd.concat(all_predictions, ignore_index=True)

        # 保存结果
        if output_file:
            result_df.to_csv(output_file, index=False, encoding="utf-8")
            self.logger.info(f"预测结果已保存: {output_file}")

        self.logger.info(f"批量预测完成: {len(result_df)} 条记录")

        return result_df

    def predict_single(
        self,
        stock_code: str,
        target_label: str = "return_5d_gt_5pct",
    ) -> Optional[Dict[str, Any]]:
        """
        预测单只股票

        Args:
            stock_code: 股票代码
            target_label: 目标标签名称

        Returns:
            预测结果字典
        """
        model, scaler = self._load_model(target_label)
        if model is None:
            return None

        return self._predict_single_stock(stock_code, model, scaler, target_label)

    def _get_available_stocks(self) -> List[str]:
        """获取可用的股票列表"""
        if not os.path.exists(self.data_dir):
            return []

        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        return [f.replace(".csv", "") for f in csv_files]

    def _load_model(self, target_label: str):
        """加载模型和scaler"""
        # 查找最新模型
        model_files = [
            f for f in os.listdir(self.model_dir)
            if f.startswith(f"random_forest_{target_label}") and f.endswith(".pkl")
            and not f.startswith("scaler_")
        ]

        if not model_files:
            self.logger.error(f"未找到模型: {target_label}")
            return None, None

        latest_model = sorted(model_files)[-1]

        # 加载
        model_path = os.path.join(self.model_dir, latest_model)
        scaler_path = os.path.join(self.model_dir, f"scaler_{latest_model}")

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            return None, None

    def _predict_single_stock(
        self,
        stock_code: str,
        model: Any,
        scaler: Any,
        target_label: str,
    ) -> Optional[pd.DataFrame]:
        """预测单只股票"""
        # 加载数据
        from src.data.providers import CSVProvider

        provider = CSVProvider(self.data_dir)
        df = provider.load(stock_code)

        if df is None or len(df) < 60:
            return None

        # 计算特征
        df_with_features = self.feature_calculator.calculate_all_features(df.reset_index())

        # 选择特征列
        feature_cols = self._select_feature_columns(df_with_features)
        if not feature_cols:
            return None

        # 获取最新特征
        latest_features = df_with_features[feature_cols].iloc[-1:].values

        # 预处理
        X = self._preprocess_features(latest_features, scaler)

        # 预测
        prediction = model.predict(X)[0]
        probability = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            probability = float(proba[1]) if len(proba) == 2 else None

        # 构建结果
        result = pd.DataFrame([{
            "stock_code": stock_code,
            "date": df_with_features["date"].iloc[-1],
            "target_label": target_label,
            "prediction": int(prediction),
            "probability": probability,
            "predict_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }])

        return result

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """选择特征列"""
        feature_patterns = [
            "close", "MA\\d+", "DIF", "DEA", "MACD",
            "K_", "D_", "J", "RSI", "BOLL_UP", "BOLL_MID", "BOLL_LOW",
            "PDI", "MDI", "ADX", "OBV", "ROC", "MAROC", "BBI",
            "rps\\d+",
        ]

        import re

        selected = []
        for pattern in feature_patterns:
            selected.extend([col for col in df.columns if re.match(pattern, col)])

        return list(set(selected))

    def _preprocess_features(self, X: np.ndarray, scaler: Any) -> np.ndarray:
        """预处理特征"""
        X_processed = X.copy()

        # 处理NaN
        nan_mask = np.isnan(X_processed)
        if nan_mask.any():
            for col in range(X_processed.shape[1]):
                col_mask = nan_mask[:, col]
                if col_mask.any():
                    col_median = np.nanmedian(X_processed[:, col])
                    X_processed[col_mask, col] = col_median

        # 处理Inf
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
            X_processed = scaler.transform(X_processed)

        return X_processed

    def get_top_predictions(
        self,
        n: int = 10,
        target_label: str = "return_5d_gt_5pct",
    ) -> Optional[pd.DataFrame]:
        """
        获取预测概率最高的N只股票

        Args:
            n: 返回数量
            target_label: 目标标签名称

        Returns:
            预测结果DataFrame
        """
        predictions = self.predict_batch(target_label=target_label)

        if predictions is None:
            return None

        # 过滤有概率的预测
        prob_df = predictions[predictions["probability"].notna()]

        # 按概率排序
        top_df = prob_df.nlargest(n, "probability")

        return top_df

    def get_prediction_summary(
        self,
        target_label: str = "return_5d_gt_5pct",
    ) -> Dict[str, Any]:
        """
        获取预测汇总统计

        Args:
            target_label: 目标标签名称

        Returns:
            汇总统计字典
        """
        predictions = self.predict_batch(target_label=target_label)

        if predictions is None:
            return {}

        summary = {
            "total_predictions": len(predictions),
            "positive_predictions": int(predictions["prediction"].sum()),
            "positive_ratio": float(predictions["prediction"].mean()),
            "avg_probability": float(predictions["probability"].mean()) if "probability" in predictions.columns else None,
            "predict_time": predictions["predict_time"].iloc[0] if len(predictions) > 0 else None,
        }

        return summary
