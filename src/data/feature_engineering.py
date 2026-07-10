"""
特征工程模块

提供技术指标计算和机器学习训练数据生成功能
"""

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.base import AnalyzerBase
from src.core.cache import CacheConfig, cached
from src.core.config import Config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MLTrainingConfig:
    """
    机器学习训练配置类

    Attributes:
        PREDICTION_HORIZONS: 预测周期列表(天)
        RETURN_THRESHOLDS: 收益率阈值列表
        FEATURE_WINDOWS: 特征窗口配置
        RPS_PERIODS: RPS周期列表
        MIN_TRAINING_SAMPLES: 最小训练样本数
        MIN_COVERAGE_RATE: 最小数据覆盖率
    """
    PREDICTION_HORIZONS: List[int] = field(default_factory=lambda: [5, 10, 20])
    RETURN_THRESHOLDS: List[float] = field(default_factory=lambda: [0.03, 0.05, 0.08])
    FEATURE_WINDOWS: Dict = field(default_factory=lambda: {
        "short": [5, 10],
        "medium": [20, 30],
        "long": [60, 120],
    })
    RPS_PERIODS: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    MIN_TRAINING_SAMPLES: int = 200
    MIN_COVERAGE_RATE: float = 0.70
    VALIDATION_SPLIT_RATIO: float = 0.2


class FeatureCalculator(AnalyzerBase):
    """
    技术指标计算器

    计算各类技术指标，用于机器学习特征

    支持的指标:
        - 移动平均线: MA5, MA10, MA20, MA30, MA60, MA120
        - MACD: DIF, DEA, MACD柱
        - KDJ: K, D, J
        - RSI: 相对强弱指标
        - BOLL: 布林带上中下轨
        - DMI: PDI, MDI, ADX
        - 其他: OBV, VR, ROC, MAROC, BBI

    Attributes:
        config: 配置实例

    Example:
        >>> calculator = FeatureCalculator()
        >>> df = calculator.calculate_all_features(stock_data)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化特征计算器

        Args:
            config: 配置实例
        """
        super().__init__(config)

    @cached(config=CacheConfig(ttl=3600))
    def calculate_ma(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        计算移动平均线

        Args:
            df: 股票数据
            periods: 周期列表，默认[5, 10, 20, 30, 60, 120]

        Returns:
            添加了MA列的DataFrame
        """
        if periods is None:
            periods = [5, 10, 20, 30, 60, 120]

        result = df.copy()
        for period in periods:
            if len(result) >= period:
                result[f"MA{period}"] = result["close"].rolling(period).mean()
        return result

    @cached(config=CacheConfig(ttl=3600))
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算MACD指标

        Args:
            df: 股票数据

        Returns:
            添加了DIF, DEA, MACD列的DataFrame
        """
        result = df.copy()
        ema12 = result["close"].ewm(span=12).mean()
        ema26 = result["close"].ewm(span=26).mean()
        result["DIF"] = ema12 - ema26
        result["DEA"] = result["DIF"].ewm(span=9).mean()
        result["MACD"] = result["DIF"] - result["DEA"]
        return result

    @cached(config=CacheConfig(ttl=3600))
    def calculate_kdj(self, df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
        """
        计算KDJ指标

        Args:
            df: 股票数据
            n: 周期，默认9

        Returns:
            添加了K, D, J列的DataFrame
        """
        result = df.copy()
        low_min = result["low"].rolling(n).min()
        high_max = result["high"].rolling(n).max()
        rsv = (result["close"] - low_min) / (high_max - low_min) * 100
        result["K"] = rsv.ewm(com=2).mean()
        result["D"] = result["K"].ewm(com=2).mean()
        result["J"] = 3 * result["K"] - 2 * result["D"]
        return result

    @cached(config=CacheConfig(ttl=3600))
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算RSI指标

        Args:
            df: 股票数据
            period: 周期，默认14

        Returns:
            添加了RSI列的DataFrame
        """
        result = df.copy()
        delta = result["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        result["RSI"] = 100 - (100 / (1 + rs))
        return result

    @cached(config=CacheConfig(ttl=3600))
    def calculate_boll(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算布林带

        Args:
            df: 股票数据
            period: 周期，默认20

        Returns:
            添加了BOLL_UP, BOLL_MID, BOLL_LOW列的DataFrame
        """
        result = df.copy()
        result["BOLL_MID"] = result["close"].rolling(period).mean()
        bb_std = result["close"].rolling(period).std()
        result["BOLL_UP"] = result["BOLL_MID"] + 2 * bb_std
        result["BOLL_LOW"] = result["BOLL_MID"] - 2 * bb_std
        return result

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标

        Args:
            df: 包含OHLCV的股票数据

        Returns:
            添加了所有技术指标的DataFrame
        """
        result = df.copy()

        # 移动平均线
        result = self.calculate_ma(result)

        # MACD
        result = self.calculate_macd(result)

        # KDJ
        result = self.calculate_kdj(result)

        # RSI
        result = self.calculate_rsi(result)

        # 布林带
        result = self.calculate_boll(result)

        # 其他指标
        result = self._calculate_additional_features(result)

        return result

    def _calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算其他技术指标"""
        result = df.copy()

        # DMI
        result["+DM"] = result["high"].diff()
        result["-DM"] = -result["low"].diff()
        result["+DM"] = result["+DM"].where(result["+DM"] > 0, 0)
        result["-DM"] = result["-DM"].where(result["-DM"] > 0, 0)

        tr = pd.concat([
            result["high"] - result["low"],
            (result["high"] - result["close"].shift(1)).abs(),
            (result["low"] - result["close"].shift(1)).abs()
        ], axis=1).max(axis=1)

        result["+DI"] = (result["+DM"].rolling(14).mean() / tr.rolling(14).mean()) * 100
        result["-DI"] = (result["-DM"].rolling(14).mean() / tr.rolling(14).mean()) * 100
        result["PDI"] = result["+DI"]
        result["MDI"] = result["-DI"]
        result["ADX"] = abs(result["PDI"] - result["MDI"]) / (result["PDI"] + result["MDI"]) * 100

        # OBV
        result["OBV"] = (np.where(result["close"] > result["close"].shift(1), 1, -1) * result["volume"]).cumsum()

        # ROC
        result["ROC"] = result["close"].pct_change(5) * 100
        result["MAROC"] = result["ROC"].rolling(10).mean()

        # BBI
        if "MA10" in result.columns and "MA30" in result.columns:
            result["BBI"] = (result["MA5"] + result["MA10"] + result["MA20"] + result["MA30"]) / 4

        # 填充NaN
        result = result.fillna(method="bfill").fillna(0)

        return result


class MLTrainingDataGenerator(AnalyzerBase):
    """
    机器学习训练数据生成器

    生成用于模型训练的特征和标签数据

    Attributes:
        market_data_dir: 市场数据目录
        rps_data_dir: RPS数据目录
        config: 训练配置
        stock_data: 股票数据缓存
        rps_data: RPS数据缓存

    Example:
        >>> generator = MLTrainingDataGenerator()
        >>> success = generator.generate_training_data()
    """

    def __init__(
        self,
        market_data_dir: Optional[str] = None,
        rps_data_dir: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        初始化训练数据生成器

        Args:
            market_data_dir: 市场数据目录
            rps_data_dir: RPS数据目录
            config: 配置实例
        """
        super().__init__(config)

        # 确定目录路径
        if market_data_dir is None:
            market_data_dir = self.config.data.data_dir
        if rps_data_dir is None:
            rps_data_dir = self.config.data.rps_dir

        self.market_data_dir = market_data_dir
        self.rps_data_dir = rps_data_dir
        self.training_config = MLTrainingConfig()

        # 数据缓存
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.rps_data: Dict[str, Dict[str, pd.Series]] = {}

        # 特征计算器
        self.feature_calculator = FeatureCalculator(self.config)

    def backup_existing_training_data(self, output_dir: str = "training_data") -> None:
        """备份现有的训练数据文件"""
        if not os.path.exists(output_dir):
            return

        backup_extensions = [".npy", ".csv", ".pkl", ".txt", ".yaml"]
        files_to_backup = []

        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path) and any(file.endswith(ext) for ext in backup_extensions):
                files_to_backup.append(file)

        if not files_to_backup:
            self.logger.info(f"训练数据目录 {output_dir} 中没有需要备份的文件")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join("back", f"training_data_backup_{timestamp}")

        try:
            os.makedirs(backup_dir, exist_ok=True)
            self.logger.info(f"创建备份目录: {backup_dir}")

            for file in files_to_backup:
                src_path = os.path.join(output_dir, file)
                dst_path = os.path.join(backup_dir, file)
                shutil.move(src_path, dst_path)
                self.logger.info(f"备份文件: {file} -> {backup_dir}")

            self.logger.info(f"成功备份 {len(files_to_backup)} 个训练数据文件")

        except Exception as e:
            self.logger.error(f"备份训练数据文件时出错: {e}")
            raise

    def load_market_data(self, max_stocks: Optional[int] = None) -> bool:
        """
        加载股票市场数据

        Args:
            max_stocks: 最大加载股票数

        Returns:
            是否成功加载
        """
        self.logger.info("开始加载市场数据...")

        if not os.path.exists(self.market_data_dir):
            self.logger.error(f"目录不存在: {self.market_data_dir}")
            return False

        csv_files = [f for f in os.listdir(self.market_data_dir) if f.endswith(".csv")]
        if max_stocks:
            csv_files = csv_files[:max_stocks]

        for csv_file in csv_files:
            try:
                file_path = os.path.join(self.market_data_dir, csv_file)
                stock_code = csv_file.replace(".csv", "")

                df = pd.read_csv(file_path)
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").set_index("date")

                # 数据清洗
                df = df.dropna(subset=["close"])
                df = df[df["close"] > 0]

                if len(df) >= self.training_config.MIN_TRAINING_SAMPLES:
                    self.stock_data[stock_code] = df

            except Exception as e:
                self.logger.warning(f"加载 {csv_file} 失败: {e}")

        self.logger.info(f"成功加载 {len(self.stock_data)} 只股票数据")
        return len(self.stock_data) > 0

    def load_rps_data(self) -> bool:
        """加载RPS数据"""
        if not os.path.exists(self.rps_data_dir):
            self.logger.warning(f"RPS目录不存在: {self.rps_data_dir}")
            return False

        csv_files = [f for f in os.listdir(self.rps_data_dir) if f.endswith(".csv")]
        if not csv_files:
            self.logger.warning(f"在{self.rps_data_dir}中没有找到CSV文件")
            return False

        # 按日期排序，获取最新的文件
        csv_files.sort(reverse=True)
        latest_date = None
        for csv_file in csv_files:
            if "RPS" in csv_file:
                parts = csv_file.split("_")
                if len(parts) >= 2:
                    date_part = parts[1].replace(".csv", "")
                    if latest_date is None:
                        latest_date = date_part
                    break

        if latest_date is None:
            self.logger.warning("无法确定RPS文件的日期")
            return False

        self.logger.info(f"使用日期为 {latest_date} 的RPS数据")

        # 加载每个周期的RPS数据
        for period in self.training_config.RPS_PERIODS:
            rps_file = f"RPS{period}_{latest_date}.csv"
            file_path = os.path.join(self.rps_data_dir, rps_file)

            if not os.path.exists(file_path):
                self.logger.warning(f"RPS{period}文件不存在: {file_path}")
                continue

            try:
                df = pd.read_csv(file_path)
                df["date"] = pd.to_datetime(df["date"])

                for stock_code, group in df.groupby("stock_code"):
                    if stock_code not in self.rps_data:
                        self.rps_data[stock_code] = {}

                    rps_series = group.set_index("date")["rps"].sort_index()
                    self.rps_data[stock_code][f"rps{period}"] = rps_series

                self.logger.info(f"加载RPS{period}文件成功")

            except Exception as e:
                self.logger.warning(f"加载RPS{period}文件失败: {e}")

        return len(self.rps_data) > 0

    def generate_training_data(self, output_dir: str = "training_data") -> bool:
        """
        生成训练数据

        Args:
            output_dir: 输出目录

        Returns:
            是否成功生成
        """
        self.logger.info("开始生成训练数据...")

        # 备份现有数据
        self.backup_existing_training_data(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # 加载数据
        if not self.load_market_data():
            self.logger.error("市场数据加载失败")
            return False

        self.load_rps_data()

        # 生成特征和标签
        all_features = []
        all_labels: Dict[str, List] = {}
        all_info = []

        # 初始化标签字典
        for horizon in self.training_config.PREDICTION_HORIZONS:
            for threshold in self.training_config.RETURN_THRESHOLDS:
                label_name = f"return_{horizon}d_gt_{int(threshold*100)}pct"
                all_labels[label_name] = []
            all_labels[f"return_{horizon}d_continuous"] = []

        for stock_code, data in self.stock_data.items():
            try:
                # 计算技术指标
                data_with_features = self.feature_calculator.calculate_all_features(data.reset_index())

                # 添加RPS特征
                if stock_code in self.rps_data:
                    for period in self.training_config.RPS_PERIODS:
                        rps_key = f"rps{period}"
                        if rps_key in self.rps_data[stock_code]:
                            rps_series = self.rps_data[stock_code][rps_key]
                            rps_values = data_with_features["date"].map(
                                lambda x: self._get_rps_value(rps_series, x)
                            )
                            data_with_features[rps_key] = rps_values.fillna(50.0)

                # 生成标签
                for horizon in self.training_config.PREDICTION_HORIZONS:
                    future_returns = data_with_features["close"].shift(-horizon) / data_with_features["close"] - 1

                    # 二分类标签
                    for threshold in self.training_config.RETURN_THRESHOLDS:
                        label_name = f"return_{horizon}d_gt_{int(threshold*100)}pct"
                        labels = (future_returns > threshold).astype(int)
                        all_labels[label_name].extend(labels.tolist())

                    # 连续标签
                    continuous_name = f"return_{horizon}d_continuous"
                    all_labels[continuous_name].extend(future_returns.tolist())

                # 选择特征列
                feature_cols = self._select_feature_columns(data_with_features)
                features_data = data_with_features[feature_cols]

                # 添加到总数据集
                for idx in range(len(features_data)):
                    all_features.append(features_data.iloc[idx].values)
                    all_info.append({
                        "stock_code": stock_code,
                        "date": data_with_features.iloc[idx]["date"].strftime("%Y-%m-%d"),
                    })

                # 保存技术指标到原始CSV文件（供后续选股和分析使用）
                try:
                    output_file = os.path.join(self.market_data_dir, f"{stock_code}.csv")
                    data_with_features.to_csv(output_file, index=False)
                    self.logger.debug(f"保存技术指标到CSV: {stock_code}")
                except Exception as e:
                    self.logger.warning(f"保存 {stock_code} CSV失败: {e}")

            except Exception as e:
                self.logger.warning(f"处理股票 {stock_code} 失败: {e}")
                continue

        if not all_features:
            self.logger.error("未生成任何训练样本")
            return False

        # 转换为numpy数组并保存
        X = np.array(all_features)
        timestamp = datetime.now().strftime("%Y%m%d")

        # 保存特征
        np.save(os.path.join(output_dir, f"features_{timestamp}.npy"), X)

        # 保存标签
        for label_name, label_values in all_labels.items():
            if len(label_values) == len(X):
                y = np.array(label_values)
                np.save(os.path.join(output_dir, f"labels_{label_name}_{timestamp}.npy"), y)

        self.logger.info(f"训练数据生成完成: {len(X)} 样本, {X.shape[1]} 特征")

        return True

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """选择特征列"""
        feature_patterns = [
            "close",  # 基础价格
            "MA\\d+",  # 移动平均线
            "DIF", "DEA", "MACD",  # MACD
            "K", "D_", "J",  # KDJ (D_避免与DIF冲突)
            "RSI",  # RSI
            "BOLL_UP", "BOLL_MID", "BOLL_LOW",  # 布林带
            "PDI", "MDI", "ADX",  # DMI
            "OBV", "ROC", "MAROC",  # 其他
            "BBI",  # BBI
            "rps\\d+",  # RPS
        ]

        import re

        selected = []
        for pattern in feature_patterns:
            selected.extend([col for col in df.columns if re.match(pattern, col)])

        return list(set(selected))

    def _get_rps_value(self, rps_series: pd.Series, target_date) -> float:
        """获取RPS值"""
        try:
            valid_dates = rps_series.index[rps_series.index <= target_date]
            if len(valid_dates) > 0:
                return float(rps_series[valid_dates.max()])
        except Exception:
            pass
        return np.nan
