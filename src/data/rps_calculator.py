"""
RPS(相对价格强度)计算器模块

计算多周期RPS值，用于评估股票相对市场表现
"""

import os
import pickle
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)


class RPSPeriodsCalculator(AnalyzerBase):
    """
    RPS周期数据计算器

    支持计算5、10、20、60、120、250天RPS周期

    RPS(Relative Price Strength)相对价格强度:
    - RPS值范围: 0-100
    - RPS=90 表示该股票表现优于90%的股票
    - RPS越高，股票相对表现越好

    Attributes:
        data_dir: 股票数据目录
        output_dir: RPS结果输出目录
        stock_data: 股票数据缓存
        price_matrix: 价格矩阵
        date_index: 日期索引
        stock_codes: 股票代码列表

    Example:
        >>> calculator = RPSPeriodsCalculator()
        >>> result = calculator.calculate_multi_period_rps([5, 10, 20, 60])
        >>> print(result['RPS5']['sh.600000'][-1])  # 打印某股票的最新RPS5值
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        初始化RPS周期计算器

        Args:
            data_dir: 数据目录路径，None则使用配置中的data_dir
            output_dir: 输出目录路径，默认为rps_results
            config: 配置实例
        """
        super().__init__(config)

        # 确定目录路径
        if data_dir is None:
            data_dir = self.config.data.data_dir
        if output_dir is None:
            output_dir = self.config.data.rps_dir

        self.data_dir = data_dir
        self.output_dir = output_dir

        # 数据缓存
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.price_matrix: Optional[np.ndarray] = None
        self.date_index: Optional[List] = None
        self.stock_codes: Optional[List[str]] = None

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"RPS计算器初始化完成，数据目录: {self.data_dir}")

    def load_stock_data(self, stock_file: str) -> Optional[pd.DataFrame]:
        """
        加载单个股票数据

        Args:
            stock_file: 股票文件名

        Returns:
            股票数据DataFrame，数据不足或无效返回None
        """
        file_path = os.path.join(self.data_dir, stock_file)

        try:
            df = pd.read_csv(file_path)

            # 检查必需列
            required_columns = ["date", "close"]
            if not all(col in df.columns for col in required_columns):
                return None

            # 转换日期格式
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            # 检查数据量
            if len(df) < 250:  # 至少需要250天数据
                return None

            # 去除无效价格
            df = df[df["close"] > 0]

            if len(df) < 250:
                return None

            return df

        except Exception as e:
            self.logger.warning(f"加载 {stock_file} 失败: {e}")
            return None

    def load_all_stock_data(self) -> bool:
        """
        加载所有股票数据

        Returns:
            是否成功加载数据
        """
        self.logger.info("开始加载股票数据...")

        if not os.path.exists(self.data_dir):
            self.logger.error(f"数据目录不存在: {self.data_dir}")
            return False

        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]

        if not csv_files:
            self.logger.error(f"在 {self.data_dir} 中没有找到CSV文件")
            return False

        self.logger.info(f"发现 {len(csv_files)} 个CSV文件")

        success_count = 0

        for csv_file in csv_files:
            stock_code = csv_file.replace(".csv", "")
            df = self.load_stock_data(csv_file)

            if df is not None:
                self.stock_data[stock_code] = df
                success_count += 1

        self.logger.info(f"成功加载 {success_count} 只股票数据")

        if success_count == 0:
            self.logger.error("没有成功加载任何股票数据")
            return False

        return True

    def create_aligned_price_matrix(
        self, max_period: int = 250
    ) -> tuple[Optional[np.ndarray], Optional[List], Optional[List[str]]]:
        """
        创建对齐的价格矩阵

        Args:
            max_period: 最大周期(用于确定最小数据长度)

        Returns:
            (价格矩阵, 日期列表, 股票代码列表) 元组
        """
        self.logger.info("创建价格矩阵...")

        if not self.stock_data:
            self.logger.error("没有股票数据")
            return None, None, None

        # 获取所有日期的并集
        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df["date"])

        # 排序日期
        all_dates = sorted(list(all_dates))

        # 过滤股票：确保有足够的数据
        valid_stocks = {}
        for stock_code, df in self.stock_data.items():
            if len(df) >= max_period:
                valid_stocks[stock_code] = df

        if not valid_stocks:
            self.logger.error(f"没有股票具有足够的数据(至少{max_period}天)")
            return None, None, None

        self.logger.info(f"有效股票数量: {len(valid_stocks)}")
        self.logger.info(f"日期范围: {all_dates[0]} 到 {all_dates[-1]}")
        self.logger.info(f"总交易日数: {len(all_dates)}")

        # 创建价格矩阵
        stock_codes = list(valid_stocks.keys())
        price_matrix = np.full((len(all_dates), len(stock_codes)), np.nan)

        for stock_idx, stock_code in enumerate(stock_codes):
            df = valid_stocks[stock_code]

            # 为每个日期填充价格
            for _, row in df.iterrows():
                date_idx = all_dates.index(row["date"])
                price_matrix[date_idx, stock_idx] = row["close"]

        self.logger.info(f"价格矩阵创建完成: {price_matrix.shape}")

        return price_matrix, all_dates, stock_codes

    def calculate_returns(
        self, price_matrix: np.ndarray, period_days: int
    ) -> np.ndarray:
        """
        计算指定周期的收益率矩阵

        Args:
            price_matrix: 价格矩阵
            period_days: 计算周期

        Returns:
            收益率矩阵
        """
        num_dates, num_stocks = price_matrix.shape
        returns_matrix = np.full((num_dates, num_stocks), np.nan)

        # 从period_days开始计算收益率
        for i in range(period_days, num_dates):
            current_prices = price_matrix[i, :]
            past_prices = price_matrix[i - period_days, :]

            # 计算收益率
            valid_mask = ~(
                np.isnan(current_prices) | np.isnan(past_prices) | (past_prices == 0)
            )
            returns_matrix[i, valid_mask] = (
                (current_prices[valid_mask] - past_prices[valid_mask])
                / past_prices[valid_mask]
            )

        return returns_matrix

    def calculate_rps_for_period(
        self, returns_matrix: np.ndarray, period_days: int
    ) -> Dict[str, List[Dict]]:
        """
        计算指定周期的RPS值

        Args:
            returns_matrix: 收益率矩阵
            period_days: RPS周期

        Returns:
            RPS数据字典 {股票代码: [{"date":..., "rps":...}, ...]}
        """
        self.logger.info(f"计算RPS{period_days}...")

        num_dates, num_stocks = returns_matrix.shape
        rps_matrix = np.full((num_dates, num_stocks), np.nan)

        # 从period_days开始计算RPS
        for date_idx in range(period_days, num_dates):
            current_returns = returns_matrix[date_idx, :]

            # 移除NaN值
            valid_mask = ~np.isnan(current_returns)
            if valid_mask.sum() < 10:  # 至少需要10只股票
                continue

            valid_returns = current_returns[valid_mask]

            # 计算RPS
            for stock_idx in range(num_stocks):
                if not valid_mask[stock_idx]:
                    continue

                stock_return = current_returns[stock_idx]

                # 计算超越的股票数量
                better_count = (valid_returns < stock_return).sum()
                total_count = len(valid_returns)

                # RPS = 超越股票数 / 总股票数 * 100
                rps_value = (better_count / total_count) * 100
                rps_matrix[date_idx, stock_idx] = rps_value

        # 转换为字典格式
        rps_data = {}
        for stock_idx, stock_code in enumerate(self.stock_codes):
            stock_rps = []
            for date_idx in range(num_dates):
                if not np.isnan(rps_matrix[date_idx, stock_idx]):
                    stock_rps.append(
                        {
                            "date": self.date_index[date_idx].strftime("%Y-%m-%d"),
                            "rps": round(rps_matrix[date_idx, stock_idx], 2),
                        }
                    )

            if stock_rps:  # 只保存有数据的股票
                rps_data[stock_code] = stock_rps

        return rps_data

    def backup_existing_results(self) -> None:
        """备份现有的RPS结果文件"""
        if not os.path.exists(self.output_dir):
            return

        # 检查是否有文件需要备份
        existing_files = [
            f
            for f in os.listdir(self.output_dir)
            if f.endswith(".csv") or f.endswith(".pkl") or f.endswith(".txt")
        ]

        if not existing_files:
            self.logger.info("没有需要备份的文件")
            return

        # 创建备份目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join("back", f"rps_results_backup_{timestamp}")

        try:
            os.makedirs(backup_dir, exist_ok=True)
            self.logger.info(f"创建备份目录: {backup_dir}")

            # 移动现有文件到备份目录
            moved_count = 0
            for file_name in existing_files:
                src_path = os.path.join(self.output_dir, file_name)
                dst_path = os.path.join(backup_dir, file_name)
                shutil.move(src_path, dst_path)
                moved_count += 1

            self.logger.info(f"成功备份 {moved_count} 个文件到: {backup_dir}")

        except Exception as e:
            self.logger.error(f"备份文件时发生错误: {e}")

    def save_results(self, all_rps_data: Dict[str, Dict]) -> None:
        """
        保存RPS计算结果

        Args:
            all_rps_data: 所有周期的RPS数据
        """
        # 备份现有结果
        self.backup_existing_results()

        date_str = datetime.now().strftime("%Y%m%d")

        # 保存为pickle文件
        pickle_file = os.path.join(self.output_dir, f"rps_periods_{date_str}.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(all_rps_data, f)
        self.logger.info(f"RPS数据保存到: {pickle_file}")

        # 保存为CSV文件(每个周期单独保存)
        for period_name, rps_data in all_rps_data.items():
            # 从period_name中提取周期数字，格式：RPS5 -> 5
            period_num = period_name.replace("RPS", "")
            csv_file = os.path.join(self.output_dir, f"RPS{period_num}_{date_str}.csv")

            # 转换为DataFrame格式
            all_records = []
            for stock_code, stock_rps in rps_data.items():
                for record in stock_rps:
                    all_records.append(
                        {"stock_code": stock_code, "date": record["date"], "rps": record["rps"]}
                    )

            if all_records:
                df = pd.DataFrame(all_records)
                df.to_csv(csv_file, index=False)
                self.logger.info(f"{period_name} CSV保存到: {csv_file}")

    def calculate_multi_period_rps(
        self, periods: List[int] = None
    ) -> Optional[Dict[str, Dict]]:
        """
        计算多周期RPS

        Args:
            periods: RPS计算周期列表，默认[5, 10, 20, 60, 120, 250]

        Returns:
            各周期的RPS数据字典，格式: {"RPS5": {stock_code: [...]}, ...}
        """
        if periods is None:
            periods = [5, 10, 20, 60, 120, 250]

        self.logger.info(f"=== 开始多周期RPS计算 ===")
        self.logger.info(f"计算周期: {periods}")
        start_time = time.time()

        # 1. 加载数据
        if not self.stock_data:
            if not self.load_all_stock_data():
                return None

        if not self.stock_data:
            self.logger.error("没有可用的股票数据")
            return None

        # 2. 创建价格矩阵
        max_period = max(periods)
        if self.price_matrix is None:
            (
                price_matrix,
                date_index,
                stock_codes,
            ) = self.create_aligned_price_matrix(max_period)
        else:
            price_matrix, date_index, stock_codes = (
                self.price_matrix,
                self.date_index,
                self.stock_codes,
            )

        if price_matrix is None:
            self.logger.error("价格矩阵创建失败")
            return None

        # 保存矩阵信息
        self.price_matrix = price_matrix
        self.date_index = date_index
        self.stock_codes = stock_codes

        # 3. 计算各周期RPS
        all_rps_data = {}

        for period in periods:
            self.logger.info(f"\n处理RPS{period}周期...")

            # 计算收益率
            returns_matrix = self.calculate_returns(price_matrix, period)

            # 计算RPS
            rps_data = self.calculate_rps_for_period(returns_matrix, period)

            if rps_data:
                all_rps_data[f"RPS{period}"] = rps_data
                self.logger.info(f"RPS{period} 计算完成，包含 {len(rps_data)} 只股票")
            else:
                self.logger.error(f"RPS{period} 计算失败")

        # 4. 保存结果
        if all_rps_data:
            self.save_results(all_rps_data)

        total_time = time.time() - start_time
        self.logger.info(f"\n=== 多周期RPS计算完成 ===")
        self.logger.info(f"总耗时: {total_time:.2f} 秒")
        self.logger.info(f"计算周期: {list(all_rps_data.keys())}")

        return all_rps_data

    def get_latest_rps(
        self, stock_code: str, period: int = 20
    ) -> Optional[float]:
        """
        获取指定股票的最新RPS值

        Args:
            stock_code: 股票代码
            period: RPS周期

        Returns:
            最新RPS值，未找到返回None
        """
        # 查找最新的RPS文件
        csv_files = [
            f
            for f in os.listdir(self.output_dir)
            if f.startswith(f"RPS{period}_") and f.endswith(".csv")
        ]

        if not csv_files:
            return None

        # 使用最新的文件
        latest_file = sorted(csv_files)[-1]
        file_path = os.path.join(self.output_dir, latest_file)

        try:
            df = pd.read_csv(file_path)

            # 筛选指定股票
            stock_data = df[df["stock_code"] == stock_code]

            if stock_data.empty:
                return None

            # 返回最新的RPS值
            latest_rps = stock_data.iloc[-1]["rps"]
            return float(latest_rps)

        except Exception as e:
            self.logger.error(f"读取RPS文件失败: {e}")
            return None

    def get_top_stocks(
        self, period: int = 20, threshold: float = 80, limit: int = 50
    ) -> List[str]:
        """
        获取RPS高于指定阈值的股票列表

        Args:
            period: RPS周期
            threshold: RPS阈值
            limit: 最大返回数量

        Returns:
            股票代码列表
        """
        # 查找最新的RPS文件
        csv_files = [
            f
            for f in os.listdir(self.output_dir)
            if f.startswith(f"RPS{period}_") and f.endswith(".csv")
        ]

        if not csv_files:
            return []

        # 使用最新的文件
        latest_file = sorted(csv_files)[-1]
        file_path = os.path.join(self.output_dir, latest_file)

        try:
            df = pd.read_csv(file_path)

            # 获取每只股票的最新RPS
            latest_rps = df.groupby("stock_code").last().reset_index()

            # 筛选高于阈值的股票
            top_stocks = latest_rps[latest_rps["rps"] >= threshold]

            # 按RPS降序排序
            top_stocks = top_stocks.sort_values("rps", ascending=False)

            # 返回股票代码列表
            return top_stocks.head(limit)["stock_code"].tolist()

        except Exception as e:
            self.logger.error(f"读取RPS文件失败: {e}")
            return []
