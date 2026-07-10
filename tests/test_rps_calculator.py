"""
RPS计算器单元测试

测试 RPSPeriodsCalculator
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta

sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import pickle

from src.data.rps_calculator import RPSPeriodsCalculator


class TestRPSPeriodsCalculator:
    """RPSPeriodsCalculator单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()

        # 创建计算器
        self.calculator = RPSPeriodsCalculator(
            data_dir=self.temp_dir,
            output_dir=self.output_dir
        )

        # 创建测试数据
        self._create_test_data()

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def _create_test_data(self):
        """创建测试用的股票数据"""
        # 创建5只股票的测试数据，每只300天
        stock_codes = ["sh.600000", "sh.600001", "sh.600002", "sz.000001", "sz.000002"]
        start_date = datetime(2024, 1, 1)

        for stock_code in stock_codes:
            dates = pd.date_range(start_date, periods=300, freq="D")

            # 生成模拟价格数据
            np.random.seed(int(stock_code.split(".")[-1]))
            base_price = 10 + np.random.rand() * 20
            returns = np.random.randn(300) * 0.02
            prices = base_price * (1 + returns).cumprod()

            df = pd.DataFrame({
                "date": dates,
                "close": prices,
                "open": prices * (1 + np.random.randn(300) * 0.005),
                "high": prices * (1 + np.abs(np.random.randn(300)) * 0.01),
                "low": prices * (1 - np.abs(np.random.randn(300)) * 0.01),
                "volume": np.random.randint(1000000, 10000000, 300),
            })

            file_path = os.path.join(self.temp_dir, f"{stock_code}.csv")
            df.to_csv(file_path, index=False)

    def test_load_stock_data(self):
        """测试加载股票数据"""
        df = self.calculator.load_stock_data("sh.600000.csv")

        assert df is not None
        assert len(df) == 300
        assert "date" in df.columns
        assert "close" in df.columns
        assert df["close"].min() > 0

    def test_load_stock_data_insufficient_data(self):
        """测试加载数据不足的股票"""
        # 创建数据不足的文件
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "close": [10] * 100,
        })
        file_path = os.path.join(self.temp_dir, "sh.600999.csv")
        df.to_csv(file_path, index=False)

        result = self.calculator.load_stock_data("sh.600999.csv")
        assert result is None

    def test_load_stock_data_invalid_file(self):
        """测试加载无效文件"""
        result = self.calculator.load_stock_data("nonexistent.csv")
        assert result is None

    def test_load_all_stock_data(self):
        """测试加载所有股票数据"""
        success = self.calculator.load_all_stock_data()

        assert success is True
        assert len(self.calculator.stock_data) == 5

        for stock_code, df in self.calculator.stock_data.items():
            assert len(df) >= 250

    def test_load_all_stock_data_empty_directory(self):
        """测试空目录"""
        empty_dir = tempfile.mkdtemp()
        calculator = RPSPeriodsCalculator(data_dir=empty_dir)

        success = calculator.load_all_stock_data()
        assert success is False

        shutil.rmtree(empty_dir)

    def test_create_aligned_price_matrix(self):
        """测试创建对齐的价格矩阵"""
        # 先加载数据
        self.calculator.load_all_stock_data()

        # 创建价格矩阵
        price_matrix, date_index, stock_codes = self.calculator.create_aligned_price_matrix(
            max_period=250
        )

        assert price_matrix is not None
        assert date_index is not None
        assert stock_codes is not None

        assert len(stock_codes) == 5
        assert price_matrix.shape[0] == len(date_index)
        assert price_matrix.shape[1] == len(stock_codes)

    def test_calculate_returns(self):
        """测试计算收益率"""
        # 创建简单价格矩阵
        prices = np.array([
            [10, 20, 30],
            [11, 22, 33],
            [12, 24, 36],
            [13, 26, 39],
        ])

        returns = self.calculator.calculate_returns(prices, period_days=1)

        # 检查第一行应该是NaN（没有前一天数据）
        assert np.isnan(returns[0, 0])

        # 检查第二行及以后应该有值
        assert not np.isnan(returns[1, 0])
        assert abs(returns[1, 0] - 0.1) < 0.01  # (11-10)/10 = 0.1

    def test_calculate_rps_for_period(self):
        """测试计算指定周期的RPS"""
        # 加载数据并创建矩阵
        self.calculator.load_all_stock_data()
        price_matrix, date_index, stock_codes = self.calculator.create_aligned_price_matrix(
            max_period=60
        )

        # 保存矩阵信息
        self.calculator.price_matrix = price_matrix
        self.calculator.date_index = date_index
        self.calculator.stock_codes = stock_codes

        # 计算收益率
        returns_matrix = self.calculator.calculate_returns(price_matrix, 20)

        # 计算RPS
        rps_data = self.calculator.calculate_rps_for_period(returns_matrix, 20)

        assert rps_data is not None
        assert len(rps_data) > 0

        # 检查RPS值范围
        for stock_code, stock_rps in rps_data.items():
            for record in stock_rps:
                assert 0 <= record["rps"] <= 100

    def test_calculate_multi_period_rps(self):
        """测试计算多周期RPS"""
        result = self.calculator.calculate_multi_period_rps(periods=[5, 10, 20])

        assert result is not None
        assert "RPS5" in result
        assert "RPS10" in result
        assert "RPS20" in result

        # 检查每个周期都有数据
        for period_name, rps_data in result.items():
            assert len(rps_data) > 0

    def test_save_results(self):
        """测试保存RPS结果"""
        # 创建模拟结果
        all_rps_data = {
            "RPS5": {
                "sh.600000": [{"date": "2024-01-01", "rps": 80.5}]
            },
            "RPS10": {
                "sh.600000": [{"date": "2024-01-01", "rps": 75.0}]
            }
        }

        # 保存结果
        self.calculator.save_results(all_rps_data)

        # 检查pickle文件
        pickle_files = [f for f in os.listdir(self.output_dir) if f.endswith(".pkl")]
        assert len(pickle_files) == 1

        # 检查CSV文件
        csv_files = [f for f in os.listdir(self.output_dir) if f.endswith(".csv")]
        assert len(csv_files) == 2  # RPS5和RPS10

    def test_get_latest_rps(self):
        """测试获取最新RPS值"""
        # 先计算并保存RPS
        self.calculator.calculate_multi_period_rps(periods=[20])

        # 获取最新RPS
        latest_rps = self.calculator.get_latest_rps("sh.600000", period=20)

        assert latest_rps is not None
        assert 0 <= latest_rps <= 100

    def test_get_latest_rps_nonexistent_stock(self):
        """测试获取不存在股票的RPS"""
        # 先计算并保存RPS
        self.calculator.calculate_multi_period_rps(periods=[20])

        # 获取不存在股票的RPS
        latest_rps = self.calculator.get_latest_rps("sh.999999", period=20)
        assert latest_rps is None

    def test_get_top_stocks(self):
        """测试获取RPS高的股票"""
        # 先计算并保存RPS
        self.calculator.calculate_multi_period_rps(periods=[20])

        # 获取RPS>50的股票
        top_stocks = self.calculator.get_top_stocks(period=20, threshold=50, limit=10)

        assert isinstance(top_stocks, list)
        assert len(top_stocks) > 0

        # 所有返回的股票都应该有数据
        for stock_code in top_stocks:
            assert stock_code in self.calculator.stock_data

    def test_backup_existing_results(self):
        """测试备份现有结果"""
        # 创建一些现有文件
        for i in range(3):
            file_path = os.path.join(self.output_dir, f"test_{i}.csv")
            with open(file_path, "w") as f:
                f.write("test")

        # 执行备份
        self.calculator.backup_existing_results()

        # 检查文件已移动到备份目录
        csv_files = [f for f in os.listdir(self.output_dir) if f.endswith(".csv")]
        assert len(csv_files) == 0

        # 检查备份目录存在
        backup_dirs = [d for d in os.listdir("back") if d.startswith("rps_results_backup")]
        assert len(backup_dirs) == 1


class TestRPSValidation:
    """RPS计算验证测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()

        # 创建计算器
        self.calculator = RPSPeriodsCalculator(
            data_dir=self.temp_dir,
            output_dir=self.output_dir
        )

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_rps_values_distribution(self):
        """测试RPS值分布是否合理"""
        # 创建测试数据，使某只股票明显表现更好
        self._create_controlled_test_data()

        result = self.calculator.calculate_multi_period_rps(periods=[20])

        # 检查表现好的股票RPS更高
        rps20 = result["RPS20"]

        # sh.600000应该有较高的RPS（涨幅最大）
        rps_600000 = [r["rps"] for r in rps20.get("sh.600000", [])]
        rps_600001 = [r["rps"] for r in rps20.get("sh.600001", [])]

        if rps_600000 and rps_600001:
            # 最后一个RPS值，sh.600000应该更高
            assert rps_600000[-1] > rps_600001[-1]

    def _create_controlled_test_data(self):
        """创建可控的测试数据"""
        stock_codes = ["sh.600000", "sh.600001", "sh.600002"]
        start_date = datetime(2024, 1, 1)

        growth_rates = [0.002, 0.001, 0.0005]  # 不同涨幅

        for stock_code, growth_rate in zip(stock_codes, growth_rates):
            dates = pd.date_range(start_date, periods=300, freq="D")

            # 生成固定涨幅的价格
            returns = np.ones(300) * (1 + growth_rate)
            returns += np.random.randn(300) * 0.005  # 添加一些随机波动
            prices = 10 * returns.cumprod()

            df = pd.DataFrame({
                "date": dates,
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "volume": [1000000] * 300,
            })

            file_path = os.path.join(self.temp_dir, f"{stock_code}.csv")
            df.to_csv(file_path, index=False)


if __name__ == "__main__":
    import pytest

    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
