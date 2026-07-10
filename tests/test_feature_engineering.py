"""
特征工程单元测试

测试 FeatureCalculator, MLTrainingDataGenerator, MLTrainingConfig
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, '..')

import pandas as pd
import numpy as np

from src.data.feature_engineering import (
    FeatureCalculator,
    MLTrainingDataGenerator,
    MLTrainingConfig,
)


class TestMLTrainingConfig:
    """MLTrainingConfig单元测试"""

    def test_default_values(self):
        """测试默认配置值"""
        config = MLTrainingConfig()

        assert config.PREDICTION_HORIZONS == [5, 10, 20]
        assert config.RETURN_THRESHOLDS == [0.03, 0.05, 0.08]
        assert config.RPS_PERIODS == [5, 10, 20, 60]
        assert config.MIN_TRAINING_SAMPLES == 200
        assert config.MIN_COVERAGE_RATE == 0.70

    def test_feature_windows(self):
        """测试特征窗口配置"""
        config = MLTrainingConfig()

        assert "short" in config.FEATURE_WINDOWS
        assert "medium" in config.FEATURE_WINDOWS
        assert "long" in config.FEATURE_WINDOWS

        assert config.FEATURE_WINDOWS["short"] == [5, 10]
        assert config.FEATURE_WINDOWS["medium"] == [20, 30]
        assert config.FEATURE_WINDOWS["long"] == [60, 120]


class TestFeatureCalculator:
    """FeatureCalculator单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.calculator = FeatureCalculator()
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self, periods=100):
        """创建示例数据"""
        dates = pd.date_range("2024-01-01", periods=periods, freq="D")
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(periods) * 0.5)

        return pd.DataFrame({
            "date": dates,
            "open": close * (1 + np.random.randn(periods) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(periods)) * 0.01),
            "low": close * (1 - np.abs(np.random.randn(periods)) * 0.01),
            "close": close,
            "volume": np.random.randint(1000000, 10000000, periods),
        })

    def test_calculate_ma(self):
        """测试计算移动平均线"""
        result = self.calculator.calculate_ma(self.sample_data, periods=[5, 10, 20])

        assert "MA5" in result.columns
        assert "MA10" in result.columns
        assert "MA20" in result.columns

        # 检查MA值
        assert result["MA5"].dropna().iloc[-1] > 0
        assert result["MA20"].dropna().iloc[-1] > 0

    def test_calculate_macd(self):
        """测试计算MACD"""
        result = self.calculator.calculate_macd(self.sample_data)

        assert "DIF" in result.columns
        assert "DEA" in result.columns
        assert "MACD" in result.columns

        # MACD = DIF - DEA
        macd_check = (result["DIF"] - result["DEA"]).round(6)
        macd_actual = result["MACD"].round(6)
        assert (macd_check.fillna(0) == macd_actual.fillna(0)).all()

    def test_calculate_kdj(self):
        """测试计算KDJ"""
        result = self.calculator.calculate_kdj(self.sample_data)

        assert "K" in result.columns
        assert "D" in result.columns
        assert "J" in result.columns

        # KDJ值应该在0-100之间
        assert result["K"].dropna().between(0, 100).all()
        assert result["D"].dropna().between(0, 100).all()

    def test_calculate_rsi(self):
        """测试计算RSI"""
        result = self.calculator.calculate_rsi(self.sample_data)

        assert "RSI" in result.columns

        # RSI值应该在0-100之间
        assert result["RSI"].dropna().between(0, 100).all()

    def test_calculate_boll(self):
        """测试计算布林带"""
        result = self.calculator.calculate_boll(self.sample_data)

        assert "BOLL_UP" in result.columns
        assert "BOLL_MID" in result.columns
        assert "BOLL_LOW" in result.columns

        # 上轨 > 中轨 > 下轨
        valid_data = result.dropna()
        assert (valid_data["BOLL_UP"] > valid_data["BOLL_MID"]).all()
        assert (valid_data["BOLL_MID"] > valid_data["BOLL_LOW"]).all()

    def test_calculate_all_features(self):
        """测试计算所有指标"""
        result = self.calculator.calculate_all_features(self.sample_data)

        # 检查主要指标是否存在
        expected_indicators = [
            "MA5", "MA10", "MA20",
            "DIF", "DEA", "MACD",
            "K", "D", "J",
            "RSI",
            "BOLL_UP", "BOLL_MID", "BOLL_LOW",
            "PDI", "MDI", "ADX",
            "OBV", "ROC",
        ]

        for indicator in expected_indicators:
            assert indicator in result.columns, f"缺少指标: {indicator}"

        # 检查行数不变
        assert len(result) == len(self.sample_data)

    def test_calculate_all_features_preserves_original_data(self):
        """测试计算指标不修改原始数据"""
        original_close = self.sample_data["close"].copy()
        original_volume = self.sample_data["volume"].copy()

        result = self.calculator.calculate_all_features(self.sample_data)

        # 原始数据应该保持不变
        assert (result["close"] == original_close).all()
        assert (result["volume"] == original_volume).all()


class TestMLTrainingDataGenerator:
    """MLTrainingDataGenerator单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_market_dir = tempfile.mkdtemp()
        self.temp_rps_dir = tempfile.mkdtemp()
        self.temp_output_dir = tempfile.mkdtemp()

        self.generator = MLTrainingDataGenerator(
            market_data_dir=self.temp_market_dir,
            rps_data_dir=self.temp_rps_dir,
        )

        # 创建测试数据
        self._create_test_market_data()
        self._create_test_rps_data()

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_market_dir):
            shutil.rmtree(self.temp_market_dir)
        if os.path.exists(self.temp_rps_dir):
            shutil.rmtree(self.temp_rps_dir)
        if os.path.exists(self.temp_output_dir):
            shutil.rmtree(self.temp_output_dir)
        if os.path.exists("back"):
            backup_dirs = [d for d in os.listdir("back") if "training_data_backup" in d]
            for d in backup_dirs:
                shutil.rmtree(os.path.join("back", d))

    def _create_test_market_data(self):
        """创建测试市场数据"""
        stock_codes = ["sh.600000", "sh.600001", "sz.000001"]

        for stock_code in stock_codes:
            dates = pd.date_range("2024-01-01", periods=300, freq="D")
            np.random.seed(int(stock_code.split(".")[-1]))
            close = 100 + np.cumsum(np.random.randn(300) * 0.5)

            df = pd.DataFrame({
                "date": dates,
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.random.randint(1000000, 10000000, 300),
            })

            file_path = os.path.join(self.temp_market_dir, f"{stock_code}.csv")
            df.to_csv(file_path, index=False)

    def _create_test_rps_data(self):
        """创建测试RPS数据"""
        dates = pd.date_range("2024-01-01", periods=300, freq="D")

        for period in [5, 10, 20]:
            rps_values = np.random.rand(300) * 100

            df = pd.DataFrame({
                "date": dates,
                "stock_code": "sh.600000",
                "rps": rps_values,
            })

            file_path = os.path.join(
                self.temp_rps_dir,
                f"RPS{period}_20241224.csv"
            )
            df.to_csv(file_path, index=False)

    def test_load_market_data(self):
        """测试加载市场数据"""
        success = self.generator.load_market_data()

        assert success is True
        assert len(self.generator.stock_data) == 3

        for stock_code, df in self.generator.stock_data.items():
            assert len(df) >= 200

    def test_load_market_data_with_max_stocks(self):
        """测试限制加载股票数量"""
        success = self.generator.load_market_data(max_stocks=2)

        assert success is True
        assert len(self.generator.stock_data) == 2

    def test_load_rps_data(self):
        """测试加载RPS数据"""
        success = self.generator.load_rps_data()

        assert success is True
        assert len(self.generator.rps_data) > 0

    def test_load_rps_data_missing_directory(self):
        """测试RPS目录不存在"""
        generator = MLTrainingDataGenerator(
            market_data_dir=self.temp_market_dir,
            rps_data_dir="/nonexistent/dir",
        )

        success = generator.load_rps_data()
        # 应该返回False但不报错
        assert success is False

    def test_generate_training_data(self):
        """测试生成训练数据"""
        # 加载数据
        self.generator.load_market_data()
        self.generator.load_rps_data()

        # 生成训练数据
        success = self.generator.generate_training_data(
            output_dir=self.temp_output_dir
        )

        assert success is True

        # 检查生成的文件
        feature_files = [f for f in os.listdir(self.temp_output_dir) if f.startswith("features_")]
        assert len(feature_files) == 1

        label_files = [f for f in os.listdir(self.temp_output_dir) if f.startswith("labels_")]
        assert len(label_files) > 0

    def test_backup_existing_training_data(self):
        """测试备份现有训练数据"""
        # 创建一些现有文件
        for i in range(3):
            file_path = os.path.join(self.temp_output_dir, f"test_{i}.npy")
            np.save(file_path, np.array([1, 2, 3]))

        # 执行备份
        self.generator.backup_existing_training_data(self.temp_output_dir)

        # 检查文件已移动
        npy_files = [f for f in os.listdir(self.temp_output_dir) if f.endswith(".npy")]
        assert len(npy_files) == 0

    def test_select_feature_columns(self):
        """测试选择特征列"""
        # 创建包含所有指标的DataFrame
        df = self.generator.feature_calculator.calculate_all_features(
            self._create_sample_data(100)
        )

        feature_cols = self.generator._select_feature_columns(df)

        # 检查选择的列
        assert "close" in feature_cols
        assert any("MA" in col for col in feature_cols)
        assert any("DIF" in col or "DEA" in col or "MACD" in col for col in feature_cols)
        assert "RSI" in feature_cols

    def _create_sample_data(self, periods=100):
        """创建示例数据"""
        dates = pd.date_range("2024-01-01", periods=periods, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(periods) * 0.5)

        return pd.DataFrame({
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(1000000, 10000000, periods),
        })


class TestIntegration:
    """集成测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_market_dir = tempfile.mkdtemp()
        self.temp_rps_dir = tempfile.mkdtemp()
        self.temp_output_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """测试后清理"""
        for dir_path in [self.temp_market_dir, self.temp_rps_dir, self.temp_output_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        if os.path.exists("back"):
            backup_dirs = [d for d in os.listdir("back") if "training_data_backup" in d]
            for d in backup_dirs:
                shutil.rmtree(os.path.join("back", d))

    def test_full_feature_engineering_workflow(self):
        """测试完整特征工程流程"""
        # 创建生成器
        generator = MLTrainingDataGenerator(
            market_data_dir=self.temp_market_dir,
            rps_data_dir=self.temp_rps_dir,
        )

        # 创建测试数据
        dates = pd.date_range("2024-01-01", periods=300, freq="D")
        for i, stock_code in enumerate(["sh.600000", "sh.600001"]):
            np.random.seed(i)
            close = 100 + np.cumsum(np.random.randn(300) * 0.5)
            df = pd.DataFrame({
                "date": dates,
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.random.randint(1000000, 10000000, 300),
            })
            df.to_csv(os.path.join(self.temp_market_dir, f"{stock_code}.csv"), index=False)

        # 加载市场数据
        assert generator.load_market_data() is True

        # 计算特征
        for stock_code, df in generator.stock_data.items():
            features = generator.feature_calculator.calculate_all_features(df.reset_index())
            assert len(features) == len(df)

        print("完整特征工程流程测试通过")


if __name__ == "__main__":
    import pytest

    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
