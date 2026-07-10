#!/usr/bin/env python
"""
数据管道端到端测试

完整测试数据流程：下载 → RPS → 特征 → 训练
"""

import os
import sys
import tempfile
import shutil
import time
from datetime import datetime, timedelta

sys.path.insert(0, '..')

import numpy as np
import pandas as pd

from src.data.providers import BaostockProvider, CSVProvider
from src.data.downloaders import IncrementalDownloader
from src.data.rps_calculator import RPSPeriodsCalculator
from src.data.feature_engineering import FeatureCalculator, MLTrainingDataGenerator
from src.data.model_trainer import RandomForestTrainer
from src.data.pipeline import DataPipeline, PipelineStep, run_daily_pipeline


class TestEndToEnd:
    """端到端测试"""

    def __init__(self):
        """初始化测试环境"""
        # 创建临时目录
        self.base_dir = tempfile.mkdtemp(prefix="pipeline_test_")
        self.data_dir = os.path.join(self.base_dir, "data")
        self.rps_dir = os.path.join(self.base_dir, "rps")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.training_dir = os.path.join(self.base_dir, "training")

        # 创建目录
        for dir_path in [self.data_dir, self.rps_dir, self.models_dir, self.training_dir]:
            os.makedirs(dir_path, exist_ok=True)

        print(f"测试目录: {self.base_dir}")

    def cleanup(self):
        """清理测试环境"""
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
            print(f"已清理测试目录: {self.base_dir}")

    def setup_test_data(self):
        """设置测试数据"""
        print("\n1. 设置测试数据...")

        # 创建多只股票的测试数据 (至少10只以满足RPS计算要求)
        stock_codes = [
            "sh.600000", "sh.600001", "sh.600002", "sh.600003", "sh.600004",
            "sh.600005", "sh.600006", "sh.600007", "sh.600008", "sh.600009",
            "sz.000001", "sz.000002", "sz.000003"
        ]

        start_date = datetime(2024, 1, 1)

        for i, stock_code in enumerate(stock_codes):
            # 生成300天的数据
            dates = pd.date_range(start_date, periods=300, freq="D")

            # 使用不同的随机种子生成不同的价格走势
            np.random.seed(i + 100)
            base_price = 10 + np.random.rand() * 20

            # 生成价格序列（随机游走）
            returns = np.random.randn(300) * 0.02 + 0.0005  # 轻微上涨趋势
            prices = base_price * (1 + returns).cumprod()

            df = pd.DataFrame({
                "date": dates,
                "open": prices * (1 + np.random.randn(300) * 0.005),
                "high": prices * (1 + np.abs(np.random.randn(300)) * 0.01),
                "low": prices * (1 - np.abs(np.random.randn(300)) * 0.01),
                "close": prices,
                "preclose": prices * 0.99,
                "volume": np.random.randint(1000000, 10000000, 300),
                "amount": np.random.randint(10000000, 100000000, 300),
            })

            # 保存数据
            file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
            df.to_csv(file_path, index=False)
            print(f"  创建 {stock_code}: {len(df)} 行数据")

        print(f"  ✅ 创建了 {len(stock_codes)} 只股票的测试数据")
        return stock_codes

    def test_step1_csv_provider(self):
        """测试步骤1: CSV提供者"""
        print("\n2. 测试CSV提供者...")

        provider = CSVProvider(self.data_dir)

        # 测试exists
        exists = provider.exists("sh.600000")
        print(f"  检查文件存在: {exists}")
        assert exists is True

        # 测试load
        df = provider.load("sh.600000")
        print(f"  加载数据: {len(df)} 行, {len(df.columns)} 列")
        assert df is not None
        assert len(df) == 300

        # 测试validate
        is_valid = provider.validate_data(df)
        print(f"  数据验证: {is_valid}")
        assert is_valid is True

        print("  ✅ CSV提供者测试通过")
        return df

    def test_step2_rps_calculator(self):
        """测试步骤2: RPS计算器"""
        print("\n3. 测试RPS计算器...")

        calculator = RPSPeriodsCalculator(
            data_dir=self.data_dir,
            output_dir=self.rps_dir
        )

        # 计算多周期RPS
        print("  计算RPS指标...")
        result = calculator.calculate_multi_period_rps(periods=[5, 10, 20])

        assert result is not None
        assert "RPS5" in result
        assert "RPS10" in result
        assert "RPS20" in result

        # 打印结果
        for period_name, rps_data in result.items():
            print(f"  {period_name}: {len(rps_data)} 只股票")

        # 测试获取最新RPS
        latest_rps = calculator.get_latest_rps("sh.600000", period=20)
        print(f"  sh.600000 最新RPS20: {latest_rps}")
        assert latest_rps is not None
        assert 0 <= latest_rps <= 100

        print("  ✅ RPS计算器测试通过")
        return result

    def test_step3_feature_calculator(self, df):
        """测试步骤3: 特征计算器"""
        print("\n4. 测试特征计算器...")

        calculator = FeatureCalculator()

        # 计算所有指标
        print("  计算技术指标...")
        df_with_features = calculator.calculate_all_features(df)

        # 检查新增的列
        new_columns = [col for col in df_with_features.columns if col not in df.columns]
        print(f"  新增指标: {len(new_columns)} 个")

        # 检查关键指标
        key_indicators = ["MA5", "MA20", "DIF", "KDJ", "RSI", "BOLL_UP"]
        for indicator in key_indicators:
            if any(indicator in col for col in new_columns):
                print(f"    - {indicator}")

        assert len(new_columns) > 10
        assert len(df_with_features) == len(df)

        print("  ✅ 特征计算器测试通过")
        return df_with_features

    def test_step4_training_data_generator(self):
        """测试步骤4: 训练数据生成器"""
        print("\n5. 测试训练数据生成器...")

        generator = MLTrainingDataGenerator(
            market_data_dir=self.data_dir,
            rps_data_dir=self.rps_dir,
        )

        # 加载市场数据
        print("  加载市场数据...")
        success = generator.load_market_data()
        assert success is True
        print(f"  加载了 {len(generator.stock_data)} 只股票")

        # 加载RPS数据
        print("  加载RPS数据...")
        success = generator.load_rps_data()
        print(f"  RPS数据: {len(generator.rps_data)} 只股票")

        # 生成训练数据
        print("  生成训练数据...")
        success = generator.generate_training_data(output_dir=self.training_dir)
        assert success is True

        # 检查生成的文件
        feature_files = [f for f in os.listdir(self.training_dir) if f.startswith("features_")]
        label_files = [f for f in os.listdir(self.training_dir) if f.startswith("labels_")]

        print(f"  特征文件: {len(feature_files)} 个")
        print(f"  标签文件: {len(label_files)} 个")

        assert len(feature_files) >= 1
        assert len(label_files) >= 1

        print("  ✅ 训练数据生成器测试通过")

    def test_step5_model_trainer(self):
        """测试步骤5: 模型训练器"""
        print("\n6. 测试模型训练器...")

        trainer = RandomForestTrainer(
            training_data_dir=self.training_dir,
            model_save_dir=self.models_dir,
        )

        # 训练模型
        print("  训练模型...")
        results = trainer.train_models("return_5d_gt_5pct")

        assert results is not None

        # 打印结果
        for model_name, model_results in results.items():
            print(f"  模型: {model_name}")
            print(f"    准确率: {model_results.get('accuracy', 'N/A')}")
            print(f"    F1分数: {model_results.get('f1', 'N/A')}")
            print(f"    AUC: {model_results.get('auc', 'N/A')}")

        # 检查模型文件
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith(".pkl")]
        print(f"  模型文件: {len(model_files)} 个")

        assert len(model_files) >= 2  # 至少有模型和scaler

        print("  ✅ 模型训练器测试通过")
        return results

    def test_step6_data_pipeline(self):
        """测试步骤6: 完整数据管道"""
        print("\n7. 测试完整数据管道...")

        pipeline = DataPipeline()

        # 由于RPS计算需要较多时间，只测试部分步骤
        print("  执行下载步骤...")
        pipeline.downloader = IncrementalDownloader()

        # Mock实际下载，使用已有数据
        pipeline.downloader.get_existing_stock_list = lambda: [
            "sh.600000", "sh.600001", "sh.600002", "sz.000001", "sz.000002"
        ]

        result = pipeline._execute_download(stock_codes=["sh.600000"])

        print(f"  下载结果: {result.get('success')}")
        assert result is not None

        print("  ✅ 数据管道测试通过")

    def test_full_workflow(self):
        """测试完整工作流程"""
        print("\n" + "=" * 70)
        print("开始端到端测试")
        print("=" * 70)

        start_time = time.time()

        try:
            # 步骤1: 设置测试数据
            stock_codes = self.setup_test_data()

            # 步骤2: 测试CSV提供者
            df = self.test_step1_csv_provider()

            # 步骤3: 测试RPS计算器
            rps_results = self.test_step2_rps_calculator()

            # 步骤4: 测试特征计算器
            df_features = self.test_step3_feature_calculator(df)

            # 步骤5: 测试训练数据生成器
            self.test_step4_training_data_generator()

            # 步骤6: 测试模型训练器
            model_results = self.test_step5_model_trainer()

            # 步骤7: 测试数据管道
            self.test_step6_data_pipeline()

            elapsed = time.time() - start_time

            print("\n" + "=" * 70)
            print("端到端测试完成")
            print("=" * 70)
            print(f"总耗时: {elapsed:.2f} 秒")
            print(f"测试股票: {len(stock_codes)} 只")
            print(f"RPS周期: {len(rps_results)} 个")
            print(f"训练模型: {len(model_results)} 个")
            print("\n✅ 所有测试通过!")

            return True

        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    test = TestEndToEnd()

    try:
        success = test.test_full_workflow()
        return 0 if success else 1
    finally:
        test.cleanup()


if __name__ == "__main__":
    sys.exit(main())
