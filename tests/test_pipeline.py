"""
数据管道单元测试

测试 DataPipeline, PipelineStep, PipelineResult, run_daily_pipeline
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime
from unittest import mock

sys.path.insert(0, '..')

from src.data.pipeline import (
    PipelineStep,
    PipelineResult,
    DataPipeline,
    run_daily_pipeline,
)


class TestPipelineStep:
    """PipelineStep枚举测试"""

    def test_pipeline_step_values(self):
        """测试PipelineStep枚举值"""
        assert PipelineStep.DOWNLOAD.value == "download"
        assert PipelineStep.RPS.value == "rps"
        assert PipelineStep.FEATURES.value == "features"
        assert PipelineStep.TRAIN.value == "train"

    def test_pipeline_step_uniqueness(self):
        """测试PipelineStep枚举唯一性"""
        values = [step.value for step in PipelineStep]
        assert len(values) == len(set(values))


class TestPipelineResult:
    """PipelineResult数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        result = PipelineResult()

        assert result.success is False
        assert result.step_results == {}
        assert result.errors == []
        assert result.start_time is None
        assert result.end_time is None

    def test_duration_property(self):
        """测试duration属性"""
        result = PipelineResult(
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 0, 30),
        )

        assert result.duration == 30.0

    def test_duration_with_none_times(self):
        """测试时间为None时的duration"""
        result = PipelineResult()
        assert result.duration is None

        result.start_time = datetime.now()
        assert result.duration is None

    def test_step_results_accumulation(self):
        """测试步骤结果累积"""
        result = PipelineResult()
        result.step_results["download"] = {"success": True, "count": 100}
        result.step_results["rps"] = {"success": True, "periods": 4}

        assert len(result.step_results) == 2
        assert result.step_results["download"]["count"] == 100

    def test_errors_accumulation(self):
        """测试错误累积"""
        result = PipelineResult()
        result.errors.append("Error 1")
        result.errors.append("Error 2")

        assert len(result.errors) == 2


class TestDataPipeline:
    """DataPipeline单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.pipeline = DataPipeline()

    def test_initialization(self):
        """测试初始化"""
        assert self.pipeline.config is not None
        assert len(self.pipeline.steps) == 4
        assert PipelineStep.DOWNLOAD in self.pipeline.steps
        assert PipelineStep.TRAIN in self.pipeline.steps

    def test_initialization_with_custom_steps(self):
        """测试自定义步骤初始化"""
        custom_steps = [PipelineStep.DOWNLOAD, PipelineStep.RPS]
        pipeline = DataPipeline(steps=custom_steps)

        assert len(pipeline.steps) == 2
        assert PipelineStep.DOWNLOAD in pipeline.steps
        assert PipelineStep.TRAIN not in pipeline.steps

    def test_initialization_creates_components(self):
        """测试初始化创建组件"""
        # 组件初始化为None，在需要时才创建
        assert self.pipeline.downloader is None
        assert self.pipeline.rps_calculator is None
        assert self.pipeline.feature_generator is None
        assert self.pipeline.model_trainer is None

    @mock.patch('src.data.pipeline.IncrementalDownloader')
    def test_execute_download(self, mock_downloader_class):
        """测试执行下载步骤"""
        # Mock downloader
        mock_downloader = mock.MagicMock()
        mock_downloader_class.return_value = mock_downloader
        mock_downloader.incremental_update.return_value = {
            "success_count": 10,
            "failed_count": 0
        }

        result = self.pipeline._execute_download(backfill_days=5)

        assert result["success"] is True
        assert "data" in result

    @mock.patch('src.data.pipeline.RPSPeriodsCalculator')
    def test_execute_rps(self, mock_calculator_class):
        """测试执行RPS步骤"""
        # Mock calculator
        mock_calculator = mock.MagicMock()
        mock_calculator_class.return_value = mock_calculator
        mock_calculator.calculate_multi_period_rps.return_value = {
            "RPS5": {},
            "RPS10": {},
        }

        result = self.pipeline._execute_rps(rps_periods=[5, 10])

        assert result["success"] is True
        assert "rps_data" in result

    @mock.patch('src.data.pipeline.MLTrainingDataGenerator')
    def test_execute_features(self, mock_generator_class):
        """测试执行特征工程步骤"""
        # Mock generator
        mock_generator = mock.MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_training_data.return_value = True

        result = self.pipeline._execute_features(max_stocks=100)

        assert result["success"] is True
        assert "output_dir" in result

    @mock.patch('src.data.pipeline.RandomForestTrainer')
    def test_execute_train(self, mock_trainer_class):
        """测试执行训练步骤"""
        # Mock trainer
        mock_trainer = mock.MagicMock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.train_models.return_value = {
            "model1": {"accuracy": 0.8}
        }

        result = self.pipeline._execute_train(target_label="return_5d_gt_5pct")

        assert result["success"] is True
        assert "training_results" in result

    def test_execute_unknown_step(self):
        """测试执行未知步骤"""
        # 使用enum中不存在的步骤
        result = self.pipeline._execute_step(PipelineStep.TRAIN)

        # 应该返回None（因为mock的trainer不存在）
        assert result is None

    @mock.patch('src.data.pipeline.IncrementalDownloader')
    def test_run_full_pipeline(self, mock_downloader_class):
        """测试执行完整管道"""
        # Mock所有组件
        mock_downloader = mock.MagicMock()
        mock_downloader_class.return_value = mock_downloader
        mock_downloader.incremental_update.return_value = {"success_count": 10}

        # Mock其他组件
        self.pipeline.rps_calculator = mock.MagicMock()
        self.pipeline.rps_calculator.calculate_multi_period_rps.return_value = {}

        self.pipeline.feature_generator = mock.MagicMock()
        self.pipeline.feature_generator.generate_training_data.return_value = True

        self.pipeline.model_trainer = mock.MagicMock()
        self.pipeline.model_trainer.train_models.return_value = {}

        result = self.pipeline.run_full_pipeline(backfill_days=5)

        # 验证结果
        assert isinstance(result, PipelineResult)
        assert result.start_time is not None
        assert result.end_time is not None
        assert result.duration is not None

    def test_run_from_step(self):
        """测试从指定步骤开始执行"""
        # Mock所有组件
        self.pipeline.rps_calculator = mock.MagicMock()
        self.pipeline.rps_calculator.calculate_multi_period_rps.return_value = {}

        self.pipeline.feature_generator = mock.MagicMock()
        self.pipeline.feature_generator.generate_training_data.return_value = True

        self.pipeline.model_trainer = mock.MagicMock()
        self.pipeline.model_trainer.train_models.return_value = {}

        result = self.pipeline.run_from_step(PipelineStep.RPS)

        assert isinstance(result, PipelineResult)
        # 应该只执行RPS、FEATURES、TRAIN三个步骤
        assert len(result.step_results) == 3

    def test_run_from_invalid_step(self):
        """测试从无效步骤开始执行"""
        # 创建只包含部分步骤的pipeline
        partial_pipeline = DataPipeline(steps=[PipelineStep.DOWNLOAD])

        result = partial_pipeline.run_from_step(PipelineStep.RPS)

        # 应该失败
        assert result.success is False
        assert len(result.errors) > 0

    def test_print_results(self, capsys):
        """测试打印结果"""
        result = PipelineResult(
            success=True,
            step_results={
                "download": {"success": True, "count": 100},
                "rps": {"success": True, "periods": 4},
            },
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 1, 30),
        )

        self.pipeline.print_results(result)

        captured = capsys.readouterr()
        assert "成功" in captured.out
        assert "90.00 秒" in captured.out


class TestRunDailyPipeline:
    """run_daily_pipeline便捷函数测试"""

    @mock.patch('src.data.pipeline.DataPipeline')
    def test_run_daily_pipeline_default_params(self, mock_pipeline_class):
        """测试使用默认参数运行每日管道"""
        # Mock pipeline
        mock_pipeline = mock.MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_full_pipeline.return_value = PipelineResult(success=True)

        result = run_daily_pipeline()

        assert isinstance(result, PipelineResult)
        mock_pipeline.run_full_pipeline.assert_called_once()

    @mock.patch('src.data.pipeline.DataPipeline')
    def test_run_daily_pipeline_with_params(self, mock_pipeline_class):
        """测试使用自定义参数运行每日管道"""
        # Mock pipeline
        mock_pipeline = mock.MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_full_pipeline.return_value = PipelineResult(success=True)

        result = run_daily_pipeline(backfill_days=10, forward_fill_date="2024-12-31")

        assert isinstance(result, PipelineResult)
        mock_pipeline.run_full_pipeline.assert_called_once_with(
            backfill_days=10,
            forward_fill_date="2024-12-31"
        )


class TestPipelineErrorHandling:
    """管道错误处理测试"""

    def setup_method(self):
        """测试前准备"""
        self.pipeline = DataPipeline()

    @mock.patch('src.data.pipeline.IncrementalDownloader')
    def test_step_failure_stops_pipeline(self, mock_downloader_class):
        """测试步骤失败时停止管道"""
        # Mock下载步骤失败
        mock_downloader = mock.MagicMock()
        mock_downloader_class.return_value = mock_downloader
        mock_downloader.incremental_update.return_value = None  # 失败

        result = self.pipeline.run_full_pipeline()

        assert result.success is False
        assert len(result.errors) > 0
        assert len(result.step_results) == 0  # 第一步就失败了

    @mock.patch('src.data.pipeline.IncrementalDownloader')
    def test_step_exception_caught(self, mock_downloader_class):
        """测试捕获步骤异常"""
        # Mock下载步骤抛出异常
        mock_downloader = mock.MagicMock()
        mock_downloader_class.return_value = mock_downloader
        mock_downloader.incremental_update.side_effect = Exception("Test error")

        result = self.pipeline.run_full_pipeline()

        assert result.success is False
        assert any("Test error" in error for error in result.errors)

    @mock.patch('src.data.pipeline.IncrementalDownloader')
    def test_continue_on_step_returning_false(self, mock_downloader_class):
        """测试步骤返回False时停止"""
        # Mock下载步骤返回不成功
        mock_downloader = mock.MagicMock()
        mock_downloader_class.return_value = mock_downloader
        mock_downloader.incremental_update.return_value = {"success": False}

        result = self.pipeline.run_full_pipeline()

        assert result.success is False


class TestPipelineConfiguration:
    """管道配置测试"""

    def test_custom_config_injection(self):
        """测试注入自定义配置"""
        from src.core.config import Config

        custom_config = Config()
        custom_config.data.data_dir = "./custom_data"

        pipeline = DataPipeline(config=custom_config)

        assert pipeline.config is custom_config
        assert pipeline.config.data.data_dir == "./custom_data"

    def test_step_filtering(self):
        """测试步骤过滤"""
        # 只执行下载和RPS步骤
        pipeline = DataPipeline(steps=[
            PipelineStep.DOWNLOAD,
            PipelineStep.RPS,
        ])

        assert len(pipeline.steps) == 2
        assert PipelineStep.FEATURES not in pipeline.steps
        assert PipelineStep.TRAIN not in pipeline.steps


if __name__ == "__main__":
    import pytest

    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
