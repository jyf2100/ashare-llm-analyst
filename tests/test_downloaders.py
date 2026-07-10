"""
数据下载器单元测试

测试 AntiCrawlerController, RequestRetryManager, IncrementalDownloader
"""

import os
import sys
import time
from unittest import mock
from datetime import datetime, timedelta

sys.path.insert(0, '..')

import pandas as pd
import numpy as np

from src.data.downloaders import (
    AntiCrawlerController,
    RequestRetryManager,
    IncrementalDownloader,
)
from src.core.exceptions import DataFetchError


class TestAntiCrawlerController:
    """AntiCrawlerController单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.controller = AntiCrawlerController()

    def test_initial_state(self):
        """测试初始状态"""
        assert self.controller.request_count == 0
        assert self.controller.error_count == 0
        assert self.controller.last_request_time is None

    def test_calculate_delay(self):
        """测试延时计算"""
        # 初始延时应该是最小延时
        delay = self.controller.calculate_delay()
        assert delay >= self.controller.min_delay
        assert delay <= self.controller.max_delay

        # 模拟有错误时延时应该增加
        self.controller.error_count = 5
        delay_with_errors = self.controller.calculate_delay()
        assert delay_with_errors >= delay

    def test_before_request(self):
        """测试请求前处理"""
        start_time = time.time()
        self.controller.before_request()
        elapsed = time.time() - start_time

        # 应该等待至少min_delay
        assert elapsed >= self.controller.min_delay

    def test_after_request_success(self):
        """测试请求后成功处理"""
        self.controller.after_request(success=True)
        assert self.controller.request_count == 1
        assert self.controller.error_count == 0

    def test_after_request_error(self):
        """测试请求后错误处理"""
        self.controller.after_request(success=False, error_type="network")
        assert self.controller.request_count == 1
        assert self.controller.error_count == 1

    def test_get_error_statistics(self):
        """测试获取错误统计"""
        # 模拟各种错误
        for _ in range(3):
            self.controller.after_request(success=False, error_type="network")
        for _ in range(2):
            self.controller.after_request(success=False, error_type="timeout")

        stats = self.controller.get_error_statistics()
        assert stats.get("network", 0) == 3
        assert stats.get("timeout", 0) == 2
        assert stats["total"] == 5

    def test_reset(self):
        """测试重置"""
        self.controller.request_count = 10
        self.controller.error_count = 5
        self.controller.reset()

        assert self.controller.request_count == 0
        assert self.controller.error_count == 0


class TestRequestRetryManager:
    """RequestRetryManager单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.retry_manager = RequestRetryManager()

    def test_should_retry_with_remaining_attempts(self):
        """测试有剩余重试次数时应该重试"""
        # 参数错误不应该重试
        assert self.retry_manager.should_retry("parameter", 0, 3) is False
        assert self.retry_manager.should_retry("parameter", 1, 3) is False

        # 网络错误应该重试
        assert self.retry_manager.should_retry("network", 0, 3) is True
        assert self.retry_manager.should_retry("network", 2, 3) is True

        # 超过最大重试次数不应该重试
        assert self.retry_manager.should_retry("network", 3, 3) is False

    def test_calculate_backoff_delay(self):
        """测试退避延时计算"""
        delay1 = self.retry_manager.calculate_backoff_delay(0)
        delay2 = self.retry_manager.calculate_backoff_delay(1)
        delay3 = self.retry_manager.calculate_backoff_delay(2)

        # 延时应该随着重试次数增加
        assert delay2 > delay1
        assert delay3 > delay2

        # 延时应该在合理范围内
        assert delay1 >= 1
        assert delay3 <= 60

    def test_classify_error(self):
        """测试错误分类"""
        assert self.retry_manager.classify_error("网络超时") == "network"
        assert self.retry_manager.classify_error("连接超时") == "timeout"
        assert self.retry_manager.classify_error("参数错误") == "parameter"
        assert self.retry_manager.classify_error("其他") == "unknown"

    @mock.patch('time.sleep')
    def test_retry_request_success_on_first_try(self, mock_sleep):
        """测试第一次就成功"""
        func = mock.MagicMock(return_value=("success", None))

        result, error_type = self.retry_manager.retry_request(func, "test_arg")

        assert result == "success"
        assert error_type is None
        func.assert_called_once_with("test_arg")
        mock_sleep.assert_not_called()

    @mock.patch('time.sleep')
    def test_retry_request_success_after_retry(self, mock_sleep):
        """测试重试后成功"""
        # 第一次失败，第二次成功
        func = mock.MagicMock(side_effect=[
            (None, "network"),
            ("success", None)
        ])

        result, error_type = self.retry_manager.retry_request(func, "test_arg")

        assert result == "success"
        assert error_type is None
        assert func.call_count == 2
        mock_sleep.assert_called_once()

    @mock.patch('time.sleep')
    def test_retry_request_all_fail(self, mock_sleep):
        """测试所有重试都失败"""
        func = mock.MagicMock(return_value=(None, "network"))

        result, error_type = self.retry_manager.retry_request(
            func, "test_arg", max_retries=2
        )

        assert result is None
        assert error_type == "network"
        assert func.call_count == 3  # 初始调用 + 2次重试

    def test_retry_request_parameter_error_no_retry(self):
        """测试参数错误不重试"""
        func = mock.MagicMock(return_value=(None, "parameter"))

        result, error_type = self.retry_manager.retry_request(
            func, "test_arg", max_retries=3
        )

        assert result is None
        assert error_type == "parameter"
        # 参数错误不应该重试
        func.assert_called_once()


class TestIncrementalDownloader:
    """IncrementalDownloader单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.downloader = IncrementalDownloader()

    @mock.patch('src.data.downloaders.BaostockProvider')
    def test_backfill_data(self, mock_provider_class):
        """测试回填数据"""
        # Mock提供者
        mock_provider = mock.MagicMock()
        mock_provider_class.return_value = mock_provider

        # Mock数据返回
        test_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "close": [10, 11, 12, 13, 14],
        })
        mock_provider.load.return_value = test_data
        mock_provider.exists.return_value = True

        result = self.downloader.backfill_data(
            stock_codes=["sh.600000"],
            backfill_days=5
        )

        assert "success_count" in result
        assert "failed_count" in result

    @mock.patch('src.data.downloaders.BaostockProvider')
    def test_forward_fill_data(self, mock_provider_class):
        """测试前向填充数据"""
        mock_provider = mock.MagicMock()
        mock_provider_class.return_value = mock_provider

        test_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "close": [10, 11, 12, 13, 14],
        })
        mock_provider.load.return_value = test_data

        result = self.downloader.forward_fill_data(
            stock_codes=["sh.600000"],
            forward_fill_date="2024-12-31"
        )

        assert "success_count" in result

    @mock.patch('src.data.downloaders.BaostockProvider')
    @mock.patch('os.listdir')
    def test_get_existing_stock_list(self, mock_listdir, mock_provider_class):
        """测试获取现有股票列表"""
        mock_listdir.return_value = [
            "sh.600000.csv",
            "sz.000001.csv",
            "readme.txt",
        ]

        stock_list = self.downloader.get_existing_stock_list()

        assert "sh.600000" in stock_list
        assert "sz.000001" in stock_list
        assert "readme.txt" not in stock_list

    @mock.patch('src.data.downloaders.BaostockProvider')
    @mock.patch('os.listdir')
    def test_incremental_update(self, mock_listdir, mock_provider_class):
        """测试增量更新"""
        mock_listdir.return_value = ["sh.600000.csv"]

        mock_provider = mock.MagicMock()
        mock_provider_class.return_value = mock_provider

        # Mock加载现有数据
        existing_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100),
            "close": np.random.rand(100) * 10 + 10,
        })
        mock_provider.load.return_value = existing_data

        result = self.downloader.incremental_update(
            backfill_days=5,
            stock_codes=["sh.600000"]
        )

        assert "success_count" in result
        assert isinstance(result, dict)

    def test_determine_date_range(self):
        """测试确定日期范围"""
        # 有现有数据
        existing_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100),
        })

        start_date, end_date = self.downloader.determine_date_range(
            existing_data, backfill_days=5
        )

        assert start_date is not None
        assert end_date is not None

        # 无现有数据
        start_date, end_date = self.downloader.determine_date_range(
            None, backfill_days=5
        )

        assert start_date is not None
        assert end_date is not None


class TestIntegration:
    """集成测试"""

    @mock.patch('src.data.downloaders.BaostockProvider')
    def test_full_download_workflow(self, mock_provider_class):
        """测试完整下载流程"""
        # Mock提供者
        mock_provider = mock.MagicMock()
        mock_provider_class.return_value = mock_provider

        # Mock数据
        test_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "open": [10] * 10,
            "high": [11] * 10,
            "low": [9] * 10,
            "close": [10] * 10,
            "volume": [1000000] * 10,
        })
        mock_provider.load.return_value = test_data
        mock_provider.exists.return_value = True

        # 创建下载器
        downloader = IncrementalDownloader()

        # 执行下载
        result = downloader.backfill_data(
            stock_codes=["sh.600000"],
            backfill_days=10
        )

        # 验证结果
        assert result is not None
        mock_provider.load.assert_called()


if __name__ == "__main__":
    import pytest

    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
