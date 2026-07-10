"""
数据提供者单元测试

测试 DataProvider, BaostockProvider, CSVProvider, CachedProvider
"""

import os
import sys
import tempfile
import shutil
from unittest import mock
from datetime import datetime

sys.path.insert(0, '..')

import pandas as pd
import numpy as np

from src.data.providers import (
    DataProvider,
    BaostockProvider,
    CSVProvider,
    CachedProvider,
    create_provider,
)
from src.core.exceptions import DataFetchError


class TestCSVProvider:
    """CSVProvider单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.provider = CSVProvider(self.temp_dir)
        self.sample_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "open": np.random.rand(10) * 10 + 10,
            "high": np.random.rand(10) * 10 + 11,
            "low": np.random.rand(10) * 10 + 9,
            "close": np.random.rand(10) * 10 + 10,
            "volume": np.random.randint(1000000, 10000000, 10),
        })

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_and_load(self):
        """测试保存和加载数据"""
        # 保存数据
        success = self.provider.save("test_stock", self.sample_data)
        assert success is True

        # 检查文件存在
        file_path = os.path.join(self.temp_dir, "test_stock.csv")
        assert os.path.exists(file_path)

        # 加载数据
        loaded_data = self.provider.load("test_stock")
        assert loaded_data is not None
        assert len(loaded_data) == len(self.sample_data)
        assert "date" in loaded_data.columns

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        result = self.provider.load("nonexistent_stock")
        assert result is None

    def test_exists(self):
        """测试检查文件是否存在"""
        # 保存前
        assert self.provider.exists("test_stock") is False

        # 保存后
        self.provider.save("test_stock", self.sample_data)
        assert self.provider.exists("test_stock") is True

    def test_validate_data(self):
        """测试数据验证"""
        # 有效数据
        assert self.provider.validate_data(self.sample_data) is True

        # 缺少必需列
        invalid_data = pd.DataFrame({"date": [1, 2, 3]})
        assert self.provider.validate_data(invalid_data) is False

        # 空数据
        empty_data = pd.DataFrame({
            "date": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        })
        assert self.provider.validate_data(empty_data) is False

    def test_load_file_with_invalid_data(self):
        """测试加载无效数据"""
        # 创建一个缺少必需列的CSV文件
        invalid_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "close": [10, 11, 12, 13, 14]
        })
        file_path = os.path.join(self.temp_dir, "invalid.csv")
        invalid_data.to_csv(file_path, index=False)

        # 应该抛出DataFetchError
        try:
            self.provider.load("invalid")
            assert False, "应该抛出异常"
        except DataFetchError as e:
            assert "缺少必需列" in str(e)


class TestBaostockProvider:
    """BaostockProvider单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.provider = BaostockProvider()

    def teardown_method(self):
        """测试后清理"""
        if self.provider._logged_in:
            self.provider.close()

    def test_format_stock_code(self):
        """测试股票代码格式化"""
        # 已经是baostock格式
        assert self.provider._format_stock_code("sh.600000") == "sh.600000"
        assert self.provider._format_stock_code("sz.000001") == "sz.000001"

        # 6位数字代码
        assert self.provider._format_stock_code("600000") == "sh.600000"
        assert self.provider._format_stock_code("000001") == "sz.000001"
        assert self.provider._format_stock_code("300001") == "sz.300001"

        # 其他格式保持不变
        assert self.provider._format_stock_code("abc") == "abc"

    def test_exists_with_valid_codes(self):
        """测试检查有效的股票代码"""
        assert self.provider.exists("sh.600000") is True
        assert self.provider.exists("sz.000001") is True
        assert self.provider.exists("600000") is True
        assert self.provider.exists("000001") is True

    def test_exists_with_invalid_codes(self):
        """测试检查无效的股票代码"""
        assert self.provider.exists("invalid") is False
        assert self.provider.exists("123456") is False  # 无效的6位代码

    def test_classify_error(self):
        """测试错误分类"""
        # 参数错误
        assert self.provider._classify_error("起始日期大于终止日期") == "parameter"
        assert self.provider._classify_error("股票代码不存在") == "parameter"
        assert self.provider._classify_error("invalid parameter") == "parameter"

        # 网络错误
        assert self.provider._classify_error("网络超时") == "network"
        assert self.provider._classify_error("连接超时") == "network"

        # 数据问题
        assert self.provider._classify_error("没有数据") == "data"
        assert self.provider._classify_error("停牌") == "data"

        # 未知错误
        assert self.provider._classify_error("未知错误") == "unknown"
        assert self.provider._classify_error("") == "unknown"

    @mock.patch('src.data.providers.bs')
    def test_ensure_login(self, mock_bs):
        """测试登录管理"""
        # Mock登录成功
        mock_lg = mock.MagicMock()
        mock_lg.error_code = "0"
        mock_bs.login.return_value = mock_lg

        provider = BaostockProvider()
        assert provider._logged_in is False

        provider._ensure_login()
        assert provider._logged_in is True
        mock_bs.login.assert_called_once()

    @mock.patch('src.data.providers.bs')
    def test_ensure_login_failure(self, mock_bs):
        """测试登录失败"""
        # Mock登录失败
        mock_lg = mock.MagicMock()
        mock_lg.error_code = "1"
        mock_lg.error_msg = "登录失败"
        mock_bs.login.return_value = mock_lg

        provider = BaostockProvider()

        try:
            provider._ensure_login()
            assert False, "应该抛出异常"
        except DataFetchError as e:
            assert "baostock登录失败" in str(e)

    def test_close(self):
        """测试关闭连接"""
        provider = BaostockProvider()
        provider._logged_in = True

        with mock.patch('src.data.providers.bs.logout') as mock_logout:
            provider.close()
            assert provider._logged_in is False
            mock_logout.assert_called_once()

    def test_context_manager(self):
        """测试上下文管理器"""
        with mock.patch('src.data.providers.bs.login') as mock_login, \
             mock.patch('src.data.providers.bs.logout') as mock_logout:
            mock_login.return_value.error_code = "0"

            with BaostockProvider() as provider:
                assert provider._logged_in is True

            assert provider._logged_in is False
            mock_logout.assert_called_once()


class TestCachedProvider:
    """CachedProvider单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.mock_base = mock.MagicMock(spec=DataProvider)
        self.cached_provider = CachedProvider(self.mock_base)

    def test_load_first_call(self):
        """测试第一次调用从底层提供者获取"""
        test_data = pd.DataFrame({"col1": [1, 2, 3]})
        self.mock_base.load.return_value = test_data

        result = self.cached_provider.load("test_id")
        assert result.equals(test_data)
        self.mock_base.load.assert_called_once_with("test_id")

    def test_load_second_call_from_cache(self):
        """测试第二次调用从缓存获取"""
        test_data = pd.DataFrame({"col1": [1, 2, 3]})
        self.mock_base.load.return_value = test_data

        # 第一次调用
        result1 = self.cached_provider.load("test_id")
        # 第二次调用
        result2 = self.cached_provider.load("test_id")

        # 底层提供者应该只被调用一次
        assert self.mock_base.load.call_count == 1
        assert result1.equals(result2)

    def test_save_delegates_to_base(self):
        """测试保存委托给底层提供者"""
        test_data = pd.DataFrame({"col1": [1, 2, 3]})
        self.mock_base.save.return_value = True

        result = self.cached_provider.save("test_id", test_data)
        assert result is True
        self.mock_base.save.assert_called_once_with("test_id", test_data)

    def test_exists_delegates_to_base(self):
        """测试检查存在委托给底层提供者"""
        self.mock_base.exists.return_value = True

        result = self.cached_provider.exists("test_id")
        assert result is True
        self.mock_base.exists.assert_called_once_with("test_id")


class TestCreateProvider:
    """create_provider工厂函数测试"""

    def test_create_baostock_provider(self):
        """测试创建baostock提供者"""
        provider = create_provider("baostock")
        assert isinstance(provider, BaostockProvider)

    def test_create_csv_provider(self):
        """测试创建CSV提供者"""
        provider = create_provider("csv", base_dir="./test")
        assert isinstance(provider, CSVProvider)

    def test_create_cached_provider(self):
        """测试创建带缓存的提供者"""
        base = BaostockProvider()
        provider = create_provider("cached", provider=base)
        assert isinstance(provider, CachedProvider)
        assert provider.provider is base

    def test_create_cached_provider_without_base(self):
        """测试创建cached提供者但未指定底层提供者"""
        try:
            create_provider("cached")
            assert False, "应该抛出异常"
        except ValueError as e:
            assert "provider" in str(e)

    def test_create_invalid_provider(self):
        """测试创建无效的提供者类型"""
        try:
            create_provider("invalid_type")
            assert False, "应该抛出异常"
        except ValueError as e:
            assert "不支持的" in str(e)


class TestDataProviderAbstract:
    """DataProvider抽象基类测试"""

    def test_cannot_instantiate_abstract(self):
        """测试不能实例化抽象基类"""
        try:
            DataProvider()
            assert False, "应该抛出异常"
        except TypeError:
            pass  # 预期的异常

    def test_concrete_implementation(self):
        """测试具体实现可以实例化"""

        class ConcreteProvider(DataProvider):
            def load(self, identifier, **kwargs):
                return None

            def save(self, identifier, data, **kwargs):
                return True

            def exists(self, identifier, **kwargs):
                return True

        provider = ConcreteProvider()
        assert provider.load("test") is None
        assert provider.save("test", None) is True
        assert provider.exists("test") is True


if __name__ == "__main__":
    import pytest

    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
