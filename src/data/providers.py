"""
数据提供者模块

提供统一的数据访问接口，支持多种数据源(baostock API、CSV文件等)
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")

import baostock as bs
import pandas as pd
from functools import wraps

from src.core.base import FileSystemProvider
from src.core.cache import CacheConfig, cached
from src.core.config import Config, get_config
from src.core.exceptions import DataFetchError
from src.core.logger import get_logger

logger = get_logger(__name__)


class DataProvider(ABC, Generic[T]):
    """
    数据提供者抽象基类

    定义了数据访问的统一接口，所有数据提供者都应实现此接口

    Attributes:
        config: 配置实例

    Example:
        >>> class MyProvider(DataProvider[pd.DataFrame]):
        >>>     def load(self, identifier: str, **kwargs) -> Optional[pd.DataFrame]:
        >>>         return pd.read_csv(f"{identifier}.csv")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化数据提供者

        Args:
            config: 配置实例，None则使用全局配置
        """
        self.config = config or get_config()

    @abstractmethod
    def load(self, identifier: str, **kwargs) -> Optional["T"]:
        """
        加载指定标识符的数据

        Args:
            identifier: 股票代码、文件路径或其他标识符
            **kwargs: 额外参数(如日期范围、字段等)

        Returns:
            加载的数据，未找到则返回None

        Raises:
            DataFetchError: 数据加载失败时抛出
        """
        pass

    @abstractmethod
    def save(self, identifier: str, data: "T", **kwargs) -> bool:
        """
        保存指定标识符的数据

        Args:
            identifier: 股票代码、文件路径或其他标识符
            data: 要保存的数据
            **kwargs: 额外参数

        Returns:
            成功返回True，失败返回False
        """
        pass

    @abstractmethod
    def exists(self, identifier: str, **kwargs) -> bool:
        """
        检查指定标识符的数据是否存在

        Args:
            identifier: 股票代码、文件路径或其他标识符
            **kwargs: 额外参数

        Returns:
            数据存在返回True，否则返回False
        """
        pass

    def list_available(self, **kwargs) -> List[str]:
        """
        列出所有可用的标识符

        Args:
            **kwargs: 过滤参数

        Returns:
            可用标识符列表
        """
        return []


class BaostockProvider(DataProvider[pd.DataFrame]):
    """
    baostock API数据提供者

    封装baostock API调用，提供股票历史数据查询功能

    Attributes:
        _logged_in: 是否已登录baostock

    Example:
        >>> provider = BaostockProvider()
        >>> df = provider.load("sh.600000", start_date="2024-01-01", end_date="2024-12-31")
        >>> provider.close()
    """

    # API字段映射
    API_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"

    def __init__(self, config: Optional[Config] = None):
        """
        初始化baostock数据提供者

        Args:
            config: 配置实例
        """
        super().__init__(config)
        self._logged_in = False

    def _ensure_login(self) -> None:
        """确保已登录baostock系统"""
        if self._logged_in:
            return

        lg = bs.login()
        if lg.error_code != "0":
            raise DataFetchError(
                f"baostock登录失败: {lg.error_msg}",
                source="baostock",
                context={"error_code": lg.error_code},
            )
        self._logged_in = True
        logger.info("baostock登录成功")

    def _format_stock_code(self, stock_code: str) -> str:
        """
        格式化股票代码为baostock格式

        Args:
            stock_code: 股票代码(如 "600000", "sh.600000", "sz.000001")

        Returns:
            baostock格式的股票代码 (sh.xxxxxx 或 sz.xxxxxx)
        """
        # 已经是baostock格式
        if "." in stock_code:
            return stock_code

        # 根据代码首位判断市场
        if len(stock_code) == 6 and stock_code.isdigit():
            if stock_code.startswith("6"):
                return f"sh.{stock_code}"
            elif stock_code.startswith(("0", "3")):
                return f"sz.{stock_code}"

        # 无法转换，返回原值
        return stock_code

    def load(
        self,
        identifier: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "d",
        adjustflag: str = "3",
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        从baostock加载股票历史数据

        Args:
            identifier: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            frequency: 数据频率 (d=日线, w=周线, m=月线)
            adjustflag: 复权类型 (1=后复权, 2=前复权, 3=不复权)
            **kwargs: 其他参数

        Returns:
            股票数据DataFrame，失败返回None

        Raises:
            DataFetchError: 数据获取失败时抛出
        """
        self._ensure_login()

        # 格式化股票代码
        stock_code = self._format_stock_code(identifier)

        # 默认日期范围
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            # 默认获取最近一年数据
            start_date = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y-%m-%d")

        logger.debug(f"查询baostock: {stock_code}, {start_date} ~ {end_date}")

        # 调用API
        rs = bs.query_history_k_data_plus(
            stock_code,
            self.API_FIELDS,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjustflag,
        )

        if rs.error_code != "0":
            error_type = self._classify_error(rs.error_msg)
            if error_type == "parameter":
                raise DataFetchError(
                    f"baostock查询失败: {rs.error_msg}",
                    source="baostock",
                    context={"stock_code": stock_code, "error_code": rs.error_code},
                )
            logger.warning(f"baostock查询返回错误: {rs.error_msg}")
            return None

        # 收集数据
        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            logger.warning(f"股票 {stock_code} 无数据返回")
            return None

        # 创建DataFrame
        df = pd.DataFrame(data_list, columns=rs.fields)

        # 数据类型转换
        numeric_columns = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"])

        # 过滤停牌和ST股票
        df = df[(df["tradestatus"] == "1") & (df["isST"] == "0")]

        # 数据验证
        if len(df) == 0:
            logger.warning(f"股票 {stock_code} 过滤后无有效数据")
            return None

        logger.debug(f"成功获取 {len(df)} 条数据")
        return df

    def save(self, identifier: str, data: pd.DataFrame, **kwargs) -> bool:
        """
        Baostock提供者不支持直接保存数据

        使用CSVProvider保存数据

        Args:
            identifier: 股票代码
            data: 要保存的数据
            **kwargs: 其他参数

        Returns:
            False (不支持)
        """
        logger.warning("BaostockProvider不支持save方法，请使用CSVProvider")
        return False

    def exists(self, identifier: str, **kwargs) -> bool:
        """
        检查股票代码是否有效

        Args:
            identifier: 股票代码
            **kwargs: 其他参数

        Returns:
            股票代码是否有效
        """
        # 简单验证格式
        code = self._format_stock_code(identifier)
        pattern = r"^(sh\.6\d{5}|sz\.[03]\d{5})$"
        import re

        return bool(re.match(pattern, code))

    def get_all_stocks(self) -> List[str]:
        """
        获取全量A股股票列表

        Returns:
            股票代码列表 (sh.xxxxxx 或 sz.xxxxxx 格式)

        Raises:
            DataFetchError: 获取股票列表失败时抛出
        """
        self._ensure_login()

        try:
            # 获取所有证券信息
            stock_rs = bs.query_stock_basic()

            if stock_rs.error_code != "0":
                raise DataFetchError(
                    f"获取股票列表失败: {stock_rs.error_msg}",
                    source="baostock",
                    context={"error_code": stock_rs.error_code},
                )

            # 转换为DataFrame
            stock_df = stock_rs.get_data()
            if stock_df is None or len(stock_df) == 0:
                logger.warning("未获取到股票列表")
                return []

            # type=1 是股票，type=2 是指数，type=4 是其他
            # 只保留股票类型（过滤指数、基金等）
            stock_df = stock_df[stock_df["type"] == "1"]

            # 过滤掉已退市的股票 (outDate不为空表示已退市)
            stock_df = stock_df[stock_df["outDate"].isna() | (stock_df["outDate"] == "")]

            # 提取股票代码列表
            stock_codes = stock_df["code"].tolist()

            logger.info(f"获取到 {len(stock_codes)} 只A股股票")
            return stock_codes

        except Exception as e:
            raise DataFetchError(
                f"获取股票列表异常: {e}",
                source="baostock",
            )

    def list_available(self, **kwargs) -> List[str]:
        """
        列出所有可用的股票代码

        Args:
            **kwargs: 过滤参数

        Returns:
            股票代码列表
        """
        return self.get_all_stocks()

    def close(self) -> None:
        """关闭baostock连接"""
        if self._logged_in:
            bs.logout()
            self._logged_in = False
            logger.info("baostock连接已关闭")

    def __enter__(self):
        """上下文管理器入口"""
        self._ensure_login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False

    @staticmethod
    def _classify_error(error_msg: str) -> str:
        """
        分类错误类型

        Args:
            error_msg: 错误消息

        Returns:
            错误类型: 'parameter', 'network', 'data', 'unknown'
        """
        if not error_msg:
            return "unknown"

        error_msg_lower = error_msg.lower()

        # 参数错误
        param_keywords = [
            "起始日期大于终止日期",
            "股票代码不存在",
            "股票代码错误",
            "日期格式错误",
            "参数错误",
            "invalid parameter",
            "invalid date",
            "invalid code",
        ]
        if any(kw in error_msg_lower for kw in param_keywords):
            return "parameter"

        # 网络错误
        network_keywords = ["网络超时", "连接被拒绝", "连接超时", "network timeout", "connection"]
        if any(kw in error_msg_lower for kw in network_keywords):
            return "network"

        # 数据问题
        data_keywords = ["没有数据", "数据为空", "no data", "empty data", "停牌", "suspended"]
        if any(kw in error_msg_lower for kw in data_keywords):
            return "data"

        return "unknown"


class CSVProvider(FileSystemProvider[pd.DataFrame]):
    """
    CSV文件数据提供者

    从CSV文件读写股票数据，继承自FileSystemProvider

    Attributes:
        base_dir: CSV文件基础目录
        required_columns: 必需的列名

    Example:
        >>> provider = CSVProvider("market_data")
        >>> df = provider.load("sh.600000")
        >>> provider.save("sh.600000", df)
    """

    REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

    def __init__(self, base_dir: str, config: Optional[Config] = None):
        """
        初始化CSV数据提供者

        Args:
            base_dir: CSV文件基础目录
            config: 配置实例
        """
        super().__init__(base_dir)
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)

    def load_file(self, path: str) -> pd.DataFrame:
        """
        从CSV文件加载数据

        Args:
            path: CSV文件路径

        Returns:
            股票数据DataFrame

        Raises:
            DataFetchError: 文件读取失败时抛出
        """
        try:
            df = pd.read_csv(path)

            # 验证必需列
            missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
            if missing_cols:
                raise DataFetchError(
                    f"CSV文件缺少必需列: {missing_cols}",
                    source="csv",
                    context={"path": path, "missing_columns": list(missing_cols)},
                )

            # 转换日期
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            # 转换数值列
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 按日期排序
            if "date" in df.columns:
                df = df.sort_values("date")

            return df

        except Exception as e:
            raise DataFetchError(
                f"读取CSV文件失败: {e}",
                source="csv",
                context={"path": path},
            )

    def save_file(self, path: str, data: pd.DataFrame) -> None:
        """
        保存数据到CSV文件

        Args:
            path: CSV文件路径
            data: 要保存的数据

        Raises:
            DataFetchError: 文件写入失败时抛出
        """
        try:
            # 确保目录存在
            import os

            os.makedirs(os.path.dirname(path), exist_ok=True)

            # 保存为CSV
            data.to_csv(path, index=False, encoding="utf-8")
            self.logger.debug(f"数据已保存到: {path}")

        except Exception as e:
            raise DataFetchError(
                f"保存CSV文件失败: {e}",
                source="csv",
                context={"path": path},
            )

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据质量

        Args:
            df: 要验证的数据

        Returns:
            数据是否有效
        """
        # 检查必需列
        if not all(col in df.columns for col in self.REQUIRED_COLUMNS):
            return False

        # 检查数据行数
        if len(df) == 0:
            return False

        # 检查价格合理性
        if "close" in df.columns:
            if (df["close"] <= 0).all():
                return False

        return True


class CachedProvider(DataProvider[pd.DataFrame]):
    """
    带缓存的数据提供者装饰器

    为其他数据提供者添加缓存功能，减少重复请求

    Attributes:
        provider: 被装饰的数据提供者
        cache_config: 缓存配置

    Example:
        >>> base_provider = BaostockProvider()
        >>> provider = CachedProvider(base_provider)
        >>> # 第一次调用会从baostock获取
        >>> df1 = provider.load("sh.600000")
        >>> # 第二次调用会从缓存获取
        >>> df2 = provider.load("sh.600000")
    """

    def __init__(
        self,
        provider: DataProvider[pd.DataFrame],
        cache_config: Optional[CacheConfig] = None,
    ):
        """
        初始化缓存数据提供者

        Args:
            provider: 被装饰的数据提供者
            cache_config: 缓存配置，None则使用默认配置
        """
        self.provider = provider
        self.cache_config = cache_config or CacheConfig()
        self.config = provider.config

    def load(self, identifier: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        加载数据(带缓存)

        首次调用从底层提供者获取并缓存，后续调用从缓存返回

        Args:
            identifier: 股票代码或标识符
            **kwargs: 额外参数

        Returns:
            数据DataFrame，失败返回None
        """
        # 生成缓存键
        cache_key = self._make_cache_key(identifier, **kwargs)

        # 检查缓存
        if self.cache_config.enabled:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached

        # 从底层提供者获取
        data = self.provider.load(identifier, **kwargs)

        # 存入缓存
        if data is not None and self.cache_config.enabled:
            self._save_to_cache(cache_key, data)

        return data

    def save(self, identifier: str, data: pd.DataFrame, **kwargs) -> bool:
        """保存数据(委托给底层提供者)"""
        return self.provider.save(identifier, data, **kwargs)

    def exists(self, identifier: str, **kwargs) -> bool:
        """检查数据是否存在(委托给底层提供者)"""
        return self.provider.exists(identifier, **kwargs)

    def list_available(self, **kwargs) -> List[str]:
        """列出可用数据(委托给底层提供者)"""
        return self.provider.list_available(**kwargs)

    def _make_cache_key(self, identifier: str, **kwargs) -> str:
        """生成缓存键"""
        parts = [identifier]
        # 添加关键参数到键
        for key in ["start_date", "end_date", "frequency", "adjustflag"]:
            if key in kwargs:
                parts.append(f"{key}={kwargs[key]}")
        return ":".join(parts)

    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        # 这里可以集成到 src.core.cache
        # 目前简化为内存缓存
        if not hasattr(self, "_cache"):
            self._cache = {}
        return self._cache.get(key)

    def _save_to_cache(self, key: str, data: pd.DataFrame) -> None:
        """保存数据到缓存"""
        if not hasattr(self, "_cache"):
            self._cache = {}
        self._cache[key] = data


def create_provider(provider_type: str, **kwargs) -> DataProvider:
    """
    工厂函数: 创建数据提供者实例

    Args:
        provider_type: 提供者类型 ('baostock', 'csv', 'cached')
        **kwargs: 提供者特定参数

    Returns:
        数据提供者实例

    Raises:
        ValueError: 不支持的提供者类型

    Example:
        >>> # 创建baostock提供者
        >>> provider = create_provider("baostock")
        >>>
        >>> # 创建CSV提供者
        >>> provider = create_provider("csv", base_dir="market_data")
        >>>
        >>> # 创建带缓存的提供者
        >>> base = create_provider("baostock")
        >>> provider = create_provider("cached", provider=base)
    """
    if provider_type == "baostock":
        return BaostockProvider(**kwargs)
    elif provider_type == "csv":
        return CSVProvider(**kwargs)
    elif provider_type == "cached":
        provider = kwargs.pop("provider", None)
        if provider is None:
            raise ValueError("cached provider需要指定provider参数")
        return CachedProvider(provider, **kwargs)
    else:
        raise ValueError(f"不支持的provider类型: {provider_type}")
