"""
数据提供者模块

提供统一的数据访问接口，支持多种数据源(baostock API、CSV文件等)。

硬化要点（PRD: baostock 行情模块反扒硬化 + 磁盘缓存 + 错误必 raise）：
    - ``BaostockProvider`` 内置「反扒限速 + 礼貌重试 + 磁盘缓存 + 错误必 raise +
      线程安全 login」，使得**任何** baostock 调用方（downloaders 内部 /
      ``pipeline.py`` 直连 / 其它直连）都统一受保护，消除「绕过 downloaders 即裸奔」。
    - 反扒/重试的共享实现见 :mod:`src.core.anti_crawler`（自 downloaders 下沉复用）。
    - 磁盘缓存走 :class:`~src.core.cache.FileCache`（joblib；环境无 pyarrow/fastparquet）。
"""

import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, List, Optional, TypeVar

import baostock as bs
import pandas as pd

from src.core.anti_crawler import AntiCrawlerController, RequestRetryManager
from src.core.base import FileSystemProvider
from src.core.cache import CacheConfig, FileCache, MemoryCache
from src.core.config import Config, get_config
from src.core.exceptions import DataFetchError
from src.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# ========== baostock 调用的全局保护（模块级共享，所有 BaostockProvider 实例复用） ==========

# 相邻两次 baostock 网络请求的**最小间隔**（秒）。0 = 不限速。
# 可由环境变量 BAOSTOCK_MIN_INTERVAL 覆盖；测试中可直接改 ``_baostock_rate_limiter.min_interval``。
BAOSTOCK_MIN_INTERVAL = float(os.environ.get("BAOSTOCK_MIN_INTERVAL", "0.2"))
# 单次 load 内对网络/未知错误的最大重试次数（不含首次调用）。
BAOSTOCK_MAX_RETRIES = int(os.environ.get("BAOSTOCK_MAX_RETRIES", "3"))


class RateLimiter:
    """
    最小间隔限速器（pacer）

    保证任意两次 :meth:`acquire` 「放行」之间相距至少 ``min_interval`` 秒。

    与 :class:`~src.core.anti_crawler.AntiCrawlerController` 的自适应延时不同，这里给出的是
    **硬下界**（无向下随机抖动），便于可测试地断言「相邻请求间隔 ≥ X」。
    线程安全。``min_interval <= 0`` 时退化为不限速（:meth:`acquire` 立即返回）。
    """

    def __init__(self, min_interval: float = 0.2):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_release = 0.0  # monotonic 时间戳

    def acquire(self) -> None:
        """阻塞至距上次放行已满 min_interval（min_interval<=0 时立即返回）。"""
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_release)
            if wait > 0:
                time.sleep(wait)
                self._last_release = time.monotonic()
            else:
                self._last_release = now


# 模块级单例限速闸：所有 BaostockProvider 实例共享，避免各自独立计时导致全局超频。
_baostock_rate_limiter = RateLimiter(BAOSTOCK_MIN_INTERVAL)

# baostock 全局 login 会话的共享状态（bs.login 是进程级单例会话）。
# 用锁 + 引用计数实现「多实例共享一次 login、互不踩 close」。
_login_lock = threading.Lock()
_global_logged_in = False
_login_refcount = 0


def _reset_login_state() -> None:
    """重置模块级共享 login 状态（仅供测试隔离使用）。"""
    global _global_logged_in, _login_refcount
    with _login_lock:
        _global_logged_in = False
        _login_refcount = 0


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

    封装baostock API调用，提供股票历史数据查询功能。

    内置保护（对所有调用方统一生效）：
        - **反扒限速**：模块级共享 :data:`_baostock_rate_limiter`，相邻请求间隔 ≥
          :data:`BAOSTOCK_MIN_INTERVAL`。
        - **礼貌重试**：网络/未知错误按指数退避（0.5/1/2s + 抖动）重试 ≤
          :data:`BAOSTOCK_MAX_RETRIES` 次；参数错误不重试。
        - **磁盘缓存**：默认 joblib 落盘（``cache/``），同参重复 load 不重复打 baostock。
        - **错误必 raise**：network/unknown/parameter 错误抛 :class:`DataFetchError`；
          data 类（无数据/停牌）返回 ``None``（合法空）。
        - **线程安全 login**：模块级锁 + 引用计数，多实例共享一次 login、互不踩 close。

    Example:
        >>> provider = BaostockProvider()
        >>> df = provider.load("sh.600000", start_date="2024-01-01", end_date="2024-12-31")
        >>> provider.close()
    """

    # API字段映射
    API_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"

    def __init__(
        self,
        config: Optional[Config] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        """
        初始化baostock数据提供者

        Args:
            config: 配置实例
            cache_config: 缓存配置，None 则默认启用 joblib 磁盘缓存（``cache/``）；
                传 ``CacheConfig(enabled=False)`` 可关闭缓存。
        """
        super().__init__(config)
        self._logged_in = False
        # 指向模块级共享限速闸单例（便于测试断言多实例共享同一闸）
        self._rate_limiter = _baostock_rate_limiter
        self._retry_manager = RequestRetryManager(max_retries=BAOSTOCK_MAX_RETRIES)
        cc = cache_config if cache_config is not None else CacheConfig(backend="file", cache_dir="cache")
        self._cache = self._build_cache(cc)

    @staticmethod
    def _build_cache(cc: CacheConfig):
        """根据 CacheConfig 构造缓存实例（None 表示不缓存）。"""
        if not cc.enabled:
            return None
        if cc.backend == "file":
            return FileCache(ttl=cc.ttl, cache_dir=cc.cache_dir)
        return MemoryCache(ttl=cc.ttl)

    @staticmethod
    def _reset_login_state() -> None:
        """重置模块级共享 login 状态（仅供测试隔离使用）。"""
        _reset_login_state()

    def _ensure_login(self) -> None:
        """确保已登录baostock系统（线程安全共享：多实例只 login 一次）。"""
        global _global_logged_in, _login_refcount
        with _login_lock:
            if self._logged_in:
                return
            if not _global_logged_in:
                lg = bs.login()
                if lg.error_code != "0":
                    raise DataFetchError(
                        f"baostock登录失败: {getattr(lg, 'error_msg', '')}",
                        source="baostock",
                        error_code=DataFetchError.NETWORK_ERROR,
                        context={"error_code": lg.error_code},
                    )
                _global_logged_in = True
                logger.info("baostock登录成功")
            self._logged_in = True
            _login_refcount += 1

    def _format_stock_code(self, stock_code: str) -> str:
        """
        格式化股票代码为baostock格式

        Args:
            stock_code: 股票代码(如 "600000", "sh.600000", "sz.000001", "bj.830000")

        Returns:
            baostock格式的股票代码 (sh./sz./bj. 前缀)
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
            elif stock_code.startswith("8"):
                # 北交所
                return f"bj.{stock_code}"

        # 无法转换，返回原值
        return stock_code

    def load(
        self,
        identifier: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "d",
        adjustflag: str = "3",
        filter_st: bool = True,
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
            filter_st: 是否过滤停牌/ST 行（默认 True，与历史行为一致）；False 则保留全部行。
            **kwargs: 其他参数

        Returns:
            股票数据DataFrame；data 类（无数据/停牌）返回 None。

        Raises:
            DataFetchError: 网络/未知/参数错误时抛出（绝不静默返回 None）。
        """
        # 默认日期范围
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            # 默认获取最近一年数据
            start_date = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y-%m-%d")

        stock_code = self._format_stock_code(identifier)

        # 1) 缓存命中则直接返回（命中时无需 login，省一次 baostock 调用）
        cache_key = (
            f"bs_k:{stock_code}:start={start_date}:end={end_date}"
            f":freq={frequency}:adj={adjustflag}"
        )
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"BaostockProvider缓存命中: {cache_key}")
                return cached.copy()

        logger.debug(f"查询baostock: {stock_code}, {start_date} ~ {end_date}")

        # 2) login（线程安全共享）
        self._ensure_login()

        # 3) 查询：限速 + 礼貌重试 + 错误必 raise
        rs = self._query_with_retry(stock_code, start_date, end_date, frequency, adjustflag)
        if rs is None:
            # data 类（无数据/停牌）：合法空，不缓存 None
            return None

        # 4) 收集数据
        data_list = []
        while (rs.error_code == "0") and rs.next():
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

        # 5) 过滤停牌和ST股票（可选；默认行为与改动前一致）
        if filter_st:
            df = df[(df["tradestatus"] == "1") & (df["isST"] == "0")]

        # 数据验证
        if len(df) == 0:
            logger.warning(f"股票 {stock_code} 过滤后无有效数据")
            return None

        # 6) 写缓存
        if self._cache is not None:
            self._cache.set(cache_key, df)

        logger.debug(f"成功获取 {len(df)} 条数据")
        return df

    def _query_with_retry(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        frequency: str,
        adjustflag: str,
    ):
        """
        带限速/重试/错误分类的 baostock 查询

        - 抛异常：network/unknown 重试至耗尽后 raise DataFetchError(NETWORK_ERROR)；
          parameter 立即 raise DataFetchError(API_ERROR)。
        - rs.error_code != "0"：data 类返回 None；network/unknown/parameter raise。
        - rs.error_code == "0"：返回 rs 供调用方收集行。
        """
        attempt = 0
        while True:
            self._rate_limiter.acquire()
            try:
                rs = bs.query_history_k_data_plus(
                    stock_code,
                    self.API_FIELDS,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    adjustflag=adjustflag,
                )
            except Exception as e:
                etype = self._classify_error(str(e))
                if etype == "parameter":
                    # 参数错误重试无意义，立即抛出
                    raise DataFetchError(
                        f"baostock查询参数错误: {e}",
                        source="baostock",
                        error_code=DataFetchError.API_ERROR,
                        context={"stock_code": stock_code},
                    )
                if self._retry_manager.should_retry(etype, attempt):
                    wait = 0.5 * (2 ** attempt) + random.uniform(0, 0.1)
                    logger.warning(
                        f"baostock查询异常({etype})，第{attempt + 1}次重试，等待{wait:.2f}s: {e}"
                    )
                    time.sleep(wait)
                    attempt += 1
                    continue
                # 重试耗尽（或不可重试）→ 必 raise，绝不静默 None
                raise DataFetchError(
                    f"baostock查询失败(网络异常，重试耗尽): {e}",
                    source="baostock",
                    error_code=DataFetchError.NETWORK_ERROR,
                    context={
                        "stock_code": stock_code,
                        "attempts": attempt + 1,
                        "error_type": etype,
                    },
                )

            # baostock 返回了 rs（未抛异常）
            if rs.error_code != "0":
                etype = self._classify_error(rs.error_msg or "")
                if etype == "data":
                    # 无数据/停牌：合法空，返回 None（不缓存、不 raise）
                    logger.info(f"baostock无数据/停牌({rs.error_msg})，返回None: {stock_code}")
                    return None
                code = DataFetchError.NETWORK_ERROR if etype == "network" else DataFetchError.API_ERROR
                raise DataFetchError(
                    f"baostock查询失败: {rs.error_msg}",
                    source="baostock",
                    error_code=code,
                    context={
                        "stock_code": stock_code,
                        "error_code": rs.error_code,
                        "error_type": etype,
                    },
                )

            return rs

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
        检查股票代码是否有效（含沪深+北交所）

        Args:
            identifier: 股票代码
            **kwargs: 其他参数

        Returns:
            股票代码是否有效
        """
        code = self._format_stock_code(identifier)
        # 沪市 sh.6xxxxx / 深市 sz.[03]xxxxx / 北交所 bj.8xxxxx
        pattern = r"^(sh\.6\d{5}|sz\.[03]\d{5}|bj\.8\d{5})$"
        return bool(re.match(pattern, code))

    def get_all_stocks(self) -> List[str]:
        """
        获取全量A股股票列表

        Returns:
            股票代码列表 (sh./sz./bj. 前缀格式)

        Raises:
            DataFetchError: 获取股票列表失败时抛出
        """
        self._ensure_login()
        self._rate_limiter.acquire()

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

        except DataFetchError:
            raise
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
        """关闭baostock连接（引用计数：最后一个实例才真正 logout，互不踩）。"""
        global _global_logged_in, _login_refcount
        with _login_lock:
            if not self._logged_in:
                return
            self._logged_in = False
            _login_refcount = max(0, _login_refcount - 1)
            if _login_refcount == 0 and _global_logged_in:
                bs.logout()
                _global_logged_in = False
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
        分类错误类型（复用 :meth:`AntiCrawlerController.classify_error`）

        Args:
            error_msg: 错误消息

        Returns:
            错误类型: 'parameter', 'network', 'data', 'unknown'
        """
        return AntiCrawlerController.classify_error("", error_msg)


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

    为其他数据提供者添加缓存功能，减少重复请求。支持内存与磁盘（joblib）两种后端：
    ``CacheConfig(backend="file")`` 时落盘并按 TTL（文件 mtime）过期；默认内存后端。

    Attributes:
        provider: 被装饰的数据提供者
        cache_config: 缓存配置

    Example:
        >>> base_provider = BaostockProvider()
        >>> provider = CachedProvider(base_provider, CacheConfig(backend="file", cache_dir="cache"))
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
            cache_config: 缓存配置，None 则使用默认（内存）配置
        """
        self.provider = provider
        self.cache_config = cache_config or CacheConfig()
        # 兼容底层 provider 无 config 属性的情况（如测试用的 mock 对象）
        self.config = getattr(provider, "config", None)
        self._cache = BaostockProvider._build_cache(self.cache_config)

    def load(self, identifier: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        加载数据(带缓存)

        首次调用从底层提供者获取并缓存，后续调用从缓存返回。

        Args:
            identifier: 股票代码或标识符
            **kwargs: 额外参数

        Returns:
            数据DataFrame，底层返回None则不缓存、直接返回None
        """
        if self._cache is not None:
            cache_key = self._make_cache_key(identifier, **kwargs)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"CachedProvider缓存命中: {cache_key}")
                return cached.copy() if hasattr(cached, "copy") else cached

        # 从底层提供者获取
        data = self.provider.load(identifier, **kwargs)

        # 存入缓存（None 不缓存）
        if data is not None and self._cache is not None:
            self._cache.set(self._make_cache_key(identifier, **kwargs), data)

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
        """生成缓存键（含 identifier + 关键查询参数）。"""
        parts = [identifier]
        # 添加关键参数到键
        for key in ["start_date", "end_date", "frequency", "adjustflag"]:
            if key in kwargs:
                parts.append(f"{key}={kwargs[key]}")
        return ":".join(parts)


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
        >>> # 创建baostock提供者（默认自带磁盘缓存）
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
