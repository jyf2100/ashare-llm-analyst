"""
简单缓存系统

提供内存缓存和装饰器支持，具有TTL(生存时间)功能

功能特性:
- 基于装饰器的函数结果缓存
- TTL过期时间支持
- 缓存统计信息
- 自动清理过期条目
"""

import functools
import hashlib
import inspect
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from src.core.exceptions import CacheError
from src.core.logger import get_logger

logger = get_logger(__name__)

# 类型变量，用于类型注解
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CacheConfig:
    """
    缓存配置类

    Attributes:
        enabled: 是否启用缓存
        ttl: 缓存存活时间(秒)，默认3600秒(1小时)
        backend: 缓存后端类型 (memory=内存, file=文件)
        cache_dir: 文件缓存目录
        key_prefix: 缓存键前缀，用于区分不同缓存
    """
    enabled: bool = True  # 默认启用缓存
    ttl: int = 3600  # 缓存存活时间: 1小时
    backend: str = "memory"  # 后端类型: 内存缓存
    cache_dir: str = "cache"  # 文件缓存目录
    key_prefix: str = ""  # 缓存键前缀


@dataclass
class CacheEntry:
    """
    单个缓存条目

    Attributes:
        value: 缓存的值
        expiry: 过期时间(Unix时间戳)
        hits: 命中次数
        created_at: 创建时间(Unix时间戳)
    """
    value: Any  # 缓存值
    expiry: float  # 过期时间戳
    hits: int = 0  # 命中次数
    created_at: float = field(default_factory=time.time)  # 创建时间


class MemoryCache:
    """
    内存缓存后端

    提供基于字典的内存缓存，支持TTL过期

    Attributes:
        ttl: 默认缓存存活时间(秒)
        _store: 缓存存储字典 {key: CacheEntry}

    Example:
        >>> cache = MemoryCache(ttl=600)
        >>> cache.set("key1", "value1")
        >>> value = cache.get("key1")
    """

    def __init__(self, ttl: int = 3600):
        """
        初始化内存缓存

        Args:
            ttl: 默认缓存存活时间(秒)
        """
        self.ttl = ttl  # 默认TTL
        self._store: Dict[str, CacheEntry] = {}  # 缓存存储

    def get(self, key: str) -> Optional[Any]:
        """
        从缓存获取值

        Args:
            key: 缓存键

        Returns:
            缓存值，如果不存在或已过期则返回None
        """
        entry = self._store.get(key)
        if entry is None:
            # 缓存未命中
            return None

        # 检查是否过期
        if time.time() > entry.expiry:
            # 已过期，删除条目
            del self._store[key]
            return None

        # 命中，增加计数
        entry.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 要缓存的值
            ttl: 存活时间(秒)，None则使用默认TTL
        """
        # 计算过期时间戳
        expiry = time.time() + (ttl or self.ttl)
        # 创建缓存条目
        self._store[key] = CacheEntry(value=value, expiry=expiry)

    def delete(self, key: str) -> bool:
        """
        删除缓存条目

        Args:
            key: 缓存键

        Returns:
            True if deleted, False if not found
        """
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """清空所有缓存条目"""
        self._store.clear()

    def cleanup_expired(self) -> int:
        """
        清理过期的缓存条目

        Returns:
            清理的条目数量
        """
        now = time.time()
        # 找出所有过期的键
        expired_keys = [k for k, v in self._store.items() if v.expiry < now]
        # 删除过期条目
        for key in expired_keys:
            del self._store[key]
        return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            包含entries(条目数)、total_hits(总命中次数)、expired(过期数)的字典
        """
        total_hits = sum(e.hits for e in self._store.values())
        return {
            "entries": len(self._store),  # 当前条目数
            "total_hits": total_hits,  # 总命中次数
            "expired": sum(1 for e in self._store.values() if e.expiry < time.time()),  # 过期条目数
        }


# ========== 全局缓存实例管理 ==========

# 全局缓存字典 {cache_key: MemoryCache}
_caches: Dict[str, MemoryCache] = {}


def _get_cache(config: CacheConfig) -> MemoryCache:
    """
    获取或创建缓存实例

    Args:
        config: 缓存配置

    Returns:
        MemoryCache实例
    """
    # 使用backend和key_prefix组合作为缓存标识
    cache_key = f"{config.backend}_{config.key_prefix}"
    if cache_key not in _caches:
        # 创建新缓存实例
        _caches[cache_key] = MemoryCache(ttl=config.ttl)
    return _caches[cache_key]


def _generate_cache_key(
    func: Callable, args: tuple, kwargs: dict, prefix: str = ""
) -> str:
    """
    生成缓存键

    根据函数名、参数值生成唯一的缓存键

    Args:
        func: 被调用的函数
        args: 位置参数元组
        kwargs: 关键字参数字典
        prefix: 键前缀

    Returns:
        缓存键字符串
    """
    # 组成键的各个部分
    parts = [prefix, func.__name__]

    # 尝试序列化参数
    try:
        # 优先使用JSON序列化(更快)
        args_repr = json.dumps(args, sort_keys=True, default=str)
        kwargs_repr = json.dumps(kwargs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # 复杂对象回退到pickle+MD5
        args_repr = hashlib.md5(pickle.dumps(args)).hexdigest()
        kwargs_repr = hashlib.md5(pickle.dumps(kwargs)).hexdigest()

    parts.extend([args_repr, kwargs_repr])

    # 用冒号连接各部分
    key = ":".join(str(p) for p in parts)

    # 如果键太长则使用MD5哈希
    if len(key) > 200:
        key = hashlib.md5(key.encode()).hexdigest()

    return key


def cached(
    config: Optional[CacheConfig] = None,
    key: Optional[str] = None,
) -> Callable[[F], F]:
    """
    缓存装饰器

    用于缓存函数的返回值，减少重复计算

    Args:
        config: 缓存配置，None则使用默认配置
        key: 自定义缓存键前缀

    Returns:
        装饰器函数

    Example:
        >>> # 使用默认配置
        >>> @cached()
        >>> def load_data(symbol):
        >>>     return fetch_data(symbol)
        >>>
        >>> # 自定义TTL
        >>> @cached(ttl=600)  # 缓存10分钟
        >>> def calculate_indicators(df):
        >>>     return df.indicator()
        >>>
        >>> # 自定义键前缀
        >>> @cached(key="stock_data")
        >>> def get_stock_info(code):
        >>>     return api.get_info(code)
    """
    if config is None:
        config = CacheConfig()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查缓存是否启用
            if not config.enabled:
                return func(*args, **kwargs)

            # 生成缓存键
            cache_key = _generate_cache_key(func, args, kwargs, config.key_prefix or key)

            # 获取缓存实例
            cache = _get_cache(config)

            # 尝试从缓存获取
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_value

            # 缓存未命中，调用原函数
            logger.debug(f"缓存未命中: {cache_key}")
            result = func(*args, **kwargs)

            # 存入缓存
            cache.set(cache_key, result, config.ttl)

            return result

        return wrapper  # type: ignore

    return decorator


def clear_cache(
    config: Optional[CacheConfig] = None, key: Optional[str] = None
) -> int:
    """
    清理缓存

    Args:
        config: 缓存配置，决定清理哪个缓存
        key: 指定要清理的键，None表示清理全部

    Returns:
        清理的条目数量

    Example:
        >>> # 清理所有缓存
        >>> clear_cache()
        >>>
        >>> # 清理特定缓存
        >>> clear_cache(key="my_cache")
    """
    if config is None:
        config = CacheConfig()

    cache = _get_cache(config)

    if key:
        # 删除特定键
        if cache.delete(key):
            return 1
        return 0

    # 清空全部缓存
    count = len(cache._store)
    cache.clear()
    return count


def get_cache_stats(config: Optional[CacheConfig] = None) -> Dict[str, Any]:
    """
    获取缓存统计信息

    Args:
        config: 缓存配置

    Returns:
        统计信息字典，包含entries、total_hits、expired

    Example:
        >>> stats = get_cache_stats()
        >>> print(f"缓存条目: {stats['entries']}")
        >>> print(f"命中次数: {stats['total_hits']}")
    """
    if config is None:
        config = CacheConfig()

    cache = _get_cache(config)
    return cache.stats()


def cleanup_expired_caches() -> int:
    """
    清理所有缓存中的过期条目

    Returns:
        清理的过期条目总数

    Example:
        >>> count = cleanup_expired_caches()
        >>> print(f"清理了 {count} 个过期条目")
    """
    total = 0
    for cache in _caches.values():
        total += cache.cleanup_expired()
    return total
