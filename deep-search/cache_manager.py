#!/usr/bin/env python3
"""
缓存管理模块
提供内存缓存功能，减少重复的API调用
"""
import time
import hashlib
import json
from typing import Any, Optional, Dict, Tuple, Callable, List
from threading import Lock
from config_manager import config
from logger_manager import get_logger

logger = get_logger('cache')

class CacheManager:
    """缓存管理器"""
    
    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock: Lock = Lock()
        self._config: Dict[str, Any] = config.get_cache_config()
        self._enabled: bool = self._config['enabled']
        self._ttl: int = self._config['ttl']
        self._max_size: int = self._config['max_size']
    
    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        # 将参数转换为字符串并生成哈希
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """检查缓存是否过期"""
        return time.time() - timestamp > self._ttl
    
    def _cleanup_expired(self) -> None:
        """清理过期的缓存项"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self._ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")
    
    def _evict_oldest(self) -> None:
        """驱逐最旧的缓存项"""
        if not self._cache:
            return
        
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]
        logger.debug(f"驱逐最旧缓存项: {oldest_key}")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self._enabled:
            return None
        
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if not self._is_expired(timestamp):
                    logger.debug(f"缓存命中: {key}")
                    return value
                else:
                    del self._cache[key]
                    logger.debug(f"缓存过期: {key}")
            
            return None
    
    def set(self, key: str, value: Any, ttl: int = None, expire_time: int = None) -> None:
        """设置缓存值"""
        if not self._enabled:
            return
        
        # 兼容两种参数名
        if ttl is not None:
            expire_time = ttl
        elif expire_time is None:
            expire_time = self._ttl
        
        with self._lock:
            # 清理过期项
            self._cleanup_expired()
            
            # 如果缓存已满，驱逐最旧的项
            while len(self._cache) >= self._max_size:
                self._evict_oldest()
            
            expiry = time.time() + expire_time
            self._cache[key] = (value, expiry)
            logger.debug(f"缓存设置: {key}, 过期时间: {expire_time}秒")
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        if not self._enabled:
            return False
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"缓存删除: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            cache_size = len(self._cache)
            self._cache.clear()
            logger.info(f"清空缓存，共删除 {cache_size} 个项目")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for _, timestamp in self._cache.values()
                if current_time - timestamp > self._ttl
            )
            
            return {
                'enabled': self._enabled,
                'total_items': len(self._cache),
                'expired_items': expired_count,
                'valid_items': len(self._cache) - expired_count,
                'max_size': self._max_size,
                'ttl': self._ttl
            }
    
    def cached_call(self, func: Callable, *args, **kwargs) -> Any:
        """装饰器：缓存函数调用结果"""
        if not self._enabled:
            return func(*args, **kwargs)
        
        # 生成缓存键
        cache_key = f"{func.__name__}_{self._generate_key(*args, **kwargs)}"
        
        # 尝试从缓存获取
        cached_result = self.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 执行函数并缓存结果
        try:
            result = func(*args, **kwargs)
            self.set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"函数调用失败: {func.__name__}, 错误: {e}")
            raise

def cached(ttl: Optional[int] = None, cache_manager_instance: Optional[CacheManager] = None) -> Callable:
    """缓存装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_mgr = cache_manager_instance or cache_manager
            
            if not cache_mgr._enabled:
                return func(*args, **kwargs)
            
            # 生成缓存键
            cache_key = f"{func.__name__}_{cache_mgr._generate_key(*args, **kwargs)}"
            
            # 尝试从缓存获取
            cached_result = cache_mgr.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            try:
                result = func(*args, **kwargs)
                cache_mgr.set(cache_key, result, ttl=ttl)
                return result
            except Exception as e:
                logger.error(f"函数调用失败: {func.__name__}, 错误: {e}")
                raise
        return wrapper
    return decorator

# 全局缓存管理器实例
cache_manager = CacheManager()