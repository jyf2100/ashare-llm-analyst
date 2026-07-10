#!/usr/bin/env python3
"""
异常处理模块
统一管理财报分析系统的异常处理机制
"""
import functools
import traceback
from typing import Any, Callable, Optional, Type, Union, Dict, Tuple
from logger_manager import get_logger

logger = get_logger('exception')

class FinancialAnalysisError(Exception):
    """财务分析基础异常类"""
    pass

class APIError(FinancialAnalysisError):
    """API调用异常"""
    def __init__(self, message: str, api_name: str = "", status_code: Optional[int] = None):
        super().__init__(message)
        self.api_name = api_name
        self.status_code = status_code

class DataError(FinancialAnalysisError):
    """数据处理异常"""
    pass

class ConfigError(FinancialAnalysisError):
    """配置异常"""
    pass

class CacheError(FinancialAnalysisError):
    """缓存异常"""
    pass

class KnowledgeBaseError(FinancialAnalysisError):
    """知识库异常"""
    pass

class ExceptionHandler:
    """异常处理器"""
    
    def __init__(self) -> None:
        self.retry_count: Dict[str, int] = {}
    
    def handle_api_error(self, func: Callable, max_retries: int = 3, delay: float = 1.0) -> Callable:
        """API错误处理装饰器"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                        if delay > 0:
                            import time
                            time.sleep(delay * (attempt + 1))  # 指数退避
                    else:
                        logger.error(f"API调用最终失败: {e}")
                        raise APIError(f"API调用失败: {e}", func.__name__)
            
            raise last_exception
        
        return wrapper
    
    def handle_data_error(self, func: Callable, default_value: Any = None) -> Callable:
        """数据错误处理装饰器"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"数据处理错误: {e}")
                logger.debug(f"错误详情: {traceback.format_exc()}")
                
                if default_value is not None:
                    logger.info(f"返回默认值: {default_value}")
                    return default_value
                
                raise DataError(f"数据处理失败: {e}")
        
        return wrapper
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Tuple[bool, Any]:
        """安全执行函数，返回(成功标志, 结果)"""
        try:
            result = func(*args, **kwargs)
            return True, result
        except Exception as e:
            logger.error(f"函数执行失败: {func.__name__}, 错误: {e}")
            logger.debug(f"错误详情: {traceback.format_exc()}")
            return False, str(e)
    
    def log_and_reraise(self, func: Callable):
        """记录异常并重新抛出"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"函数 {func.__name__} 发生异常: {e}")
                logger.debug(f"异常详情: {traceback.format_exc()}")
                raise
        
        return wrapper
    
    def suppress_exceptions(self, func: Callable, exceptions: tuple = (Exception,), default_value: Any = None):
        """抑制指定异常"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.warning(f"抑制异常: {func.__name__}, 错误: {e}")
                return default_value
        
        return wrapper

# 全局异常处理器实例
exception_handler = ExceptionHandler()

# 便捷装饰器
def api_retry(max_retries: int = 3, delay: float = 1.0):
    """API重试装饰器"""
    def decorator(func):
        return exception_handler.handle_api_error(func, max_retries, delay)
    return decorator

def safe_data_processing(default_value: Any = None):
    """安全数据处理装饰器"""
    def decorator(func):
        return exception_handler.handle_data_error(func, default_value)
    return decorator

def log_exceptions(func):
    """记录异常装饰器"""
    return exception_handler.log_and_reraise(func)

def suppress_errors(*exceptions, default_value=None):
    """抑制错误装饰器"""
    def decorator(func):
        return exception_handler.suppress_exceptions(func, exceptions, default_value)
    return decorator

def safe_call(default_return=None, log_error=True):
    """安全调用装饰器，捕获异常并返回默认值"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"函数 {func.__name__} 执行失败: {e}")
                return default_return
        return wrapper
    return decorator