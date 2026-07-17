"""
股票分析系统自定义异常层次结构

提供结构化错误处理，包含上下文信息和错误码

功能特性:
- 基础异常类提供统一错误格式
- 每种错误类型都有专门的错误码
- 上下文信息支持错误追踪
- 序列化为字典用于日志记录
"""

from typing import Any, Dict, Optional
from typing_extensions import TypeAlias

# ========== 类型别名 ==========

ErrorCode: TypeAlias = str  # 错误码类型
ContextDict: TypeAlias = Dict[str, Any]  # 上下文字典类型


class StockAnalysisError(Exception):
    """
    股票分析系统基础异常类

    所有自定义异常的基类，提供统一的错误处理格式

    Attributes:
        message: 人类可读的错误消息
        context: 额外的上下文信息字典
        error_code: 机器可读的错误码

    错误码规范:
        E0000: 未知错误
        E1xxx: 数据获取错误
        E2xxx: 模型加载错误
        E3xxx: 分析执行错误
        E4xxx: 配置错误
        E5xxx: 缓存错误
        E6xxx: 验证错误

    Example:
        >>> try:
        >>>     # 某些操作
        >>>     pass
        >>> except StockAnalysisError as e:
        >>>     print(f"错误码: {e.error_code}")
        >>>     print(f"消息: {e.message}")
        >>>     print(f"上下文: {e.context}")
        >>>     # 转换为字典用于日志
        >>>     error_dict = e.to_dict()
    """

    # ========== 错误码常量 ==========
    UNKNOWN = "E0000"  # 未知错误
    DATA_FETCH_FAILED = "E1000"  # 数据获取失败
    MODEL_LOAD_FAILED = "E2000"  # 模型加载失败
    ANALYSIS_FAILED = "E3000"  # 分析执行失败
    CONFIG_INVALID = "E4000"  # 配置无效
    CACHE_ERROR = "E5000"  # 缓存错误
    VALIDATION_ERROR = "E6000"  # 验证错误

    def __init__(
        self,
        message: str,
        context: Optional[ContextDict] = None,
        error_code: Optional[ErrorCode] = None,
    ) -> None:
        """
        初始化异常

        Args:
            message: 错误消息描述
            context: 额外的上下文信息字典
            error_code: 错误码，None则使用UNKNOWN
        """
        self.message = message
        self.context = context or {}
        self.error_code = error_code or self.UNKNOWN
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """
        格式化包含上下文的错误消息

        Returns:
            格式化的错误消息字符串
        """
        parts = [f"[{self.error_code}] {self.message}"]
        if self.context:
            # 添加上下文信息
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"({context_str})")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        将异常转换为字典

        用于日志记录或序列化

        Returns:
            包含error_code、message、context、type的字典
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "type": self.__class__.__name__,
        }


class DataFetchError(StockAnalysisError):
    """
    数据获取失败异常

    当从外部数据源获取数据失败时抛出

    典型原因:
    - 网络连接错误
    - API调用失败
    - 文件未找到
    - 数据格式无效

    错误码:
        E1001: 网络错误
        E1002: API错误
        E1003: 文件未找到
        E1004: 格式无效

    Example:
        >>> raise DataFetchError(
        >>>     "无法连接到数据源",
        >>>     source="baostock",
        >>>     context={"url": "http://example.com"}
        >>> )
    """

    NETWORK_ERROR = "E1001"  # 网络错误
    API_ERROR = "E1002"  # API错误
    FILE_NOT_FOUND = "E1003"  # 文件未找到
    INVALID_FORMAT = "E1004"  # 格式无效

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        context: Optional[ContextDict] = None,
        error_code: Optional[ErrorCode] = None,
    ) -> None:
        """
        初始化数据获取错误

        Args:
            message: 错误消息
            source: 数据源名称(可选)
            context: 额外上下文信息
            error_code: 错误码(可选)。None 则回退到 ``DATA_FETCH_FAILED`` (E1000)；
                可传本类的细分码常量（如 :attr:`NETWORK_ERROR` / :attr:`API_ERROR`）
                以便调用方按码区分网络错误与 API/参数错误。
        """
        ctx = context or {}
        if source:
            ctx["source"] = source
        super().__init__(message, ctx, error_code or self.DATA_FETCH_FAILED)


class ModelLoadError(StockAnalysisError):
    """
    模型加载失败异常

    当加载机器学习模型文件失败时抛出

    典型原因:
    - 模型文件未找到
    - 模型版本不兼容
    - 模型文件损坏
    - 缺少依赖库

    错误码:
        E2001: 文件未找到
        E2002: 版本不匹配
        E2003: 文件损坏
        E2004: 缺少依赖

    Example:
        >>> raise ModelLoadError(
        >>>     "模型文件不存在",
        >>>     model_name="xgboost_predictor",
        >>>     context={"path": "/models/predictor.pkl"}
        >>> )
    """

    FILE_NOT_FOUND = "E2001"  # 文件未找到
    VERSION_MISMATCH = "E2002"  # 版本不匹配
    CORRUPTED_FILE = "E2003"  # 文件损坏
    MISSING_DEPENDENCY = "E2004"  # 缺少依赖

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        context: Optional[ContextDict] = None,
    ) -> None:
        """
        初始化模型加载错误

        Args:
            message: 错误消息
            model_name: 模型名称(可选)
            context: 额外上下文信息
        """
        ctx = context or {}
        if model_name:
            ctx["model_name"] = model_name
        super().__init__(message, ctx, self.MODEL_LOAD_FAILED)


class AnalysisError(StockAnalysisError):
    """
    分析执行失败异常

    当执行股票分析过程中出现错误时抛出

    典型原因:
    - 数据不足
    - 参数无效
    - 计算错误
    - 外部服务失败

    错误码:
        E3001: 数据不足
        E3002: 参数无效
        E3003: 计算错误
        E3004: 服务错误

    Example:
        >>> raise AnalysisError(
        >>>     "需要至少100天的数据",
        >>>     stock_code="000001",
        >>>     context={"available_days": 30, "required_days": 100}
        >>> )
    """

    INSUFFICIENT_DATA = "E3001"  # 数据不足
    INVALID_PARAMETER = "E3002"  # 参数无效
    CALCULATION_ERROR = "E3003"  # 计算错误
    SERVICE_ERROR = "E3004"  # 服务错误

    def __init__(
        self,
        message: str,
        stock_code: Optional[str] = None,
        context: Optional[ContextDict] = None,
    ) -> None:
        """
        初始化分析执行错误

        Args:
            message: 错误消息
            stock_code: 股票代码(可选)
            context: 额外上下文信息
        """
        ctx = context or {}
        if stock_code:
            ctx["stock_code"] = stock_code
        super().__init__(message, ctx, self.ANALYSIS_FAILED)


class ConfigurationError(StockAnalysisError):
    """
    配置无效异常

    当配置缺失或无效时抛出

    典型原因:
    - 缺少必需配置
    - 配置值无效
    - 配置文件错误

    错误码:
        E4001: 缺少必需配置
        E4002: 值无效
        E4003: 文件错误

    Example:
        >>> raise ConfigurationError(
        >>>     "缺少API密钥",
        >>>     config_key="OPENAI_API_KEY"
        >>> )
    """

    MISSING_REQUIRED = "E4001"  # 缺少必需配置
    INVALID_VALUE = "E4002"  # 值无效
    FILE_ERROR = "E4003"  # 文件错误

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        context: Optional[ContextDict] = None,
    ) -> None:
        """
        初始化配置错误

        Args:
            message: 错误消息
            config_key: 配置键名(可选)
            context: 额外上下文信息
        """
        ctx = context or {}
        if config_key:
            ctx["config_key"] = config_key
        super().__init__(message, ctx, self.CONFIG_INVALID)


class CacheError(StockAnalysisError):
    """
    缓存操作失败异常

    当缓存操作失败时抛出

    典型原因:
    - 缓存后端不可用
    - 序列化错误
    - 缓存数据损坏

    错误码:
        E5001: 后端错误
        E5002: 序列化错误
        E5003: 缓存损坏

    Example:
        >>> raise CacheError(
        >>>     "无法连接到Redis",
        >>>     cache_key="stock_data:000001"
        >>> )
    """

    BACKEND_ERROR = "E5001"  # 后端错误
    SERIALIZATION_ERROR = "E5002"  # 序列化错误
    CORRUPTION = "E5003"  # 缓存损坏

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        context: Optional[ContextDict] = None,
    ) -> None:
        """
        初始化缓存错误

        Args:
            message: 错误消息
            cache_key: 缓存键(可选)
            context: 额外上下文信息
        """
        ctx = context or {}
        if cache_key:
            ctx["cache_key"] = cache_key
        super().__init__(message, ctx, self.CACHE_ERROR)


class ValidationError(StockAnalysisError):
    """
    输入验证失败异常

    当输入数据验证失败时抛出

    典型原因:
    - 股票代码格式无效
    - 参数超出范围
    - 业务规则违规

    错误码:
        E6001: 格式无效
        E6002: 超出范围
        E6003: 规则违规

    Example:
        >>> raise ValidationError(
        >>>     "股票代码格式无效",
        >>>     field_name="stock_code",
        >>>     context={"value": "abc", "pattern": "^d{6}$"}
        >>> )
    """

    INVALID_FORMAT = "E6001"  # 格式无效
    OUT_OF_RANGE = "E6002"  # 超出范围
    RULE_VIOLATION = "E6003"  # 规则违规

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        context: Optional[ContextDict] = None,
    ) -> None:
        """
        初始化验证错误

        Args:
            message: 错误消息
            field_name: 字段名(可选)
            context: 额外上下文信息
        """
        ctx = context or {}
        if field_name:
            ctx["field"] = field_name
        super().__init__(message, ctx, self.VALIDATION_ERROR)


# ========== 异常工具函数 ==========

def format_exception(exc: Exception) -> str:
    """
    格式化任何异常用于日志输出

    Args:
        exc: 要格式化的异常

    Returns:
        格式化的错误消息字符串

    Example:
        >>> try:
        >>>     risky_operation()
        >>> except Exception as e:
        >>>     logger.error(format_exception(e))
    """
    if isinstance(exc, StockAnalysisError):
        return exc.format_message()
    return f"{exc.__class__.__name__}: {exc}"


def get_error_code(exc: Exception) -> ErrorCode:
    """
    从异常中提取错误码

    Args:
        exc: 要提取错误码的异常

    Returns:
        错误码字符串，非自定义异常返回UNKNOWN

    Example:
        >>> try:
        >>>     risky_operation()
        >>> except Exception as e:
        >>>     code = get_error_code(e)
        >>>     logger.error(f"错误码: {code}")
    """
    if isinstance(exc, StockAnalysisError):
        return exc.error_code
    return StockAnalysisError.UNKNOWN
