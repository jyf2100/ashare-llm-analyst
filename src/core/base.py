"""
基础类和接口

为股票分析系统提供公共功能：
- 分析器基类
- 数据提供者接口
- 信号类
- 模型基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from src.core.config import Config, get_config
from src.core.logger import LoggerMixin, get_logger

# 类型变量，用于泛型支持
T = TypeVar("T")


class AnalyzerBase(LoggerMixin, ABC):
    """
    分析器基类

    所有分析器的基类，提供通用功能：
    - 配置管理
    - 日志记录
    - 缓存支持
    - 生命周期管理

    Attributes:
        config: 配置实例
        _initialized: 是否已初始化

    Example:
        >>> class MyAnalyzer(AnalyzerBase):
        >>>     def analyze(self, data):
        >>>         self.logger.info("开始分析")
        >>>         # 使用self.config访问配置
        >>>         return result
        >>>
        >>> # 使用上下文管理器
        >>> with MyAnalyzer() as analyzer:
        >>>     result = analyzer.analyze(data)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化分析器

        Args:
            config: 配置实例，None则使用全局配置
        """
        self.config = config or get_config()  # 使用全局配置或自定义配置
        self._initialized = False  # 初始化标志

    def initialize(self) -> None:
        """
        初始化分析器

        在构造后调用，用于设置资源
        子类可重写此方法实现自定义初始化
        """
        if self._initialized:
            # 已初始化，直接返回
            return

        self._initialized = True
        self.logger.debug(f"{self.__class__.__name__} 初始化完成")

    def cleanup(self) -> None:
        """
        清理资源

        当分析器不再需要时调用
        子类可重写此方法实现自定义清理
        """
        self._initialized = False
        self.logger.debug(f"{self.__class__.__name__} 资源已清理")

    def __enter__(self):
        """
        上下文管理器入口

        Returns:
            self
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口

        Args:
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常追踪

        Returns:
            False (不抑制异常)
        """
        self.cleanup()
        return False

    @property
    def cache_enabled(self) -> bool:
        """
        检查缓存是否启用

        Returns:
            缓存是否启用
        """
        return self.config.cache.enabled

    def cache_key(self, *parts: str) -> str:
        """
        生成缓存键

        Args:
            *parts: 键的组成部分

        Returns:
            缓存键字符串
        """
        base = self.__class__.__name__
        return ":".join([base] + list(parts))


class DataProvider(ABC, Generic[T]):
    """
    数据提供者接口

    所有数据提供者都应实现此接口

    Type Parameters:
        T: 数据类型 (如 pd.DataFrame)

    Example:
        >>> class CSVDataProvider(DataProvider[pd.DataFrame]):
        >>>     def load(self, stock_code):
        >>>         return pd.read_csv(f"{stock_code}.csv")
        >>>
        >>>     def save(self, stock_code, data):
        >>>         data.to_csv(f"{stock_code}.csv")
        >>>
        >>>     def exists(self, stock_code):
        >>>         return os.path.exists(f"{stock_code}.csv")
    """

    @abstractmethod
    def load(self, identifier: str, **kwargs) -> Optional[T]:
        """
        加载指定标识符的数据

        Args:
            identifier: 股票代码、文件路径或其他标识符
            **kwargs: 额外参数

        Returns:
            加载的数据，未找到则返回None
        """
        pass

    @abstractmethod
    def save(self, identifier: str, data: T, **kwargs) -> bool:
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


class FileSystemProvider(DataProvider[T]):
    """
    基于文件系统的数据提供者基类

    为文件存储提供通用的加载/保存逻辑

    Attributes:
        base_dir: 数据文件基础目录
        logger: 日志记录器

    Example:
        >>> class CSVProvider(FileSystemProvider[pd.DataFrame]):
        >>>     def load_file(self, path):
        >>>         return pd.read_csv(path)
        >>>
        >>>     def save_file(self, path, data):
        >>>         data.to_csv(path)
    """

    def __init__(self, base_dir: str):
        """
        初始化文件系统提供者

        Args:
            base_dir: 数据文件基础目录
        """
        self.base_dir = base_dir
        self.logger = get_logger(self.__class__.__name__)

    def get_path(self, identifier: str, **kwargs) -> str:
        """
        获取标识符的完整文件路径

        Args:
            identifier: 标识符
            **kwargs: 额外参数

        Returns:
            文件路径
        """
        import os

        return os.path.join(self.base_dir, f"{identifier}.csv")

    def load(self, identifier: str, **kwargs) -> Optional[T]:
        """
        从文件加载数据

        Args:
            identifier: 标识符
            **kwargs: 额外参数

        Returns:
            加载的数据，失败返回None
        """
        path = self.get_path(identifier, **kwargs)
        if not self.exists(identifier):
            self.logger.warning(f"文件不存在: {path}")
            return None

        try:
            return self.load_file(path)
        except Exception as e:
            self.logger.error(f"加载文件失败 {path}: {e}")
            return None

    def save(self, identifier: str, data: T, **kwargs) -> bool:
        """
        保存数据到文件

        Args:
            identifier: 标识符
            data: 要保存的数据
            **kwargs: 额外参数

        Returns:
            成功返回True，失败返回False
        """
        path = self.get_path(identifier, **kwargs)
        try:
            self.save_file(path, data)
            return True
        except Exception as e:
            self.logger.error(f"保存文件失败 {path}: {e}")
            return False

    def exists(self, identifier: str, **kwargs) -> bool:
        """
        检查文件是否存在

        Args:
            identifier: 标识符
            **kwargs: 额外参数

        Returns:
            文件存在返回True，否则返回False
        """
        import os

        path = self.get_path(identifier, **kwargs)
        return os.path.exists(path)

    def list_available(self, **kwargs) -> List[str]:
        """
        列出可用的文件

        Args:
            **kwargs: 过滤参数

        Returns:
            可用标识符列表(不含扩展名)
        """
        import os

        if not os.path.exists(self.base_dir):
            return []

        files = os.listdir(self.base_dir)
        # 移除文件扩展名
        return [f.replace(".csv", "") for f in files if f.endswith(".csv")]

    @abstractmethod
    def load_file(self, path: str) -> T:
        """
        从文件路径加载数据

        Args:
            path: 文件路径

        Returns:
            加载的数据
        """
        pass

    @abstractmethod
    def save_file(self, path: str, data: T) -> None:
        """
        保存数据到文件路径

        Args:
            path: 文件路径
            data: 要保存的数据
        """
        pass


class BaseModel(LoggerMixin):
    """
    ML/AI模型基类

    提供:
    - 模型加载/保存
    - 预测接口
    - 版本管理

    Attributes:
        model_name: 模型名称
        version: 模型版本
        _model: 模型实例
        _loaded: 是否已加载

    Example:
        >>> class MyModel(BaseModel):
        >>>     def load(self):
        >>>         self._model = joblib.load("model.pkl")
        >>>         self._loaded = True
        >>>
        >>>     def predict(self, data):
        >>>         self.ensure_loaded()
        >>>         return self._model.predict(data)
    """

    def __init__(self, model_name: str, version: str = "1.0"):
        """
        初始化模型

        Args:
            model_name: 模型名称
            version: 模型版本
        """
        self.model_name = model_name
        self.version = version
        self._model = None  # 模型实例
        self._loaded = False  # 加载状态

    @abstractmethod
    def load(self) -> bool:
        """
        加载模型

        Returns:
            成功返回True
        """
        pass

    @abstractmethod
    def save(self) -> bool:
        """
        保存模型

        Returns:
            成功返回True
        """
        pass

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """
        对数据进行预测

        Args:
            data: 输入数据

        Returns:
            预测结果
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """
        检查模型是否已加载

        Returns:
            模型是否已加载
        """
        return self._loaded

    def ensure_loaded(self) -> None:
        """
        确保模型已加载，必要时加载

        如果模型未加载则自动调用load()方法
        """
        if not self._loaded:
            self.load()


class Signal:
    """
    交易信号类

    表示一个交易信号(买入/卖出/持有)

    Attributes:
        type: 信号类型 (buy=买入, sell=卖出, hold=持有)
        strength: 信号强度 (0-100)
        reason: 信号原因说明
        metadata: 附加元数据

    Example:
        >>> # 创建买入信号
        >>> signal = Signal(
        ...     signal_type=Signal.BUY,
        ...     strength=75,
        ...     reason="MACD金叉",
        ...     metadata={"indicator": "MACD"}
        ... )
        >>>
        >>> if signal.is_buy():
        >>>     print(f"买入信号: {signal.reason}")
        >>>
        >>> # 转换为字典
        >>> signal_dict = signal.to_dict()
    """

    # 信号类型常量
    BUY = "buy"      # 买入信号
    SELL = "sell"    # 卖出信号
    HOLD = "hold"    # 持有信号

    def __init__(
        self,
        signal_type: str,
        strength: float,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化信号

        Args:
            signal_type: 信号类型 (BUY/SELL/HOLD)
            strength: 信号强度 (0-100)
            reason: 信号原因
            metadata: 附加元数据
        """
        self.type = signal_type
        self.strength = strength
        self.reason = reason
        self.metadata = metadata or {}  # 空字典作为默认值

    def is_buy(self) -> bool:
        """
        是否为买入信号

        Returns:
            是买入信号返回True
        """
        return self.type == self.BUY

    def is_sell(self) -> bool:
        """
        是否为卖出信号

        Returns:
            是卖出信号返回True
        """
        return self.type == self.SELL

    def is_hold(self) -> bool:
        """
        是否为持有信号

        Returns:
            是持有信号返回True
        """
        return self.type == self.HOLD

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            包含type、strength、reason、metadata的字典
        """
        return {
            "type": self.type,
            "strength": self.strength,
            "reason": self.reason,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """
        字符串表示

        Returns:
            格式化的信号字符串
        """
        return f"Signal({self.type}, strength={self.strength:.1f}, reason={self.reason})"
