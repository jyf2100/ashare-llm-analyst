"""
统一日志系统

提供结构化日志记录，支持多种处理器和一致的格式化输出

功能特性:
- 彩色控制台输出(不同级别不同颜色)
- 文件日志输出
- 结构化日志格式
- LoggerMixin混入类支持
- 全局日志配置管理
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.core.config import Config, LoggingConfig


class StructuredFormatter(logging.Formatter):
    """
    结构化日志格式化器

    输出一致的日志格式，包含时间、级别、模块名和消息
    支持彩色控制台输出，不同日志级别使用不同颜色

    Attributes:
        use_colors: 是否启用彩色输出

    颜色映射:
        DEBUG: 青色 (Cyan)
        INFO: 绿色 (Green)
        WARNING: 黄色 (Yellow)
        ERROR: 红色 (Red)
        CRITICAL: 洋红 (Magenta)

    Example:
        >>> formatter = StructuredFormatter(use_colors=True)
        >>> handler.setFormatter(formatter)
    """

    # 控制台输出的颜色代码(ANSI转义序列)
    COLORS = {
        logging.DEBUG: "\033[36m",  # 青色 - 调试信息
        logging.INFO: "\033[32m",  # 绿色 - 一般信息
        logging.WARNING: "\033[33m",  # 黄色 - 警告信息
        logging.ERROR: "\033[31m",  # 红色 - 错误信息
        logging.CRITICAL: "\033[35m",  # 洋红 - 严重错误
    }
    RESET = "\033[0m"  # 重置颜色

    def __init__(self, use_colors: bool = True):
        """
        初始化格式化器

        Args:
            use_colors: 是否启用彩色输出
        """
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录

        Args:
            record: 日志记录对象

        Returns:
            格式化后的日志字符串
        """
        # 基础日志格式
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

        # 如果启用彩色且输出到终端，添加颜色
        if self.use_colors and hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            # 为控制台输出添加颜色
            level_color = self.COLORS.get(record.levelno, "")
            record.levelname = f"{level_color}{record.levelname}{self.RESET}"

        return formatter.format(record)


# ========== 全局日志管理 ==========

# 全局日志缓存 {logger_name: Logger}
_loggers: dict[str, logging.Logger] = {}
# 全局配置标志
_configured: bool = False


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    设置全局日志配置

    配置根日志记录器，添加控制台和文件处理器
    该函数只执行一次，后续调用会被忽略

    Args:
        config: 日志配置，None则使用全局配置或默认配置

    Example:
        >>> # 使用默认配置
        >>> setup_logging()
        >>>
        >>> # 使用自定义配置
        >>> config = LoggingConfig(level="DEBUG", file="app.log")
        >>> setup_logging(config)
    """
    global _configured

    # 避免重复配置
    if _configured:
        return

    # 获取配置
    if config is None:
        # 尝试从全局Config获取
        try:
            app_config = Config.get_instance()
            config = app_config.logging
        except Exception:
            # 使用默认配置
            config = LoggingConfig()

    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # 清除现有处理器
    root_logger.handlers.clear()

    # 控制台处理器
    if config.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter(use_colors=True))
        root_logger.addHandler(console_handler)

    # 文件处理器
    if config.file:
        log_path = Path(config.file)
        # 确保日志目录存在
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter(use_colors=False))
        root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    获取指定名称的日志记录器

    如果日志系统未配置，会自动调用setup_logging
    支持日志记录器缓存，相同名称返回同一实例

    Args:
        name: 日志记录器名称(通常使用模块的__name__)
        config: 可选的日志配置

    Returns:
        日志记录器实例

    Example:
        >>> from src.core.logger import get_logger
        >>>
        >>> # 在模块中使用
        >>> logger = get_logger(__name__)
        >>> logger.info("Hello, world!")
        >>> logger.error("Something went wrong!")
    """
    global _loggers

    # 确保日志系统已配置
    if not _configured:
        setup_logging(config)

    # 返回缓存的日志记录器
    if name in _loggers:
        return _loggers[name]

    # 创建新的日志记录器
    logger = logging.getLogger(name)
    _loggers[name] = logger

    return logger


class LoggerMixin:
    """
    日志混入类

    为任何类添加日志功能的混入类
    通过属性访问自动创建和管理日志记录器

    Attributes:
        logger: 该类的日志记录器(懒加载)

    Example:
        >>> class MyClass(LoggerMixin):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.logger.info("初始化完成")
        >>>
        >>>     def do_work(self):
        >>>         self.logger.debug("开始工作...")
        >>>         try:
        >>>             # 执行操作
        >>>             self.logger.info("工作完成")
        >>>         except Exception as e:
        >>>             self.logger.error(f"工作失败: {e}")
    """

    @property
    def logger(self) -> logging.Logger:
        """
        获取该类的日志记录器

        使用懒加载模式，首次访问时创建
        日志记录器名称为: 模块名.类名

        Returns:
            该类的日志记录器实例
        """
        if not hasattr(self, "_logger"):
            # 创建日志记录器: module.ClassName
            name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = get_logger(name)
        return self._logger
