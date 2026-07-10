"""
统一配置管理系统

支持从多个来源加载配置，优先级为：
环境变量 > 配置文件 > 默认值

功能特性:
- 支持从环境变量加载配置
- 支持从.env文件加载配置
- 配置验证和类型转换
- 单例模式管理全局配置
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from dotenv import load_dotenv

from src.core.exceptions import ConfigurationError

# 类型变量，用于类型注解
T = TypeVar("T", bound="Config")


@dataclass
class CacheConfig:
    """
    缓存配置类

    Attributes:
        enabled: 是否启用缓存
        ttl: 缓存过期时间(秒)，默认3600秒(1小时)
        backend: 缓存后端类型，可选: memory(内存), file(文件), redis(Redis)
        max_size: 内存缓存最大条目数
    """
    enabled: bool = True  # 默认启用缓存
    ttl: int = 3600  # 缓存存活时间: 1小时
    backend: str = "memory"  # 缓存后端: 内存缓存
    max_size: int = 1000  # 最大缓存条目数


@dataclass
class LoggingConfig:
    """
    日志配置类

    Attributes:
        level: 日志级别，可选: DEBUG, INFO, WARNING, ERROR, CRITICAL
        format: 日志格式字符串
        file: 日志文件路径，None表示不输出到文件
        console: 是否输出到控制台
    """
    level: str = "INFO"  # 默认日志级别
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # 日志格式
    file: Optional[str] = None  # 日志文件路径
    console: bool = True  # 是否输出到控制台


@dataclass
class APIConfig:
    """
    API配置类

    Attributes:
        openai_api_key: OpenAI API密钥
        openai_base_url: OpenAI API基础URL
        openai_model: 默认使用的模型名称
        openai_timeout: API请求超时时间(秒)
    """
    openai_api_key: str = ""  # API密钥
    openai_base_url: str = "https://api.openai.com/v1"  # API地址
    openai_model: str = "gpt-4"  # 默认模型
    openai_timeout: int = 60  # 请求超时: 60秒


@dataclass
class DataConfig:
    """
    数据源配置类

    Attributes:
        data_dir: 股票数据目录
        rps_dir: RPS(相对价格强度)数据目录
        models_dir: ML模型文件目录
        cache_dir: 缓存文件目录
        reports_dir: 报告输出目录
        financial_reports_dir: 财务报告输出目录
        training_data_dir: 训练数据目录
        selection_results_file: 选股结果文件名
        export_csv: 是否导出CSV格式训练数据
        enable_financial_reports: 是否启用财务报告生成
        financial_analysis_timeout: 财务分析超时时间(秒)
        concurrent_reports: 财务分析并发数量
    """
    data_dir: str = "market_data"  # 市场数据目录
    rps_dir: str = "rps_results"  # RPS计算结果目录
    models_dir: str = "models"  # 训练模型目录
    cache_dir: str = "cache"  # 缓存目录
    reports_dir: str = "reports"  # 报告输出目录
    financial_reports_dir: str = "reports"  # 财务报告输出目录
    training_data_dir: str = "training_data"  # 训练数据目录
    selection_results_file: str = "stock_selection_results.json"  # 选股结果文件
    export_csv: bool = False  # 是否导出CSV
    enable_financial_reports: bool = True  # 是否启用财务报告(默认启用，使用内部LLM)
    financial_analysis_timeout: int = 300  # 财务分析超时(秒)
    concurrent_reports: int = 1  # 财务分析并发数


@dataclass
class ParallelConfig:
    """
    并行处理配置类

    Attributes:
        max_workers: 最大工作进程数，None表示使用CPU核心数
        chunk_size: 每个工作进程一次处理的任务数
        enable_progress: 是否显示进度条
    """
    max_workers: Optional[int] = None  # 最大工作进程，None=自动检测CPU核心数
    chunk_size: int = 1  # 任务分块大小
    enable_progress: bool = True  # 是否启用进度条


@dataclass
class Config:
    """
    主配置类

    管理所有子模块配置，支持从多种来源加载：
    - 环境变量
    - .env文件
    - 默认值

    Attributes:
        cache: 缓存配置
        logging: 日志配置
        api: API配置
        data: 数据配置
        parallel: 并行处理配置
        work_dir: 工作目录
        conda_env: Conda环境名称

    Example:
        >>> # 使用默认配置
        >>> config = Config.load()
        >>>
        >>> # 指定.env文件路径
        >>> config = Config.load(env_file="custom.env")
        >>>
        >>> # 从环境变量创建
        >>> config = Config.from_env()
    """

    # ========== 子配置 ==========
    cache: CacheConfig = field(default_factory=CacheConfig)  # 缓存配置
    logging: LoggingConfig = field(default_factory=LoggingConfig)  # 日志配置
    api: APIConfig = field(default_factory=APIConfig)  # API配置
    data: DataConfig = field(default_factory=DataConfig)  # 数据配置
    parallel: ParallelConfig = field(default_factory=ParallelConfig)  # 并行配置

    # ========== 兼容性字段 ==========
    work_dir: str = "."  # 工作目录
    conda_env: str = "ashare-llm-analyst"  # Conda环境名

    # ========== 单例管理 ==========
    _instance: Optional["Config"] = None  # 全局单例实例

    @classmethod
    def load(
        cls: Type[T],
        config_path: Optional[str] = None,
        env_file: Optional[str] = None,
    ) -> T:
        """
        加载配置

        加载优先级: 环境变量 > .env文件 > 默认值

        Args:
            config_path: 配置文件路径(YAML或.env格式)
            env_file: .env文件路径，默认为工作目录下的.env

        Returns:
            配置实例

        Example:
            >>> config = Config.load()
            >>> config = Config.load(env_file="production.env")
        """
        # 步骤1: 加载.env文件
        if env_file:
            # 使用指定的env文件
            load_dotenv(env_file)
        else:
            # 优先加载项目根目录 .env（单一配置来源），回退到当前工作目录
            project_root = Path(__file__).resolve().parent.parent.parent
            candidate_paths = [
                project_root / ".env",   # 项目根 .env（推荐，单一来源）
                Path.cwd() / ".env",     # 当前目录 .env（回退）
            ]
            for env_path in candidate_paths:
                if env_path.exists():
                    load_dotenv(str(env_path))
                    break

        # 步骤2: 创建默认配置实例
        instance = cls()

        # 步骤3: 应用环境变量覆盖
        instance._apply_env_overrides()

        # 步骤4: 保存为单例
        Config._instance = instance
        return instance

    def _apply_env_overrides(self) -> None:
        """
        应用环境变量覆盖配置

        从环境变量读取配置并覆盖默认值
        支持的环境变量:
        - CACHE_ENABLED, CACHE_TTL, CACHE_BACKEND
        - LOG_LEVEL, LOG_FILE
        - OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_TIMEOUT
        - DATA_DIR, RPS_DIR, MODELS_DIR
        - MAX_WORKERS
        - WORK_DIR, CONDA_ENV
        """
        # ========== 缓存配置 ==========
        self.cache.enabled = self._get_env_bool("CACHE_ENABLED", self.cache.enabled)
        self.cache.ttl = self._get_env_int("CACHE_TTL", self.cache.ttl)
        self.cache.backend = os.getenv("CACHE_BACKEND", self.cache.backend)

        # ========== 日志配置 ==========
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.file = os.getenv("LOG_FILE", self.logging.file)

        # ========== API配置 ==========
        self.api.openai_api_key = os.getenv("OPENAI_API_KEY", self.api.openai_api_key)
        self.api.openai_base_url = os.getenv(
            "OPENAI_BASE_URL", self.api.openai_base_url
        )
        self.api.openai_model = os.getenv("OPENAI_MODEL", self.api.openai_model)
        self.api.openai_timeout = self._get_env_int(
            "OPENAI_TIMEOUT", self.api.openai_timeout
        )

        # ========== 数据配置 ==========
        self.data.data_dir = os.getenv("DATA_DIR", self.data.data_dir)
        self.data.rps_dir = os.getenv("RPS_DIR", self.data.rps_dir)
        self.data.models_dir = os.getenv("MODELS_DIR", self.data.models_dir)

        # ========== 并行配置 ==========
        self.parallel.max_workers = self._get_env_int(
            "MAX_WORKERS", self.parallel.max_workers
        )

        # ========== 工作目录 ==========
        self.work_dir = os.getenv("WORK_DIR", self.work_dir)
        self.conda_env = os.getenv("CONDA_ENV", self.conda_env)

    @classmethod
    def from_env(cls: Type[T]) -> T:
        """
        仅从环境变量创建配置

        Returns:
            配置实例
        """
        return cls.load()

    @classmethod
    def get_instance(cls: Type[T]) -> T:
        """
        获取全局单例实例

        如果实例不存在则自动加载

        Returns:
            配置单例实例
        """
        if cls._instance is None:
            return cls.load()
        return cls._instance

    def validate(self) -> None:
        """
        验证配置有效性

        检查:
        - API密钥格式
        - 日志级别有效性
        - 目录路径有效性(可选)

        Raises:
            ConfigurationError: 配置验证失败时抛出
        """
        errors = []

        # 验证API密钥存在性（内部 Qwen 部署的 key 未必 sk- 开头，仅校验非空）
        if not self.api.openai_api_key:
            errors.append("OPENAI_API_KEY 未配置")

        # 验证日志级别
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level.upper() not in valid_levels:
            errors.append(f"无效的LOG_LEVEL: {self.logging.level}")

        # 汇总错误并抛出异常
        if errors:
            raise ConfigurationError(
                f"配置验证失败: {', '.join(errors)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        用于序列化和调试输出

        Returns:
            配置字典
        """
        return {
            "cache": {
                "enabled": self.cache.enabled,
                "ttl": self.cache.ttl,
                "backend": self.cache.backend,
            },
            "logging": {
                "level": self.logging.level,
                "file": self.logging.file,
                "console": self.logging.console,
            },
            "api": {
                "openai_base_url": self.api.openai_base_url,
                "openai_model": self.api.openai_model,
                "openai_timeout": self.api.openai_timeout,
            },
            "data": {
                "data_dir": self.data.data_dir,
                "rps_dir": self.data.rps_dir,
                "models_dir": self.data.models_dir,
            },
            "parallel": {
                "max_workers": self.parallel.max_workers,
                "chunk_size": self.parallel.chunk_size,
                "enable_progress": self.parallel.enable_progress,
            },
        }

    @staticmethod
    def _get_env_int(key: str, default: Optional[int] = None) -> Optional[int]:
        """
        从环境变量获取整数值

        Args:
            key: 环境变量名
            default: 默认值

        Returns:
            整数值或默认值
        """
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _get_env_bool(key: str, default: bool = False) -> bool:
        """
        从环境变量获取布尔值

        Args:
            key: 环境变量名
            default: 默认值

        Returns:
            布尔值

        Note:
            可接受的真值: 1, true, yes, on
            可接受的假值: 0, false, no, off
        """
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes", "on")


def get_config() -> Config:
    """
    获取全局配置实例的快捷函数

    Returns:
        配置单例实例
    """
    return Config.get_instance()


def load_config(
    config_path: Optional[str] = None, env_file: Optional[str] = None
) -> Config:
    """
    加载并返回配置的快捷函数

    Args:
        config_path: 配置文件路径
        env_file: .env文件路径

    Returns:
        配置实例
    """
    return Config.load(config_path, env_file)
