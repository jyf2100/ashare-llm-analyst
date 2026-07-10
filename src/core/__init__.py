"""
Core infrastructure module for stock analysis system.

This module provides shared infrastructure including:
- Configuration management
- Logging
- Custom exceptions
- Caching
- Parallel processing
- Base classes
"""

from src.core.config import Config, get_config, load_config
from src.core.logger import get_logger, setup_logging
from src.core.exceptions import (
    StockAnalysisError,
    DataFetchError,
    ModelLoadError,
    AnalysisError,
    ConfigurationError,
)
from src.core.cache import cached, CacheConfig, clear_cache
from src.core.parallel import ParallelProcessor
from src.core.base import AnalyzerBase, DataProvider

__version__ = "0.1.0"

__all__ = [
    # Config
    "Config",
    "get_config",
    "load_config",
    # Logger
    "get_logger",
    "setup_logging",
    # Exceptions
    "StockAnalysisError",
    "DataFetchError",
    "ModelLoadError",
    "AnalysisError",
    "ConfigurationError",
    # Cache
    "cached",
    "CacheConfig",
    "clear_cache",
    # Parallel
    "ParallelProcessor",
    # Base
    "AnalyzerBase",
    "DataProvider",
]
