"""
性能监控模块

提供数据管道性能监控功能
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """
    性能指标数据类

    Attributes:
        cpu_percent: CPU使用率
        memory_mb: 内存使用量(MB)
        memory_percent: 内存使用百分比
        disk_usage_mb: 磁盘使用量(MB)
        timestamp: 记录时间
    """
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_usage_mb: float
    timestamp: datetime = field(default_factory=datetime.now)


class PipelineMonitor(AnalyzerBase):
    """
    管道监控器

    监控数据管道的性能和资源使用情况

    Attributes:
        config: 配置实例
        metrics_history: 指标历史记录

    Example:
        >>> monitor = PipelineMonitor()
        >>> metrics = monitor.collect_metrics()
        >>> print(f"CPU: {metrics.cpu_percent}%")
    """

    def __init__(self, config: Optional[Config] = None):
        """初始化监控器"""
        super().__init__(config)
        self.metrics_history: List[PerformanceMetrics] = []

    def collect_metrics(self) -> Dict[str, Any]:
        """
        收集当前性能指标

        Returns:
            指标字典
        """
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil未安装，无法收集性能指标")
            return {
                "cpu_percent": 0,
                "cpu_count": 0,
                "memory_mb": 0,
                "memory_percent": 0,
                "total_memory_gb": 0,
                "disk_usage_mb": 0,
                "disk_free_gb": 0,
                "timestamp": datetime.now().isoformat(),
            }

        process = psutil.Process()

        # CPU使用率
        cpu_percent = process.cpu_percent(interval=0.1)

        # 内存使用
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        memory_percent = process.memory_percent()

        # 磁盘使用
        disk_usage = psutil.disk_usage('.')
        disk_usage_mb = disk_usage.used / (1024 * 1024)

        # 系统信息
        cpu_count = psutil.cpu_count()
        total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)

        metrics = PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            disk_usage_mb=disk_usage_mb,
        )

        self.metrics_history.append(metrics)

        return {
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "memory_mb": memory_mb,
            "memory_percent": memory_percent,
            "total_memory_gb": total_memory_gb,
            "disk_usage_mb": disk_usage_mb,
            "disk_free_gb": disk_usage.free / (1024 * 1024 * 1024),
            "timestamp": metrics.timestamp.isoformat(),
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        获取指标汇总统计

        Returns:
            汇总统计字典
        """
        if not self.metrics_history:
            return {}

        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_mb for m in self.metrics_history]

        return {
            "samples": len(self.metrics_history),
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "cpu_max": max(cpu_values),
            "cpu_min": min(cpu_values),
            "memory_avg_mb": sum(memory_values) / len(memory_values),
            "memory_max_mb": max(memory_values),
            "memory_min_mb": min(memory_values),
            "duration_seconds": (
                self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp
            ).total_seconds(),
        }

    def monitor_function(
        self,
        func,
        *args,
        **kwargs
    ) -> tuple:
        """
        监控函数执行

        Args:
            func: 要监控的函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            (函数结果, 性能指标) 元组
        """
        # 收集初始指标
        start_time = time.time()
        start_metrics = self.collect_metrics()

        # 执行函数
        result = func(*args, **kwargs)

        # 收集结束指标
        end_time = time.time()
        end_metrics = self.collect_metrics()

        performance = {
            "duration_seconds": end_time - start_time,
            "cpu_used": end_metrics["cpu_percent"] - start_metrics["cpu_percent"],
            "memory_delta_mb": end_metrics["memory_mb"] - start_metrics["memory_mb"],
        }

        return result, performance

    def check_resource_limits(self, limits: Dict[str, float]) -> Dict[str, bool]:
        """
        检查资源使用是否超限

        Args:
            limits: 限制字典，如 {"cpu_percent": 80, "memory_percent": 90}

        Returns:
            各资源是否超限的字典
        """
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil未安装，无法检查资源限制")
            return {"cpu_ok": True, "memory_ok": True}

        metrics = self.collect_metrics()

        status = {
            "cpu_ok": metrics["cpu_percent"] < limits.get("cpu_percent", 100),
            "memory_ok": metrics["memory_percent"] < limits.get("memory_percent", 100),
        }

        return status

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        记录指标到日志

        Args:
            metrics: 指标字典
        """
        self.logger.info(
            f"性能指标: CPU={metrics['cpu_percent']:.1f}%, "
            f"内存={metrics['memory_mb']:.1f}MB ({metrics['memory_percent']:.1f}%)"
        )

    def get_data_directory_stats(self) -> Dict[str, Any]:
        """
        获取数据目录统计信息

        Returns:
            目录统计字典
        """
        stats = {}

        for name, path in [
            ("data_dir", self.config.data.data_dir),
            ("rps_dir", self.config.data.rps_dir),
            ("models_dir", self.config.data.models_dir),
        ]:
            if os.path.exists(path):
                files = []
                total_size = 0

                for root, dirs, filenames in os.walk(path):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        if os.path.isfile(file_path):
                            files.append(file_path)
                            total_size += os.path.getsize(file_path)

                stats[name] = {
                    "path": path,
                    "exists": True,
                    "file_count": len(files),
                    "total_size_mb": total_size / (1024 * 1024),
                }
            else:
                stats[name] = {
                    "path": path,
                    "exists": False,
                    "file_count": 0,
                    "total_size_mb": 0,
                }

        return stats

    def alert_if_high_usage(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0,
    ) -> List[str]:
        """
        如果资源使用过高则发出警告

        Args:
            cpu_threshold: CPU阈值
            memory_threshold: 内存阈值

        Returns:
            警告消息列表
        """
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil未安装，无法检查资源使用")
            return []

        alerts = []
        metrics = self.collect_metrics()

        if metrics["cpu_percent"] > cpu_threshold:
            alerts.append(
                f"CPU使用率过高: {metrics['cpu_percent']:.1f}% > {cpu_threshold}%"
            )

        if metrics["memory_percent"] > memory_threshold:
            alerts.append(
                f"内存使用率过高: {metrics['memory_percent']:.1f}% > {memory_threshold}%"
            )

        for alert in alerts:
            self.logger.warning(alert)

        return alerts
