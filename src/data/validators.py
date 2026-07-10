"""
数据验证模块

提供数据质量检查和验证功能
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """
    数据验证结果

    Attributes:
        passed: 是否通过验证
        total_files: 总文件数
        valid_files: 有效文件数
        invalid_files: 无效文件数
        issues: 问题列表
        warnings: 警告列表
        stats: 统计信息
    """
    passed: bool = False
    total_files: int = 0
    valid_files: int = 0
    invalid_files: int = 0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class DataValidator(AnalyzerBase):
    """
    数据验证器

    检查数据目录中所有CSV文件的数据质量

    Attributes:
        config: 配置实例
        required_columns: 必需列名
        min_rows: 最小行数
        max_null_ratio: 最大空值比例

    Example:
        >>> validator = DataValidator()
        >>> result = validator.validate_directory("data/")
        >>> print(f"通过: {result.passed}, 有效文件: {result.valid_files}")
    """

    REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

    def __init__(
        self,
        config: Optional[Config] = None,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 100,
        max_null_ratio: float = 0.1,
    ):
        """
        初始化数据验证器

        Args:
            config: 配置实例
            required_columns: 必需列名列表
            min_rows: 最小行数
            max_null_ratio: 最大空值比例
        """
        super().__init__(config)

        self.required_columns = required_columns or self.REQUIRED_COLUMNS.copy()
        self.min_rows = min_rows
        self.max_null_ratio = max_null_ratio

    def validate_directory(
        self,
        data_dir: str,
        strict_mode: bool = False,
    ) -> ValidationResult:
        """
        验证数据目录中的所有文件

        Args:
            data_dir: 数据目录路径
            strict_mode: 严格模式，任何问题都会导致验证失败

        Returns:
            验证结果
        """
        self.logger.info(f"开始验证数据目录: {data_dir}")

        result = ValidationResult()

        if not os.path.exists(data_dir):
            result.issues.append(f"数据目录不存在: {data_dir}")
            return result

        # 获取所有CSV文件
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        result.total_files = len(csv_files)

        if result.total_files == 0:
            result.issues.append("数据目录中没有CSV文件")
            return result

        self.logger.info(f"发现 {result.total_files} 个CSV文件")

        # 验证每个文件
        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)
            file_result = self.validate_file(file_path)

            if file_result["valid"]:
                result.valid_files += 1
            else:
                result.invalid_files += 1
                result.issues.append(f"{csv_file}: {file_result.get('error', '未知错误')}")

            # 添加警告
            for warning in file_result.get("warnings", []):
                result.warnings.append(f"{csv_file}: {warning}")

        # 统计信息
        result.stats = self._collect_statistics(data_dir)

        # 判断是否通过
        if strict_mode:
            result.passed = result.invalid_files == 0 and len(result.issues) == 0
        else:
            result.passed = result.valid_files > 0

        self.logger.info(
            f"验证完成: {result.valid_files}/{result.total_files} 有效, "
            f"{result.invalid_files} 无效, {len(result.warnings)} 警告"
        )

        return result

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        验证单个文件

        Args:
            file_path: 文件路径

        Returns:
            验证结果字典
        """
        result = {"valid": True, "warnings": []}

        try:
            # 读取文件
            df = pd.read_csv(file_path)

            # 检查行数
            if len(df) < self.min_rows:
                result["valid"] = False
                result["error"] = f"行数不足: {len(df)} < {self.min_rows}"
                return result

            # 检查必需列
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                result["valid"] = False
                result["error"] = f"缺少列: {missing_cols}"
                return result

            # 检查空值比例
            for col in self.required_columns:
                null_ratio = df[col].isnull().sum() / len(df)
                if null_ratio > self.max_null_ratio:
                    result["warnings"].append(f"{col} 空值比例高: {null_ratio:.1%}")

            # 检查价格合理性
            if "close" in df.columns:
                if (df["close"] <= 0).any():
                    result["warnings"].append("存在非正价格")
                if (df["close"] > 10000).any():
                    result["warnings"].append("存在异常高价")

            # 检查价格顺序（high >= low, high >= close等）
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                invalid_high = (df["high"] < df[["open", "low", "close"]].max(axis=1)).sum()
                invalid_low = (df["low"] > df[["open", "high", "close"]].min(axis=1)).sum()

                if invalid_high > 0:
                    result["warnings"].append(f"{invalid_high} 行高价异常")
                if invalid_low > 0:
                    result["warnings"].append(f"{invalid_low} 行低价异常")

        except Exception as e:
            result["valid"] = False
            result["error"] = str(e)

        return result

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        验证DataFrame

        Args:
            df: 要验证的DataFrame

        Returns:
            验证结果
        """
        result = ValidationResult(total_files=1)

        # 检查列
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            result.issues.append(f"缺少列: {missing_cols}")
            return result

        # 检查行数
        if len(df) < self.min_rows:
            result.issues.append(f"行数不足: {len(df)} < {self.min_rows}")
            return result

        # 检查空值
        for col in self.required_columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_ratio = null_count / len(df)
                if null_ratio > self.max_null_ratio:
                    result.warnings.append(f"{col} 空值比例高: {null_ratio:.1%}")

        result.valid_files = 1
        result.passed = True

        return result

    def _collect_statistics(self, data_dir: str) -> Dict[str, Any]:
        """
        收集数据目录统计信息

        Args:
            data_dir: 数据目录路径

        Returns:
            统计信息字典
        """
        stats = {
            "total_files": 0,
            "total_size_mb": 0,
            "date_range": None,
            "stock_codes": [],
        }

        try:
            csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            stats["total_files"] = len(csv_files)
            stats["stock_codes"] = [f.replace(".csv", "") for f in csv_files]

            # 计算总大小
            for csv_file in csv_files:
                file_path = os.path.join(data_dir, csv_file)
                stats["total_size_mb"] += os.path.getsize(file_path) / (1024 * 1024)

            # 获取日期范围
            all_dates = []
            for csv_file in csv_files[:10]:  # 只检查前10个文件
                file_path = os.path.join(data_dir, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    if "date" in df.columns:
                        all_dates.extend(pd.to_datetime(df["date"]).tolist())
                except:
                    pass

            if all_dates:
                stats["date_range"] = {
                    "start": min(all_dates).strftime("%Y-%m-%d"),
                    "end": max(all_dates).strftime("%Y-%m-%d"),
                }

        except Exception as e:
            self.logger.warning(f"收集统计信息时出错: {e}")

        return stats
