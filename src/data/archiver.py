"""
数据归档模块

提供旧数据归档功能
"""

import os
import shutil
import tarfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)


class DataArchiver(AnalyzerBase):
    """
    数据归档器

    归档旧的数据文件和模型文件

    Attributes:
        config: 配置实例
        archive_dir: 归档目录

    Example:
        >>> archiver = DataArchiver()
        >>> archiver.archive_old_data(days_to_keep=90)
    """

    def __init__(
        self,
        archive_dir: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """初始化数据归档器"""
        super().__init__(config)

        if archive_dir is None:
            archive_dir = "archive"

        self.archive_dir = archive_dir
        os.makedirs(archive_dir, exist_ok=True)

    def archive_old_data(
        self,
        days_to_keep: int = 90,
        archive_type: str = "all",
    ) -> Dict[str, int]:
        """
        归档旧数据

        Args:
            days_to_keep: 保留天数
            archive_type: 归档类型 (data, models, all)

        Returns:
            归档统计信息
        """
        stats = {
            "archived_count": 0,
            "archived_size_mb": 0,
            "archive_file": "",
        }

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        if archive_type in ["data", "all"]:
            data_stats = self._archive_data_dir(
                self.config.data.data_dir,
                cutoff_date,
                "data"
            )
            stats["archived_count"] += data_stats["archived_count"]
            stats["archived_size_mb"] += data_stats["archived_size_mb"]

        if archive_type in ["models", "all"]:
            model_stats = self._archive_data_dir(
                self.config.data.models_dir,
                cutoff_date,
                "models"
            )
            stats["archived_count"] += model_stats["archived_count"]
            stats["archived_size_mb"] += model_stats["archived_size_mb"]

        self.logger.info(
            f"归档完成: {stats['archived_count']} 文件, "
            f"{stats['archived_size_mb']:.2f} MB"
        )

        return stats

    def _archive_data_dir(
        self,
        source_dir: str,
        cutoff_date: datetime,
        archive_name: str,
    ) -> Dict[str, int]:
        """归档指定目录"""
        if not os.path.exists(source_dir):
            return {"archived_count": 0, "archived_size_mb": 0}

        files_to_archive = []
        total_size = 0

        for file_name in os.listdir(source_dir):
            file_path = os.path.join(source_dir, file_name)

            if not os.path.isfile(file_path):
                continue

            # 检查文件修改时间
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            if mtime < cutoff_date:
                files_to_archive.append(file_path)
                total_size += os.path.getsize(file_path)

        if not files_to_archive:
            return {"archived_count": 0, "archived_size_mb": 0}

        # 创建归档文件
        timestamp = datetime.now().strftime("%Y%m%d")
        archive_file = os.path.join(
            self.archive_dir,
            f"{archive_name}_archive_{timestamp}.tar.gz"
        )

        with tarfile.open(archive_file, "w:gz") as tar:
            for file_path in files_to_archive:
                tar.add(file_path, arcname=os.path.basename(file_path))
                os.remove(file_path)

        return {
            "archived_count": len(files_to_archive),
            "archived_size_mb": total_size / (1024 * 1024),
            "archive_file": archive_file,
        }

    def archive_training_data(
        self,
        days_to_keep: int = 30,
    ) -> int:
        """
        归档训练数据

        Args:
            days_to_keep: 保留天数

        Returns:
            归档的文件数
        """
        training_dir = "training_data"

        if not os.path.exists(training_dir):
            return 0

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        archived_count = 0

        archive_subdir = os.path.join(self.archive_dir, "training_data")
        os.makedirs(archive_subdir, exist_ok=True)

        for file_name in os.listdir(training_dir):
            file_path = os.path.join(training_dir, file_name)

            if not os.path.isfile(file_path):
                continue

            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            if mtime < cutoff_date:
                # 移动到归档目录
                dest_path = os.path.join(archive_subdir, file_name)
                shutil.move(file_path, dest_path)
                archived_count += 1

        self.logger.info(f"归档训练数据: {archived_count} 文件")

        return archived_count

    def cleanup_old_archives(
        self,
        days_to_keep: int = 365,
    ) -> int:
        """
        清理旧的归档文件

        Args:
            days_to_keep: 归档保留天数

        Returns:
            删除的文件数
        """
        if not os.path.exists(self.archive_dir):
            return 0

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0

        for file_name in os.listdir(self.archive_dir):
            file_path = os.path.join(self.archive_dir, file_name)

            if not os.path.isfile(file_path):
                continue

            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            if mtime < cutoff_date:
                os.remove(file_path)
                deleted_count += 1

        self.logger.info(f"清理旧归档: {deleted_count} 文件")

        return deleted_count

    def get_archive_info(self) -> Dict[str, any]:
        """
        获取归档信息

        Returns:
            归档信息字典
        """
        if not os.path.exists(self.archive_dir):
            return {
                "archive_dir_exists": False,
                "total_archives": 0,
                "total_size_mb": 0,
            }

        total_files = 0
        total_size = 0

        for file_name in os.listdir(self.archive_dir):
            file_path = os.path.join(self.archive_dir, file_name)

            if os.path.isfile(file_path):
                total_files += 1
                total_size += os.path.getsize(file_path)

        return {
            "archive_dir_exists": True,
            "archive_dir": self.archive_dir,
            "total_archives": total_files,
            "total_size_mb": total_size / (1024 * 1024),
        }
