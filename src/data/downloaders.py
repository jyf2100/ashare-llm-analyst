"""
数据下载器模块

提供增量股票数据下载功能，包含反爬虫控制和请求重试机制
"""

import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.core.anti_crawler import AntiCrawlerController, RequestRetryManager
from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.exceptions import DataFetchError
from src.core.logger import get_logger

logger = get_logger(__name__)


class IncrementalDownloader(AnalyzerBase):
    """
    增量股票数据下载器

    支持往前补数据和往后补数据的增量更新功能

    Attributes:
        output_dir: 输出目录路径
        csv_provider: CSV数据提供者
        baostock_provider: baostock数据提供者
        anti_crawler: 反爬虫控制器
        retry_manager: 重试管理器

    Example:
        >>> downloader = IncrementalDownloader()
        >>> # 往前补30天数据
        >>> result = downloader.incremental_update(backfill_days=30)
        >>> # 往后补到指定日期
        >>> result = downloader.incremental_update(forward_fill_date="2024-12-31")
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        初始化增量下载器

        Args:
            output_dir: 输出目录路径，None则使用配置中的data_dir
            config: 配置实例
        """
        super().__init__(config)

        # 确定输出目录
        if output_dir is None:
            output_dir = self.config.data.data_dir

        self.output_dir = output_dir

        # 创建输出目录
        import os

        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化数据提供者
        from src.data.providers import BaostockProvider, CSVProvider

        self.csv_provider = CSVProvider(self.output_dir, self.config)
        self.baostock_provider = BaostockProvider(self.config)

        # 初始化反爬虫和重试控制器
        self.anti_crawler = AntiCrawlerController()
        self.retry_manager = RequestRetryManager(max_retries=3)

        # 请求统计
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

    def _format_stock_code(self, code: str) -> str:
        """格式化股票代码"""
        if "." in code:
            return code

        if code.startswith("sz") or code.startswith("sh"):
            market = code[:2]
            stock_num = code[2:]
            return f"{market}.{stock_num}"
        else:
            if len(code) == 6 and code.isdigit():
                if code.startswith("0") or code.startswith("3"):
                    return f"sz.{code}"
                elif code.startswith("6"):
                    return f"sh.{code}"

        return code

    def get_existing_stocks(self) -> List[str]:
        """
        获取已存在的股票数据文件列表

        Returns:
            股票代码列表
        """
        return self.csv_provider.list_available()

    def get_stock_date_range(self, stock_code: str) -> Tuple[Optional[str], Optional[str]]:
        """
        获取股票数据的日期范围

        Args:
            stock_code: 股票代码

        Returns:
            (最小日期, 最大日期) 元组，None表示无数据
        """
        formatted_code = self._format_stock_code(stock_code)

        if not self.csv_provider.exists(formatted_code):
            return None, None

        try:
            df = self.csv_provider.load(formatted_code)
            if "date" not in df.columns or len(df) == 0:
                return None, None

            min_date = df["date"].min().strftime("%Y-%m-%d")
            max_date = df["date"].max().strftime("%Y-%m-%d")

            return min_date, max_date

        except Exception as e:
            self.logger.error(f"读取股票 {stock_code} 数据失败: {e}")
            return None, None

    def download_stock_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        下载指定日期范围的股票数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            股票数据DataFrame，失败返回None
        """
        self.anti_crawler.before_request()
        self.total_requests += 1

        def _download():
            return self.baostock_provider.load(
                stock_code,
                start_date=start_date,
                end_date=end_date,
            )

        try:
            result, error_type = self.retry_manager.retry_request(
                _download, self.anti_crawler
            )
            if result is not None:
                self.anti_crawler.after_request(True)
                self.successful_requests += 1
                return result
            else:
                self.anti_crawler.after_request(False, error_type)
                self.failed_requests += 1
                return None

        except Exception as e:
            error_type = self.anti_crawler.classify_error("", str(e))
            self.anti_crawler.after_request(False, error_type)
            self.failed_requests += 1
            return None

    def merge_and_save_data(
        self,
        stock_code: str,
        new_data: pd.DataFrame,
    ) -> bool:
        """
        合并新数据与现有数据并保存

        Args:
            stock_code: 股票代码
            new_data: 新下载的数据

        Returns:
            是否成功
        """
        formatted_code = self._format_stock_code(stock_code)

        try:
            # 读取现有数据
            if self.csv_provider.exists(formatted_code):
                existing_data = self.csv_provider.load(formatted_code)

                # 合并数据
                combined_data = pd.concat(
                    [existing_data, new_data], ignore_index=True
                )

                # 去重并排序
                combined_data = combined_data.drop_duplicates(
                    subset=["date"], keep="last"
                )
                combined_data = combined_data.sort_values("date")

                self.logger.info(
                    f"股票 {stock_code}: 原有 {len(existing_data)} 条，"
                    f"新增 {len(new_data)} 条，合并后 {len(combined_data)} 条"
                )
            else:
                combined_data = new_data.sort_values("date")
                self.logger.info(
                    f"股票 {stock_code}: 新建文件，共 {len(combined_data)} 条"
                )

            # 保存数据
            return self.csv_provider.save(formatted_code, combined_data)

        except Exception as e:
            self.logger.error(f"合并保存股票 {stock_code} 数据失败: {e}")
            return False

    def backfill_data(self, stock_code: str, days: int) -> bool:
        """
        往前补充股票数据

        Args:
            stock_code: 股票代码
            days: 往前补充的天数

        Returns:
            是否成功
        """
        min_date, max_date = self.get_stock_date_range(stock_code)

        if min_date is None:
            self.logger.warning(f"股票 {stock_code} 没有现有数据，跳过往前补")
            return False

        # 计算需要补充的开始日期
        min_date_obj = datetime.strptime(min_date, "%Y-%m-%d")
        start_date = (min_date_obj - timedelta(days=days + 50)).strftime("%Y-%m-%d")
        end_date = (min_date_obj - timedelta(days=1)).strftime("%Y-%m-%d")

        self.logger.info(
            f"股票 {stock_code} 往前补数据: {start_date} 到 {end_date} (目标 {days} 个交易日)"
        )

        # 下载数据
        new_data = self.download_stock_data(stock_code, start_date, end_date)
        if new_data is None or len(new_data) == 0:
            self.logger.warning(f"股票 {stock_code} 往前补数据失败：没有获取到数据")
            return False

        # 取最新的指定天数
        new_data = new_data.tail(days)

        # 合并并保存
        return self.merge_and_save_data(stock_code, new_data)

    def forward_fill_data(self, stock_code: str, target_date: str) -> bool:
        """
        往后补充股票数据到指定日期

        Args:
            stock_code: 股票代码
            target_date: 目标日期

        Returns:
            是否成功
        """
        _, max_date = self.get_stock_date_range(stock_code)

        if max_date is None:
            self.logger.warning(f"股票 {stock_code} 没有现有数据，跳过往后补")
            return False

        # 检查是否需要补充
        max_date_obj = datetime.strptime(max_date, "%Y-%m-%d")
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")

        if target_date_obj <= max_date_obj:
            self.logger.info(
                f"股票 {stock_code} 现有数据已覆盖到 {max_date}，无需补充到 {target_date}"
            )
            return True

        # 计算需要补充的日期范围
        start_date = (max_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = target_date

        self.logger.info(f"股票 {stock_code} 往后补数据: {start_date} 到 {end_date}")

        # 下载数据
        new_data = self.download_stock_data(stock_code, start_date, end_date)
        if new_data is None or len(new_data) == 0:
            self.logger.warning(f"股票 {stock_code} 往后补数据失败：没有获取到数据")
            return False

        # 合并并保存
        return self.merge_and_save_data(stock_code, new_data)

    def incremental_update(
        self,
        backfill_days: Optional[int] = None,
        forward_fill_date: Optional[str] = None,
        stock_codes: Optional[List[str]] = None,
        max_stocks: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        增量更新股票数据

        Args:
            backfill_days: 往前补充的天数
            forward_fill_date: 往后补充到的日期
            stock_codes: 指定的股票代码列表，None则处理所有现有股票
            max_stocks: 最大处理股票数量

        Returns:
            更新结果字典
        """
        # 重置统计
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # 确定要处理的股票列表
        if stock_codes:
            target_stocks = stock_codes
        else:
            target_stocks = self.get_existing_stocks()

        if max_stocks:
            target_stocks = target_stocks[:max_stocks]

        if not target_stocks:
            self.logger.error("没有找到需要处理的股票")
            return {"success": 0, "failed": 0, "total": 0}

        self.logger.info(f"找到 {len(target_stocks)} 只股票需要增量更新")

        success_count = 0
        failed_stocks = []

        for i, stock_code in enumerate(target_stocks, 1):
            self.logger.info(f"进度: {i}/{len(target_stocks)} - 处理股票 {stock_code}")

            # 检查是否需要暂停
            if self.anti_crawler.should_pause():
                self.anti_crawler.pause_and_recover()

            stock_success = True

            # 往前补数据
            if backfill_days and backfill_days > 0:
                if not self.backfill_data(stock_code, backfill_days):
                    self.logger.warning(f"股票 {stock_code} 往前补数据失败")
                    stock_success = False

            # 往后补数据
            if forward_fill_date:
                if not self.forward_fill_data(stock_code, forward_fill_date):
                    self.logger.warning(f"股票 {stock_code} 往后补数据失败")
                    stock_success = False

            if stock_success:
                success_count += 1
            else:
                failed_stocks.append(stock_code)

            # 反爬虫延时
            if i % 10 == 0:
                if not self.anti_crawler.should_pause():
                    delay = random.uniform(1.0, 2.0)
                    self.logger.info(f"已处理{i}只股票，休息{delay:.1f}秒...")
                    time.sleep(delay)
            else:
                delay = random.uniform(0.3, 0.8)
                time.sleep(delay)

        # 计算成功率
        success_rate = (
            (self.successful_requests / self.total_requests * 100)
            if self.total_requests > 0
            else 0
        )

        self.logger.info(f"\n=== 增量更新完成 ===")
        self.logger.info(f"总请求数: {self.total_requests}")
        self.logger.info(f"成功: {success_count}")
        self.logger.info(f"失败: {len(failed_stocks)}")
        self.logger.info(f"API成功率: {success_rate:.2f}%")

        # 关闭baostock连接
        self.baostock_provider.close()

        return {
            "success": success_count,
            "failed": len(failed_stocks),
            "failed_stocks": failed_stocks,
            "total": len(target_stocks),
            "anti_crawler_stats": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
            },
        }

    def download_latest(
        self,
        stock_codes: Optional[List[str]] = None,
        max_stocks: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        下载最新交易日数据

        默认行为：只下载前一个交易日（最新一天）的数据

        Args:
            stock_codes: 股票代码列表，None则处理所有现有股票
            max_stocks: 最大处理股票数量

        Returns:
            下载结果字典
        """
        # 重置统计
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # 确定要处理的股票列表
        if stock_codes:
            target_stocks = stock_codes
        else:
            target_stocks = self.get_existing_stocks()

        if max_stocks:
            target_stocks = target_stocks[:max_stocks]

        if not target_stocks:
            self.logger.error("没有找到需要处理的股票")
            return {"success": 0, "failed": 0, "total": 0}

        self.logger.info(f"开始下载 {len(target_stocks)} 只股票的最新数据")

        success_count = 0
        failed_stocks = []

        # 计算日期范围：只下载最近 5 天的数据（确保包含最新交易日）
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date_obj = datetime.now() - timedelta(days=7)  # 7天确保包含最新交易日
        start_date = start_date_obj.strftime("%Y-%m-%d")

        for i, stock_code in enumerate(target_stocks, 1):
            self.logger.info(f"进度: {i}/{len(target_stocks)} - 下载股票 {stock_code}")

            # 检查是否需要暂停
            if self.anti_crawler.should_pause():
                self.anti_crawler.pause_and_recover()

            try:
                # 获取现有数据的最新日期
                existing_data = self.csv_provider.load(stock_code)
                if existing_data is not None and len(existing_data) > 0:
                    latest_date = existing_data["date"].max()
                    self.logger.debug(f"股票 {stock_code} 现有数据最新日期: {latest_date}")

                # 下载最近的数据
                new_data = self.download_stock_data(stock_code, start_date, end_date)

                if new_data is None or len(new_data) == 0:
                    self.logger.warning(f"股票 {stock_code} 没有获取到新数据")
                    failed_stocks.append(stock_code)
                    continue

                # 如果有现有数据，合并并去重
                if existing_data is not None and len(existing_data) > 0:
                    # 合并数据
                    merged_data = pd.concat([existing_data, new_data], ignore_index=True)
                    # 去除重复（按日期和股票代码）
                    merged_data = merged_data.drop_duplicates(subset=["date", "code"], keep="last")
                    # 按日期排序
                    merged_data = merged_data.sort_values("date").reset_index(drop=True)

                    new_rows = len(merged_data) - len(existing_data)
                    self.logger.info(f"股票 {stock_code}: 原有 {len(existing_data)} 条，新增 {new_rows} 条，合并后 {len(merged_data)} 条")

                    # 保存合并后的数据
                    self.csv_provider.save(stock_code, merged_data)
                else:
                    # 没有现有数据，直接保存
                    self.csv_provider.save(stock_code, new_data)
                    self.logger.info(f"股票 {stock_code} 下载成功: {len(new_data)} 行")

                success_count += 1

            except Exception as e:
                self.logger.error(f"股票 {stock_code} 处理失败: {e}")
                failed_stocks.append(stock_code)

            # 反爬虫延时
            if i % 10 == 0:
                if not self.anti_crawler.should_pause():
                    delay = random.uniform(1.0, 2.0)
                    self.logger.info(f"已处理{i}只股票，休息{delay:.1f}秒...")
                    time.sleep(delay)
            else:
                delay = random.uniform(0.3, 0.8)
                time.sleep(delay)

        # 计算成功率
        success_rate = (
            (self.successful_requests / self.total_requests * 100)
            if self.total_requests > 0
            else 0
        )

        self.logger.info(f"\n=== 最新数据下载完成 ===")
        self.logger.info(f"总请求数: {self.total_requests}")
        self.logger.info(f"成功: {success_count}")
        self.logger.info(f"失败: {len(failed_stocks)}")
        self.logger.info(f"API成功率: {success_rate:.2f}%")

        # 关闭baostock连接
        self.baostock_provider.close()

        return {
            "success": success_count,
            "failed": len(failed_stocks),
            "failed_stocks": failed_stocks,
            "total": len(target_stocks),
            "anti_crawler_stats": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
            },
        }

    def initial_download(
        self,
        stock_codes: List[str],
        days: int = 300,
        max_stocks: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        初始下载股票数据

        适用于首次下载，当数据目录为空时使用

        Args:
            stock_codes: 股票代码列表
            days: 下载数据的天数 (默认300天)
            max_stocks: 最大处理股票数量

        Returns:
            下载结果字典
        """
        # 重置统计
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # 限制股票数量
        if max_stocks:
            stock_codes = stock_codes[:max_stocks]

        if not stock_codes:
            self.logger.error("没有提供股票代码")
            return {"success": 0, "failed": 0, "total": 0}

        self.logger.info(f"开始初始下载 {len(stock_codes)} 只股票，各 {days} 天数据")

        success_count = 0
        failed_stocks = []

        # 计算日期范围
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date_obj = datetime.now() - timedelta(days=days + 100)  # 多加100天以保证足够的交易日
        start_date = start_date_obj.strftime("%Y-%m-%d")

        for i, stock_code in enumerate(stock_codes, 1):
            self.logger.info(f"进度: {i}/{len(stock_codes)} - 下载股票 {stock_code}")

            # 检查是否需要暂停
            if self.anti_crawler.should_pause():
                self.anti_crawler.pause_and_recover()

            # 下载数据
            data = self.download_stock_data(stock_code, start_date, end_date)

            if data is not None and len(data) > 0:
                # 取最新的指定天数
                data = data.tail(days).copy()

                # 保存数据
                self.csv_provider.save(stock_code, data)

                self.logger.info(f"股票 {stock_code} 下载成功: {len(data)} 行")
                success_count += 1
            else:
                self.logger.warning(f"股票 {stock_code} 下载失败")
                failed_stocks.append(stock_code)

            # 反爬虫延时
            if i % 10 == 0:
                if not self.anti_crawler.should_pause():
                    delay = random.uniform(1.0, 2.0)
                    self.logger.info(f"已处理{i}只股票，休息{delay:.1f}秒...")
                    time.sleep(delay)
            else:
                delay = random.uniform(0.3, 0.8)
                time.sleep(delay)

        # 计算成功率
        success_rate = (
            (self.successful_requests / self.total_requests * 100)
            if self.total_requests > 0
            else 0
        )

        self.logger.info(f"\n=== 初始下载完成 ===")
        self.logger.info(f"总请求数: {self.total_requests}")
        self.logger.info(f"成功: {success_count}")
        self.logger.info(f"失败: {len(failed_stocks)}")
        self.logger.info(f"API成功率: {success_rate:.2f}%")

        # 关闭baostock连接
        self.baostock_provider.close()

        return {
            "success": success_count,
            "failed": len(failed_stocks),
            "failed_stocks": failed_stocks,
            "total": len(stock_codes),
            "anti_crawler_stats": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
            },
        }
