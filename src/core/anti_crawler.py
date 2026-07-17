"""
反爬虫与重试控制（核心基础设施）

本模块原属于 ``src/data/downloaders.py``，现下沉到 ``src/core/`` 作为**共享**基础设施，
使 ``BaostockProvider``（providers 层）与 ``IncrementalDownloader``（downloaders 层）
等所有 baostock 调用方复用**同一份**限速/重试/错误分类实现（DRY）。

对外提供：
    - :class:`AntiCrawlerController`：自适应延时 + 错误分类（含静态 ``classify_error``）。
    - :class:`RequestRetryManager`：基于错误类型的指数退避重试决策。

注意：这两个类原本仅被 ``IncrementalDownloader`` 使用；下沉后其**对外行为保持不变**，
``downloaders.py`` 通过 re-export 继续暴露这两个名字以兼容既有导入
（``src/data/__init__.py``、``tests/test_downloaders.py``）。
"""

import random
import threading
import time
from collections import deque
from typing import Any, Callable, Optional, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)


class AntiCrawlerController:
    """
    反爬虫控制器

    实现智能的请求频率控制和延时策略，避免被baostock API限制

    Attributes:
        base_delay: 基础延时时间(秒)
        max_delay: 最大延时时间(秒)
        requests_per_minute: 每分钟最大请求数
        consecutive_failures: 连续失败次数

    Example:
        >>> controller = AntiCrawlerController()
        >>> controller.before_request()  # 执行延时
        >>> # 执行请求...
        >>> controller.after_request(success=True)
    """

    def __init__(
        self,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        requests_per_minute: int = 45,
    ):
        """
        初始化反爬虫控制器

        Args:
            base_delay: 基础延时时间(秒)
            max_delay: 最大延时时间(秒)
            requests_per_minute: 每分钟最大请求数
        """
        self.request_times = deque(maxlen=100)  # 记录最近100次请求时间
        self.failed_requests = deque(maxlen=50)  # 记录最近50次失败请求
        self.lock = threading.Lock()
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.requests_per_minute = requests_per_minute
        self.consecutive_failures = 0  # 连续失败次数
        self.parameter_errors = 0  # 参数错误次数
        self.network_errors = 0  # 网络错误次数

    def calculate_delay(self) -> float:
        """
        根据请求历史和失败情况计算智能延时

        Returns:
            延时时间(秒)
        """
        with self.lock:
            current_time = time.time()

            # 清理过期的请求记录(超过1分钟)
            while self.request_times and current_time - self.request_times[0] > 60:
                self.request_times.popleft()

            # 基础延时
            delay = self.base_delay

            # 根据请求频率调整延时
            if len(self.request_times) >= self.requests_per_minute:
                delay *= 2  # 请求过于频繁，延时翻倍

            # 根据连续失败次数调整延时
            if self.consecutive_failures > 0:
                delay *= (1 + self.consecutive_failures * 0.5)

            # 根据最近失败率调整延时
            recent_failures = sum(
                1
                for t in self.failed_requests
                if current_time - t < 300
            )  # 5分钟内的失败
            if recent_failures > 5:
                delay *= (1 + recent_failures * 0.2)

            # 添加随机因子，避免规律性
            delay *= random.uniform(0.8, 1.5)

            # 限制最大延时
            delay = min(delay, self.max_delay)

            return delay

    def before_request(self) -> None:
        """
        请求前的处理：记录请求时间并执行延时

        在每次API请求前调用此方法
        """
        delay = self.calculate_delay()

        with self.lock:
            self.request_times.append(time.time())

        if delay > 1.0:
            logger.info(f"反爬虫延时: {delay:.2f}秒")

        time.sleep(delay)

    def after_request(self, success: bool, error_type: str = "unknown") -> None:
        """
        请求后的处理：记录成功/失败状态和错误类型

        Args:
            success: 请求是否成功
            error_type: 错误类型 ('parameter', 'network', 'data', 'unknown')
        """
        with self.lock:
            if success:
                self.consecutive_failures = 0
            else:
                # 只有网络错误和未知错误才计入连续失败
                if error_type in ["network", "unknown"]:
                    self.consecutive_failures += 1
                    self.failed_requests.append(time.time())
                    logger.warning(
                        f"请求失败，连续失败次数: {self.consecutive_failures}，错误类型: {error_type}"
                    )
                elif error_type == "parameter":
                    self.parameter_errors += 1
                    logger.warning(f"参数错误，不计入连续失败，参数错误总数: {self.parameter_errors}")
                elif error_type == "data":
                    logger.info(f"数据相关问题，错误类型: {error_type}")
                else:
                    self.consecutive_failures += 1
                    self.failed_requests.append(time.time())
                    logger.warning(f"未知错误，连续失败次数: {self.consecutive_failures}")

    def should_pause(self) -> bool:
        """
        判断是否需要暂停请求(连续失败过多时)

        Returns:
            是否需要暂停
        """
        return self.consecutive_failures >= 5

    def pause_and_recover(self) -> None:
        """
        暂停并恢复策略

        当连续失败过多时调用，暂停一段时间后重试
        """
        if self.should_pause():
            pause_time = min(
                30 + self.consecutive_failures * 10, 300
            )  # 最多暂停5分钟
            logger.warning(f"连续失败过多，暂停 {pause_time} 秒后重试")
            time.sleep(pause_time)
            with self.lock:
                self.consecutive_failures = max(0, self.consecutive_failures - 2)

    @staticmethod
    def classify_error(error_code: str, error_msg: str) -> str:
        """
        分类错误类型

        Args:
            error_code: 错误代码
            error_msg: 错误消息

        Returns:
            错误类型: 'parameter', 'network', 'data', 'unknown'
        """
        if not error_msg:
            return "unknown"

        error_msg_lower = error_msg.lower()

        # 参数错误
        parameter_keywords = [
            "起始日期大于终止日期",
            "股票代码不存在",
            "股票代码错误",
            "日期格式错误",
            "参数错误",
            "invalid parameter",
            "invalid date",
            "invalid code",
            "start date is greater than end date",
            "stock code does not exist",
        ]

        for keyword in parameter_keywords:
            if keyword in error_msg_lower:
                return "parameter"

        # 网络错误
        network_keywords = [
            "网络超时",
            "连接被拒绝",
            "连接超时",
            "网络连接失败",
            "network timeout",
            "connection refused",
            "connection timeout",
            "network error",
            "timeout",
            "connection failed",
        ]

        for keyword in network_keywords:
            if keyword in error_msg_lower:
                return "network"

        # 数据相关问题(不是真正的错误)
        data_keywords = [
            "没有数据",
            "数据为空",
            "no data",
            "empty data",
            "停牌",
            "suspended",
        ]

        for keyword in data_keywords:
            if keyword in error_msg_lower:
                return "data"

        return "unknown"


class RequestRetryManager:
    """
    请求重试管理器

    根据错误类型决定是否重试，使用指数退避算法

    Attributes:
        max_retries: 最大重试次数

    Example:
        >>> manager = RequestRetryManager(max_retries=3)
        >>> result, error_type = manager.retry_request(
        >>>     lambda: api_call(),
        >>>     anti_crawler_controller
        >>> )
    """

    def __init__(self, max_retries: int = 3):
        """
        初始化重试管理器

        Args:
            max_retries: 最大重试次数
        """
        self.max_retries = max_retries

    def should_retry(self, error_type: str, attempt: int) -> bool:
        """
        根据错误类型和尝试次数决定是否重试

        Args:
            error_type: 错误类型
            attempt: 当前尝试次数

        Returns:
            是否应该重试
        """
        if attempt >= self.max_retries:
            return False

        # 参数错误不重试
        if error_type == "parameter":
            return False

        # 网络错误和未知错误可以重试
        if error_type in ["network", "unknown"]:
            return True

        # 数据相关问题重试一次
        if error_type == "data" and attempt == 0:
            return True

        return False

    def retry_request(
        self,
        func: Callable,
        anti_crawler_controller: AntiCrawlerController,
        *args,
        **kwargs,
    ) -> Tuple[Optional[Any], str]:
        """
        带重试的请求执行

        Args:
            func: 要执行的函数
            anti_crawler_controller: 反爬虫控制器
            *args, **kwargs: 函数参数

        Returns:
            (结果, 错误类型) 元组
        """
        last_exception = None
        last_error_type = "unknown"

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result, "success"
                else:
                    # 函数返回None，可能是数据问题
                    last_error_type = "data"
                    if not self.should_retry(last_error_type, attempt):
                        logger.info("数据相关问题，不再重试")
                        break

            except Exception as e:
                last_exception = e
                # 尝试从异常信息中分类错误
                error_msg = str(e)
                last_error_type = anti_crawler_controller.classify_error("", error_msg)

                if not self.should_retry(last_error_type, attempt):
                    if last_error_type == "parameter":
                        logger.error(f"参数错误，不进行重试: {error_msg}")
                    else:
                        logger.error(f"错误类型 {last_error_type}，不再重试: {error_msg}")
                    break

                if attempt < self.max_retries:
                    wait_time = (2**attempt) + random.uniform(0, 1)  # 指数退避
                    logger.warning(
                        f"请求失败，第 {attempt + 1} 次重试，等待 {wait_time:.2f} 秒，错误类型: {last_error_type}, 错误: {error_msg}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"请求最终失败，已重试 {self.max_retries} 次，错误类型: {last_error_type}, 错误: {error_msg}"
                    )

        return None, last_error_type
