"""
并行处理工具

提供进程池执行和进度跟踪功能

功能特性:
- 进程池并行执行
- 可选的进度条显示(依赖tqdm)
- 任务错误处理
- 多种映射模式支持
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar

from src.core.config import Config, ParallelConfig
from src.core.logger import get_logger

logger = get_logger(__name__)

# ========== 类型变量 ==========

T = TypeVar("T")  # 输入类型
R = TypeVar("R")  # 结果类型


@dataclass
class TaskResult:
    """
    并行任务结果

    表示单个并行任务的执行结果

    Attributes:
        index: 任务在输入序列中的索引
        result: 任务执行结果
        error: 如果发生错误，存储异常对象
        success: 任务是否成功完成

    Example:
        >>> result = TaskResult(
        >>>     index=0,
        >>>     result="processed_data",
        >>>     error=None,
        >>>     success=True
        >>> )
    """
    index: int  # 任务索引
    result: Any  # 执行结果
    error: Optional[Exception] = None  # 错误信息
    success: bool = True  # 是否成功


class ParallelProcessor:
    """
    并行处理器

    提供带进度跟踪的并行处理功能

    使用进程池并行执行函数，支持进度显示
    自动处理任务结果顺序，保证结果与输入顺序一致

    Attributes:
        config: 并行处理配置
        max_workers: 最大工作进程数
        chunk_size: 任务分块大小

    Example:
        >>> from src.core.parallel import ParallelProcessor
        >>>
        >>> # 创建处理器
        >>> processor = ParallelProcessor(max_workers=4)
        >>>
        >>> # 简单映射
        >>> def process_item(item):
        >>>     return item * 2
        >>>
        >>> results = processor.map(process_item, [1, 2, 3, 4])
        >>> print(results)  # [2, 4, 6, 8]
        >>>
        >>> # 带描述的映射
        >>> results = processor.map(
        >>>     process_item,
        >>>     items,
        >>>     desc="处理数据"
        >>> )
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        初始化并行处理器

        Args:
            config: 并行处理配置，None则使用全局配置
        """
        # 获取配置
        if config is None:
            app_config = Config.get_instance()
            config = app_config.parallel

        self.config = config
        # 确定工作进程数
        self.max_workers = config.max_workers or os.cpu_count() or 1
        self.chunk_size = config.chunk_size

    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        desc: Optional[str] = None,
        show_progress: bool = None,
    ) -> List[R]:
        """
        并行应用函数到所有项目

        将函数并行应用到输入序列的每个元素
        保证结果顺序与输入顺序一致

        Args:
            func: 要应用的函数(必须可序列化)
            items: 要处理的项目序列
            desc: 进度显示的描述文字
            show_progress: 是否显示进度条，None则使用配置

        Returns:
            结果列表，顺序与输入项目一致

        Example:
            >>> def square(x):
            >>>     return x ** 2
            >>>
            >>> results = processor.map(
            >>>     square,
            >>>     [1, 2, 3, 4, 5],
            >>>     desc="计算平方"
            >>> )
            >>> # 结果: [1, 4, 9, 16, 25]
        """
        items_list = list(items)
        if not items_list:
            return []

        # 确定是否显示进度
        show_prog = show_progress if show_progress is not None else self.config.enable_progress

        # 小批量使用顺序执行
        if len(items_list) <= 1:
            return [func(item) for item in items_list]

        # 初始化结果列表
        results = [None] * len(items_list)

        try:
            # 尝试导入tqdm用于进度显示
            if show_prog:
                try:
                    from tqdm import tqdm

                    progress = tqdm(total=len(items_list), desc=desc)
                except ImportError:
                    show_prog = False

            # 使用进程池执行
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_index = {}
                for i, item in enumerate(items_list):
                    future = executor.submit(func, item)
                    future_to_index[future] = i

                # 收集完成的结果
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as e:
                        logger.error(f"任务 {index} 失败: {e}")
                        results[index] = None

                    # 更新进度
                    if show_prog:
                        progress.update(1)

            if show_prog:
                progress.close()

        except ImportError:
            # 无tqdm时的回退方案
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_index = {}
                for i, item in enumerate(items_list):
                    future = executor.submit(func, item)
                    future_to_index[future] = i

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as e:
                        logger.error(f"任务 {index} 失败: {e}")
                        results[index] = None

        return results

    def map_with_index(
        self,
        func: Callable[[int, T], R],
        items: Iterable[T],
        desc: Optional[str] = None,
    ) -> List[R]:
        """
        并行应用函数到带索引的项目

        函数接收(index, item)作为参数

        Args:
            func: 接收(index, item)的函数
            items: 要处理的项目序列
            desc: 进度显示描述

        Returns:
            结果列表

        Example:
            >>> def process_with_index(index, item):
            >>>     print(f"处理第 {index} 项")
            >>>     return item * 2
            >>>
            >>> results = processor.map_with_index(
            >>>     process_with_index,
            >>>     [10, 20, 30]
            >>> )
        """
        indexed_items = list(enumerate(items))
        indexed_func = lambda x: func(x[0], x[1])
        return self.map(indexed_func, indexed_items, desc)

    def starmap(
        self,
        func: Callable[..., R],
        args_list: Iterable[Tuple],
        desc: Optional[str] = None,
    ) -> List[R]:
        """
        并行应用函数到参数元组序列

        类似itertools.starmap，将函数应用到参数元组

        Args:
            func: 要应用的函数
            args_list: 参数元组的可迭代对象
            desc: 进度显示描述

        Returns:
            结果列表

        Example:
            >>> def add(a, b, c):
            >>>     return a + b + c
            >>>
            >>> args = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
            >>> results = processor.starmap(add, args)
            >>> # 结果: [6, 15, 24]
        """
        args_func = lambda args: func(*args)
        return self.map(args_func, args_list, desc)


# ========== 便捷函数 ==========

def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: Optional[int] = None,
    desc: Optional[str] = None,
) -> List[R]:
    """
    简单并行映射函数

    快速并行执行的便捷函数

    Args:
        func: 要应用的函数
        items: 要处理的项目
        max_workers: 最大工作进程数
        desc: 进度显示描述

    Returns:
        结果列表

    Example:
        >>> from src.core.parallel import parallel_map
        >>>
        >>> def process(x):
        >>>     return x ** 2
        >>>
        >>> results = parallel_map(
        >>>     process,
        >>>     range(100),
        >>>     max_workers=4,
        >>>     desc="处理数据"
        >>> )
    """
    config = ParallelConfig(max_workers=max_workers)
    processor = ParallelProcessor(config)
    return processor.map(func, items, desc)


def parallel_starmap(
    func: Callable[..., R],
    args_list: Iterable[Tuple],
    max_workers: Optional[int] = None,
    desc: Optional[str] = None,
) -> List[R]:
    """
    简单并行starmap函数

    快速并行执行参数元组的便捷函数

    Args:
        func: 要应用的函数
        args_list: 参数元组的可迭代对象
        max_workers: 最大工作进程数
        desc: 进度显示描述

    Returns:
        结果列表

    Example:
        >>> from src.core.parallel import parallel_starmap
        >>>
        >>> def multiply(a, b):
        >>>     return a * b
        >>>
        >>> args = [(1, 2), (3, 4), (5, 6)]
        >>> results = parallel_starmap(multiply, args)
        >>> # 结果: [2, 12, 30]
    """
    config = ParallelConfig(max_workers=max_workers)
    processor = ParallelProcessor(config)
    return processor.starmap(func, args_list, desc)
