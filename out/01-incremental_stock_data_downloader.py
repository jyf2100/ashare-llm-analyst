#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量股票数据下载器
支持往前补n天数据和往后补到指定日期的数据
基于 baostock API 实现增量数据更新
"""

import baostock as bs
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import time
import random
import re
import glob
import threading
from collections import deque
import hashlib
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AntiCrawlerController:
    """
    反爬虫控制器，实现智能的请求频率控制和延时策略
    """
    
    def __init__(self):
        self.request_times = deque(maxlen=100)  # 记录最近100次请求时间
        self.failed_requests = deque(maxlen=50)  # 记录最近50次失败请求
        self.lock = threading.Lock()
        self.base_delay = 0.5  # 基础延时
        self.max_delay = 10.0  # 最大延时
        self.requests_per_minute = 45  # 每分钟最大请求数
        self.consecutive_failures = 0  # 连续失败次数
        self.parameter_errors = 0  # 参数错误次数
        self.network_errors = 0  # 网络错误次数
        
    def calculate_delay(self) -> float:
        """
        根据请求历史和失败情况计算智能延时
        """
        with self.lock:
            current_time = time.time()
            
            # 清理过期的请求记录（超过1分钟）
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
            recent_failures = sum(1 for t in self.failed_requests 
                                if current_time - t < 300)  # 5分钟内的失败
            if recent_failures > 5:
                delay *= (1 + recent_failures * 0.2)
            
            # 添加随机因子，避免规律性
            delay *= random.uniform(0.8, 1.5)
            
            # 限制最大延时
            delay = min(delay, self.max_delay)
            
            return delay
    
    def before_request(self):
        """
        请求前的处理：记录请求时间并执行延时
        """
        delay = self.calculate_delay()
        
        with self.lock:
            self.request_times.append(time.time())
        
        if delay > 1.0:
            logger.info(f"反爬虫延时: {delay:.2f}秒")
        
        time.sleep(delay)
    
    def after_request(self, success: bool, error_type: str = 'unknown'):
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
                if error_type in ['network', 'unknown']:
                    self.consecutive_failures += 1
                    self.failed_requests.append(time.time())
                    logger.warning(f"请求失败，连续失败次数: {self.consecutive_failures}，错误类型: {error_type}")
                elif error_type == 'parameter':
                    self.parameter_errors += 1
                    logger.warning(f"参数错误，不计入连续失败，参数错误总数: {self.parameter_errors}")
                elif error_type == 'data':
                    logger.info(f"数据相关问题，错误类型: {error_type}")
                else:
                    self.consecutive_failures += 1
                    self.failed_requests.append(time.time())
                    logger.warning(f"未知错误，连续失败次数: {self.consecutive_failures}")
    
    def should_pause(self) -> bool:
        """
        判断是否需要暂停请求（连续失败过多时）
        """
        return self.consecutive_failures >= 5
    
    def pause_and_recover(self):
        """
        暂停并恢复策略
        """
        if self.should_pause():
            pause_time = min(30 + self.consecutive_failures * 10, 300)  # 最多暂停5分钟
            logger.warning(f"连续失败过多，暂停 {pause_time} 秒后重试")
            time.sleep(pause_time)
            with self.lock:
                self.consecutive_failures = max(0, self.consecutive_failures - 2)
    
    def classify_error(self, error_code: str, error_msg: str) -> str:
        """
        分类错误类型
        
        Args:
            error_code: 错误代码
            error_msg: 错误消息
            
        Returns:
            错误类型: 'parameter', 'network', 'data', 'unknown'
        """
        if not error_msg:
            return 'unknown'
            
        error_msg_lower = error_msg.lower()
        
        # 参数错误
        parameter_keywords = [
            '起始日期大于终止日期',
            '股票代码不存在',
            '股票代码错误',
            '日期格式错误',
            '参数错误',
            'invalid parameter',
            'invalid date',
            'invalid code',
            'start date is greater than end date',
            'stock code does not exist'
        ]
        
        for keyword in parameter_keywords:
            if keyword in error_msg_lower:
                return 'parameter'
        
        # 网络错误
        network_keywords = [
            '网络超时',
            '连接被拒绝',
            '连接超时',
            '网络连接失败',
            'network timeout',
            'connection refused',
            'connection timeout',
            'network error',
            'timeout',
            'connection failed'
        ]
        
        for keyword in network_keywords:
            if keyword in error_msg_lower:
                return 'network'
        
        # 数据相关问题（不是真正的错误）
        data_keywords = [
            '没有数据',
            '数据为空',
            'no data',
            'empty data',
            '停牌',
            'suspended'
        ]
        
        for keyword in data_keywords:
            if keyword in error_msg_lower:
                return 'data'
        
        return 'unknown'

class RequestRetryManager:
    """
    请求重试管理器
    """
    
    def __init__(self, max_retries: int = 3):
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
        if error_type == 'parameter':
            return False
            
        # 网络错误和未知错误可以重试
        if error_type in ['network', 'unknown']:
            return True
            
        # 数据相关问题重试一次
        if error_type == 'data' and attempt == 0:
            return True
            
        return False
    
    def retry_request(self, func, anti_crawler_controller, *args, **kwargs):
        """
        带重试的请求执行
        
        Args:
            func: 要执行的函数
            anti_crawler_controller: 反爬虫控制器
            *args, **kwargs: 函数参数
        """
        last_exception = None
        last_error_type = 'unknown'
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result, 'success'
                else:
                    # 函数返回None，可能是数据问题
                    last_error_type = 'data'
                    if not self.should_retry(last_error_type, attempt):
                        logger.info(f"数据相关问题，不再重试")
                        break
            except Exception as e:
                last_exception = e
                # 尝试从异常信息中分类错误
                error_msg = str(e)
                last_error_type = anti_crawler_controller.classify_error('', error_msg)
                
                if not self.should_retry(last_error_type, attempt):
                    if last_error_type == 'parameter':
                        logger.error(f"参数错误，不进行重试: {error_msg}")
                    else:
                        logger.error(f"错误类型 {last_error_type}，不再重试: {error_msg}")
                    break
                    
                if attempt < self.max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # 指数退避
                    logger.warning(f"请求失败，第 {attempt + 1} 次重试，等待 {wait_time:.2f} 秒，错误类型: {last_error_type}, 错误: {error_msg}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"请求最终失败，已重试 {self.max_retries} 次，错误类型: {last_error_type}, 错误: {error_msg}")
        
        return None, last_error_type

class IncrementalStockDataDownloader:
    """
    增量股票数据下载器，支持往前补数据和往后补数据
    """
    
    def __init__(self, output_dir: str = "/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/market_data/"):
        """
        初始化下载器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        self.ensure_output_dir()
        self.stock_list = []
        
        # 初始化反爬虫控制器
        self.anti_crawler = AntiCrawlerController()
        self.retry_manager = RequestRetryManager(max_retries=3)
        
        # 请求统计
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"创建输出目录: {self.output_dir}")
    
    def convert_to_baostock_code(self, code: str) -> str:
        """将股票代码转换为 baostock 格式"""
        if '.' in code:
            return code  # 已经是baostock格式
        
        if code.startswith('sz') or code.startswith('sh'):
            market = code[:2]
            stock_num = code[2:]
            return f"{market}.{stock_num}"
        else:
            # 根据股票代码首位判断市场
            if len(code) == 6 and code.isdigit():
                if code.startswith('0') or code.startswith('3'):
                    return f"sz.{code}"  # 深圳市场
                elif code.startswith('6'):
                    return f"sh.{code}"  # 上海市场
        
        return code  # 无法转换时返回原代码
    
    def login_baostock(self) -> bool:
        """登录 baostock 系统"""
        try:
            lg = bs.login()
            if lg.error_code != '0':
                logger.error(f"Baostock登录失败: {lg.error_msg}")
                return False
            logger.info("Baostock登录成功")
            return True
        except Exception as e:
            logger.error(f"Baostock登录异常: {str(e)}")
            return False
    
    def ensure_baostock_connection(self) -> bool:
        """确保baostock连接，仅在需要时连接"""
        if not hasattr(self, '_baostock_connected') or not self._baostock_connected:
            logger.info("检测到需要baostock连接，正在建立连接...")
            if self.login_baostock():
                self._baostock_connected = True
                return True
            else:
                logger.error("无法建立baostock连接")
                return False
        return True
    
    def logout_baostock(self):
        """登出 baostock 系统"""
        try:
            bs.logout()
            logger.info("Baostock登出成功")
        except Exception as e:
            logger.error(f"Baostock登出异常: {str(e)}")
    
    def get_existing_stock_files(self) -> List[str]:
        """获取已存在的股票数据文件列表"""
        pattern = os.path.join(self.output_dir, "*.csv")
        files = glob.glob(pattern)
        stock_codes = []
        
        for file in files:
            filename = os.path.basename(file)
            if filename.endswith('.csv'):
                # 提取股票代码 (如 sh.600000.csv -> sh.600000)
                stock_code = filename[:-4]
                stock_codes.append(stock_code)
        
        logger.info(f"发现 {len(stock_codes)} 个已存在的股票数据文件")
        return stock_codes
    
    def get_stock_data_date_range(self, stock_code: str) -> Tuple[Optional[str], Optional[str]]:
        """获取股票数据的日期范围"""
        baostock_code = self.convert_to_baostock_code(stock_code)
        file_path = os.path.join(self.output_dir, f"{baostock_code}.csv")
        
        if not os.path.exists(file_path):
            return None, None
        
        try:
            df = pd.read_csv(file_path)
            if 'date' not in df.columns or len(df) == 0:
                return None, None
            
            df['date'] = pd.to_datetime(df['date'])
            min_date = df['date'].min().strftime('%Y-%m-%d')
            max_date = df['date'].max().strftime('%Y-%m-%d')
            
            return min_date, max_date
        except Exception as e:
            logger.error(f"读取股票 {stock_code} 数据文件失败: {str(e)}")
            return None, None
    
    def get_stock_data_record_count(self, stock_code: str) -> int:
        """获取股票数据的记录数量"""
        baostock_code = self.convert_to_baostock_code(stock_code)
        file_path = os.path.join(self.output_dir, f"{baostock_code}.csv")
        
        if not os.path.exists(file_path):
            return 0
        
        try:
            df = pd.read_csv(file_path)
            return len(df)
        except Exception as e:
            logger.error(f"读取股票 {stock_code} 数据文件失败: {str(e)}")
            return 0
    
    def check_local_data_sufficiency(self, stock_code: str, backfill_days: int = None, forward_fill_date: str = None, max_total_days: int = None) -> dict:
        """检查本地数据充分性，避免不必要的网络请求"""
        result = {
            'need_backfill': False,
            'need_forward_fill': False,
            'backfill_days_needed': 0,
            'forward_fill_needed': False,
            'data_exists': False,
            'current_range': None
        }
        
        # 获取当前数据范围
        min_date, max_date = self.get_stock_data_date_range(stock_code)
        
        if min_date is None or max_date is None:
            # 没有本地数据，需要全部下载
            result['need_backfill'] = bool(backfill_days and backfill_days > 0)
            result['need_forward_fill'] = bool(forward_fill_date)
            result['backfill_days_needed'] = backfill_days or 0
            result['forward_fill_needed'] = bool(forward_fill_date)
            return result
        
        result['data_exists'] = True
        result['current_range'] = (min_date, max_date)
        
        # 检查是否需要往前补数据
        if backfill_days and backfill_days > 0:
            min_date_dt = datetime.strptime(min_date, '%Y-%m-%d')
            target_start_date = min_date_dt - timedelta(days=backfill_days)
            
            # 如果目标开始日期早于现有最早日期，需要补数据
            if target_start_date < min_date_dt:
                result['need_backfill'] = True
                result['backfill_days_needed'] = backfill_days
        
        # 检查是否需要往后补数据
        if forward_fill_date:
            max_date_dt = datetime.strptime(max_date, '%Y-%m-%d')
            target_end_date = datetime.strptime(forward_fill_date, '%Y-%m-%d')
            
            # 如果目标结束日期晚于现有最新日期，需要补数据
            if target_end_date > max_date_dt:
                result['need_forward_fill'] = True
                result['forward_fill_needed'] = True
        
        # 检查总记录数限制
        if max_total_days and max_total_days > 0:
            current_record_count = self.get_stock_data_record_count(stock_code)
            if current_record_count >= max_total_days:
                logger.info(f"股票 {stock_code} 当前记录数 {current_record_count} 已达到或超过限制 {max_total_days}，跳过处理")
                result['need_backfill'] = False
                result['need_forward_fill'] = False
                result['backfill_days_needed'] = 0
                result['forward_fill_needed'] = False
        
        return result
    
    def validate_date_range(self, start_date: str, end_date: str, stock_code: str = None) -> bool:
        """
        验证日期范围的有效性
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_code: 股票代码（用于日志）
            
        Returns:
            日期范围是否有效
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_dt > end_dt:
                logger.error(f"股票 {stock_code or 'unknown'}: 起始日期 {start_date} 大于终止日期 {end_date}")
                return False
                
            # 检查日期是否过于久远（超过30年）
            current_date = datetime.now()
            if start_dt < current_date - timedelta(days=30*365):
                logger.warning(f"股票 {stock_code or 'unknown'}: 起始日期 {start_date} 过于久远，可能没有数据")
                
            # 检查是否请求未来日期
            if end_dt > current_date + timedelta(days=1):
                logger.warning(f"股票 {stock_code or 'unknown'}: 终止日期 {end_date} 是未来日期，调整为今天")
                
            return True
            
        except ValueError as e:
            logger.error(f"股票 {stock_code or 'unknown'}: 日期格式错误 - {str(e)}")
            return False
    
    def download_stock_data_range(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """下载指定日期范围的股票数据"""
        # 验证日期范围
        if not self.validate_date_range(start_date, end_date, stock_code):
            logger.error(f"股票 {stock_code}: 日期范围验证失败，跳过下载")
            # 记录参数错误
            self.anti_crawler.after_request(False, 'parameter')
            self.failed_requests += 1
            return None
            
        # 确保baostock连接
        if not self.ensure_baostock_connection():
            logger.error(f"无法建立baostock连接，无法下载股票 {stock_code} 数据")
            return None
            
        def _download_data():
            baostock_code = self.convert_to_baostock_code(stock_code)
            
            logger.info(f"下载股票 {stock_code} ({baostock_code}) 数据，日期范围: {start_date} 到 {end_date}")
            
            # 检查是否需要暂停
            if self.anti_crawler.should_pause():
                self.anti_crawler.pause_and_recover()
            
            # 反爬虫措施：智能延时
            self.anti_crawler.before_request()
            self.total_requests += 1
            
            # 使用baostock API获取股票历史数据
            rs = bs.query_history_k_data_plus(
                baostock_code,
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"  # 3=后复权
            )
            
            if rs.error_code != '0':
                error_type = self.anti_crawler.classify_error(rs.error_code, rs.error_msg)
                logger.error(f"查询股票 {stock_code} 数据失败: {rs.error_msg} (错误类型: {error_type})")
                self.anti_crawler.after_request(False, error_type)
                self.failed_requests += 1
                
                # 如果是参数错误，直接抛出异常，不进行重试
                if error_type == 'parameter':
                    raise ValueError(f"参数错误: {rs.error_msg}")
                return None
            
            # 收集数据
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                logger.warning(f"股票 {stock_code} 在指定日期范围内没有返回数据")
                self.anti_crawler.after_request(False, 'data')
                self.failed_requests += 1
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 数据类型转换
            numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            
            # 过滤掉停牌和ST股票的数据
            df = df[(df['tradestatus'] == '1') & (df['isST'] == '0')]
            
            # 数据验证
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns or df[col].isna().all():
                    logger.warning(f"股票 {stock_code} 缺少必要的 {col} 数据")
                    self.anti_crawler.after_request(False, 'data')
                    self.failed_requests += 1
                    return None
            
            # 检查OHLC数据的合理性
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['open'] > df['high']) | (df['open'] < df['low']) |
                (df['close'] > df['high']) | (df['close'] < df['low']) |
                (df['high'] <= 0) | (df['low'] <= 0) |
                (df['open'] <= 0) | (df['close'] <= 0)
            )
            
            if invalid_ohlc.any():
                logger.warning(f"股票 {stock_code} 存在 {invalid_ohlc.sum()} 行无效的OHLC数据，将被移除")
                df = df[~invalid_ohlc]
            
            if len(df) == 0:
                logger.warning(f"股票 {stock_code} 移除无效数据后没有剩余数据")
                return None
            
            # 按日期排序
            df = df.sort_values('date')
            
            logger.info(f"成功获取股票 {stock_code} 数据，共 {len(df)} 条记录")
            logger.info(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
            
            self.anti_crawler.after_request(True)
            self.successful_requests += 1
            return df
        
        # 使用重试机制执行下载
        try:
            result, error_type = self.retry_manager.retry_request(_download_data, self.anti_crawler)
            if result is not None:
                return result
            else:
                # 重试失败，记录最终错误
                self.anti_crawler.after_request(False, error_type)
                self.failed_requests += 1
                return None
        except Exception as e:
            error_type = self.anti_crawler.classify_error('', str(e))
            logger.error(f"下载股票 {stock_code} 数据失败: {str(e)} (错误类型: {error_type})")
            self.anti_crawler.after_request(False, error_type)
            self.failed_requests += 1
            return None
    
    def merge_and_save_data(self, stock_code: str, new_data: pd.DataFrame) -> bool:
        """合并新数据与现有数据并保存"""
        try:
            baostock_code = self.convert_to_baostock_code(stock_code)
            file_path = os.path.join(self.output_dir, f"{baostock_code}.csv")
            
            if os.path.exists(file_path):
                # 读取现有数据
                existing_data = pd.read_csv(file_path)
                existing_data['date'] = pd.to_datetime(existing_data['date'])
                
                # 合并数据
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                
                # 去重并排序
                combined_data = combined_data.drop_duplicates(subset=['date'], keep='last')
                combined_data = combined_data.sort_values('date')
                
                logger.info(f"股票 {stock_code}: 原有 {len(existing_data)} 条记录，新增 {len(new_data)} 条记录，合并后 {len(combined_data)} 条记录")
            else:
                # 没有现有数据，直接使用新数据
                combined_data = new_data.sort_values('date')
                logger.info(f"股票 {stock_code}: 新建文件，共 {len(combined_data)} 条记录")
            
            # 保存数据
            combined_data.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"成功保存股票 {stock_code} 数据到 {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"合并保存股票 {stock_code} 数据失败: {str(e)}")
            return False
    
    def backfill_stock_data(self, stock_code: str, days: int) -> bool:
        """往前补充股票数据"""
        try:
            # 获取现有数据的日期范围
            min_date, max_date = self.get_stock_data_date_range(stock_code)
            
            if min_date is None:
                logger.warning(f"股票 {stock_code} 没有现有数据，跳过往前补数据")
                return False
            
            # 计算需要补充的开始日期
            min_date_obj = datetime.strptime(min_date, '%Y-%m-%d')
            start_date = (min_date_obj - timedelta(days=days + 50)).strftime('%Y-%m-%d')  # 多获取一些天数确保有足够交易日
            end_date = (min_date_obj - timedelta(days=1)).strftime('%Y-%m-%d')  # 到现有数据的前一天
            
            logger.info(f"股票 {stock_code} 往前补数据: {start_date} 到 {end_date} (目标补充 {days} 个交易日)")
            
            # 下载数据
            new_data = self.download_stock_data_range(stock_code, start_date, end_date)
            if new_data is None or len(new_data) == 0:
                logger.warning(f"股票 {stock_code} 往前补数据失败：没有获取到数据")
                return False
            
            # 取最新的指定天数
            new_data = new_data.tail(days)
            
            # 合并并保存数据
            return self.merge_and_save_data(stock_code, new_data)
            
        except Exception as e:
            logger.error(f"股票 {stock_code} 往前补数据失败: {str(e)}")
            return False
    
    def forward_fill_stock_data(self, stock_code: str, target_date: str) -> bool:
        """往后补充股票数据到指定日期"""
        try:
            # 获取现有数据的日期范围
            min_date, max_date = self.get_stock_data_date_range(stock_code)
            
            if max_date is None:
                logger.warning(f"股票 {stock_code} 没有现有数据，跳过往后补数据")
                return False
            
            # 检查目标日期是否在现有数据之后
            max_date_obj = datetime.strptime(max_date, '%Y-%m-%d')
            target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            
            if target_date_obj <= max_date_obj:
                logger.info(f"股票 {stock_code} 现有数据已覆盖到 {max_date}，无需补充到 {target_date}")
                return True
            
            # 计算需要补充的日期范围
            start_date = (max_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')  # 从现有数据的下一天开始
            end_date = target_date
            
            logger.info(f"股票 {stock_code} 往后补数据: {start_date} 到 {end_date}")
            
            # 下载数据
            new_data = self.download_stock_data_range(stock_code, start_date, end_date)
            if new_data is None or len(new_data) == 0:
                logger.warning(f"股票 {stock_code} 往后补数据失败：没有获取到数据")
                return False
            
            # 合并并保存数据
            return self.merge_and_save_data(stock_code, new_data)
            
        except Exception as e:
            logger.error(f"股票 {stock_code} 往后补数据失败: {str(e)}")
            return False
    
    def get_all_stock_list(self, day: str = None) -> List[str]:
        """获取A股全市场股票列表"""
        # 确保baostock连接
        if not self.ensure_baostock_connection():
            logger.error("无法建立baostock连接，无法获取股票列表")
            return []
            
        if day is None:
            day = datetime.now().strftime('%Y-%m-%d')
            
        # 尝试多个日期，从当前日期开始往前回退
        for i in range(5):
            try_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            
            try:
                logger.info(f"尝试获取A股全市场股票列表，查询日期: {try_date}")
                
                rs = bs.query_all_stock(day=try_date)
                stock_list = []
                
                while (rs.error_code == '0') & rs.next():
                    stock_list.append(rs.get_row_data())
                
                if not stock_list:
                    logger.warning(f"日期 {try_date} 未获取到股票列表，尝试前一天")
                    continue
                else:
                    logger.info(f"成功从日期 {try_date} 获取到股票列表")
                    break
                    
            except Exception as e:
                logger.warning(f"查询日期 {try_date} 失败: {str(e)}，尝试前一天")
                continue
        
        if not stock_list:
            logger.error("所有尝试日期都无法获取到股票列表")
            return []
            
        try:
            df = pd.DataFrame(stock_list, columns=rs.fields)
            logger.info(f"原始股票数据字段: {rs.fields}")
            logger.info(f"原始股票数量: {len(df)}")
            
            # 过滤A股股票
            if 'code' in df.columns:
                pattern = r'^(sh\.6\d{5}|sz\.[03]\d{5})$'
                df = df[df['code'].str.match(pattern, na=False)]
                logger.info(f"过滤A股后数量: {len(df)}")
            
            # 排除ST股票和退市股票
            if 'code_name' in df.columns:
                original_count = len(df)
                df = df[~df['code_name'].str.contains('ST|退市|暂停', na=False)]
                logger.info(f"排除ST和退市股票后数量: {len(df)} (排除了{original_count - len(df)}只)")
            
            # 提取股票代码（保留baostock格式）
            stock_codes = df['code'].tolist()
            
            logger.info(f"最终获取到{len(stock_codes)}只A股股票")
            self.stock_list = stock_codes
            return stock_codes
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {str(e)}")
            return []
    
    def incremental_update(self, backfill_days: int = None, forward_fill_date: str = None, 
                          stock_codes: List[str] = None, max_stocks: int = None, max_total_days: int = None) -> dict:
        """增量更新股票数据"""
        # 重置统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        try:
            # 确定要处理的股票列表
            if stock_codes:
                # 使用指定的股票代码
                target_stocks = stock_codes
                logger.info(f"使用指定的 {len(target_stocks)} 只股票进行增量更新")
            else:
                # 获取现有股票文件列表
                existing_stocks = self.get_existing_stock_files()
                if not existing_stocks:
                    logger.warning("没有找到现有股票数据文件，将获取全市场股票列表")
                    all_stocks = self.get_all_stock_list()
                    target_stocks = all_stocks[:max_stocks] if max_stocks else all_stocks
                else:
                    target_stocks = existing_stocks
                    if max_stocks:
                        target_stocks = target_stocks[:max_stocks]
                
                logger.info(f"找到 {len(target_stocks)} 只股票需要增量更新")
            
            if not target_stocks:
                logger.error("没有找到需要处理的股票")
                return {'success': 0, 'failed': 0, 'failed_stocks': [], 'total': 0}
            
            # 预检查：确定哪些股票真正需要更新
            logger.info("正在检查本地数据充分性...")
            stocks_needing_update = []
            for stock_code in target_stocks:
                sufficiency = self.check_local_data_sufficiency(stock_code, backfill_days, forward_fill_date, max_total_days)
                if sufficiency['need_backfill'] or sufficiency['need_forward_fill']:
                    stocks_needing_update.append((stock_code, sufficiency))
                else:
                    logger.debug(f"股票 {stock_code} 本地数据充分，跳过更新")
            
            if not stocks_needing_update:
                logger.info("所有股票的本地数据都是充分的，无需网络请求")
                return {'success': len(target_stocks), 'failed': 0, 'failed_stocks': [], 'total': len(target_stocks)}
            
            logger.info(f"需要更新的股票数量: {len(stocks_needing_update)}/{len(target_stocks)}")
            
            success_count = 0
            failed_stocks = []
            
            for i, (stock_code, sufficiency) in enumerate(stocks_needing_update, 1):
                logger.info(f"进度: {i}/{len(stocks_needing_update)} - 正在处理股票 {stock_code}")
                
                # 检查是否需要暂停
                if self.anti_crawler.should_pause():
                    logger.info(f"处理了 {i-1} 只股票，触发反爬虫暂停机制")
                    self.anti_crawler.pause_and_recover()
                
                stock_success = True
                
                # 往前补数据（只有在需要时才执行）
                if sufficiency['need_backfill'] and backfill_days and backfill_days > 0:
                    if not self.backfill_stock_data(stock_code, backfill_days):
                        logger.warning(f"股票 {stock_code} 往前补数据失败")
                        stock_success = False
                
                # 往后补数据（只有在需要时才执行）
                if sufficiency['need_forward_fill'] and forward_fill_date:
                    if not self.forward_fill_stock_data(stock_code, forward_fill_date):
                        logger.warning(f"股票 {stock_code} 往后补数据失败")
                        stock_success = False
                
                if stock_success:
                    success_count += 1
                else:
                    failed_stocks.append(stock_code)
                
                # 反爬虫措施：智能延时控制
                if i % 10 == 0:
                    if not self.anti_crawler.should_pause():
                        delay = random.uniform(1.0, 2.0)
                        logger.info(f"已处理{i}只股票，休息{delay:.1f}秒...")
                        time.sleep(delay)
                else:
                    delay = random.uniform(0.3, 0.8)
                    time.sleep(delay)
            
            # 计算成功率
            success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
            
            logger.info(f"\n=== 反爬虫统计信息 ===")
            logger.info(f"总请求数: {self.total_requests}")
            logger.info(f"成功请求数: {self.successful_requests}")
            logger.info(f"失败请求数: {self.failed_requests}")
            logger.info(f"API成功率: {success_rate:.2f}%")
            
            result = {
                'success': success_count,
                'failed': len(failed_stocks),
                'failed_stocks': failed_stocks,
                'total': len(target_stocks),
                'processed': len(stocks_needing_update),
                'skipped': len(target_stocks) - len(stocks_needing_update),
                'anti_crawler_stats': {
                    'total_requests': self.total_requests,
                    'successful_requests': self.successful_requests,
                    'failed_requests': self.failed_requests,
                    'success_rate': success_rate
                }
            }
            
            stock_success_rate = (success_count / len(stocks_needing_update)) * 100 if stocks_needing_update else 100
            efficiency_rate = ((len(target_stocks) - len(stocks_needing_update)) / len(target_stocks)) * 100 if target_stocks else 0
            
            logger.info(f"\n=== 增量更新完成 ===")
            logger.info(f"总股票数: {len(target_stocks)}")
            logger.info(f"需要更新: {len(stocks_needing_update)}")
            logger.info(f"跳过更新: {len(target_stocks) - len(stocks_needing_update)} (本地数据充分)")
            logger.info(f"成功处理: {success_count}")
            logger.info(f"处理失败: {len(failed_stocks)}")
            logger.info(f"处理成功率: {stock_success_rate:.1f}%")
            logger.info(f"效率提升: {efficiency_rate:.1f}% (跳过不必要的网络请求)")
            
            if failed_stocks:
                logger.warning(f"失败的股票数量: {len(failed_stocks)}")
                if len(failed_stocks) <= 10:
                    logger.warning(f"失败的股票: {failed_stocks}")
            
            return result
            
        finally:
            # 只有在建立了连接时才登出
            if hasattr(self, 'bs_logged_in') and self.bs_logged_in:
                self.logout_baostock()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增量股票数据下载器')
    parser.add_argument('--backfill-days', type=int, default=None, 
                       help='往前补充的天数，例如：--backfill-days 30')
    parser.add_argument('--forward-fill-date', type=str, default=None,
                       help='往后补充到的日期，格式：YYYY-MM-DD，例如：--forward-fill-date 2024-12-31')
    parser.add_argument('--stock-codes', type=str, nargs='+', default=None,
                       help='指定股票代码列表，例如：--stock-codes sh.600000 sz.000001')
    parser.add_argument('--max-stocks', type=int, default=None,
                       help='最大处理股票数量，默认全部')
    parser.add_argument('--max-total-days', type=int, default=None,
                       help='控制总记录数，超过这个值就不处理，否则补齐数据')
    parser.add_argument('--test', action='store_true',
                       help='测试模式，只处理前5只股票')
    
    args = parser.parse_args()
    
    # 参数验证
    if not args.backfill_days and not args.forward_fill_date:
        print("错误：必须指定 --backfill-days 或 --forward-fill-date 中的至少一个参数")
        print("使用 --help 查看详细帮助")
        return
    
    if args.forward_fill_date:
        try:
            datetime.strptime(args.forward_fill_date, '%Y-%m-%d')
        except ValueError:
            print(f"错误：日期格式不正确，应为 YYYY-MM-DD，您输入的是: {args.forward_fill_date}")
            return
    
    # 测试模式
    if args.test:
        args.max_stocks = 5
        logger.info("=== 测试模式：只处理前5只股票 ===")
    
    print("=== 增量股票数据下载器 ===")
    print("支持往前补数据和往后补数据")
    
    # 创建下载器
    downloader = IncrementalStockDataDownloader()
    
    print(f"输出目录: {downloader.output_dir}")
    if args.backfill_days:
        print(f"往前补充天数: {args.backfill_days}")
    if args.forward_fill_date:
        print(f"往后补充到日期: {args.forward_fill_date}")
    if args.stock_codes:
        print(f"指定股票代码: {args.stock_codes}")
    if args.max_stocks:
        print(f"限制处理数量: {args.max_stocks}")
    if args.max_total_days:
        print(f"总记录数限制: {args.max_total_days}")
    
    # 开始增量更新
    start_time = datetime.now()
    result = downloader.incremental_update(
        backfill_days=args.backfill_days,
        forward_fill_date=args.forward_fill_date,
        stock_codes=args.stock_codes,
        max_stocks=args.max_stocks,
        max_total_days=args.max_total_days
    )
    end_time = datetime.now()
    
    # 输出结果
    print("\n=== 增量更新结果 ===")
    print(f"总股票数: {result['total']}")
    print(f"成功处理: {result['success']}")
    print(f"处理失败: {result['failed']}")
    if result['total'] > 0:
        print(f"成功率: {result['success']/result['total']*100:.1f}%")
    print(f"总耗时: {end_time - start_time}")
    
    if result['failed_stocks']:
        print(f"失败的股票: {result['failed_stocks']}")
    
    print("\n增量更新完成！")

if __name__ == "__main__":
    main()