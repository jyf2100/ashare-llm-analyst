#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baostock RPS数据采集器
使用Baostock API采集股票数据并计算RPS指标
"""

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import os
import pickle
from typing import Dict, List, Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaostockRPSCollector:
    """
    使用Baostock采集股票数据并计算RPS指标的类
    """
    
    def __init__(self, cache_dir="rps_cache"):
        """
        初始化Baostock RPS采集器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.bs_session = None
        self.stock_list = None
        self.price_data = {}
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"创建缓存目录: {self.cache_dir}")
        
    def login(self) -> bool:
        """
        登录Baostock系统
        
        Returns:
            bool: 登录是否成功
        """
        try:
            lg = bs.login()
            if lg.error_code != '0':
                logger.error(f"Baostock登录失败: {lg.error_msg}")
                return False
            logger.info("Baostock登录成功")
            self.bs_session = True
            return True
        except Exception as e:
            logger.error(f"Baostock登录异常: {str(e)}")
            return False
    
    def logout(self):
        """
        登出Baostock系统
        """
        if self.bs_session:
            bs.logout()
            logger.info("Baostock登出成功")
            self.bs_session = None
    
    def _get_cache_filename(self, date_str: str) -> str:
        """
        获取缓存文件名
        
        Args:
            date_str: 日期字符串，格式为YYYY-MM-DD
            
        Returns:
            str: 缓存文件路径
        """
        return os.path.join(self.cache_dir, f"rps_data_{date_str}.pkl")
    
    def save_rps_data(self, rps_df: pd.DataFrame, latest_rps: Dict, date_str: str = None) -> bool:
        """
        保存RPS数据到硬盘
        
        Args:
            rps_df: RPS时间序列数据
            latest_rps: 最新RPS值字典
            date_str: 日期字符串，默认使用今天日期
            
        Returns:
            bool: 保存是否成功
        """
        try:
            if date_str is None:
                date_str = datetime.now().strftime('%Y-%m-%d')
            
            cache_file = self._get_cache_filename(date_str)
            
            # 准备保存的数据
            cache_data = {
                'rps_df': rps_df,
                'latest_rps': latest_rps,
                'save_time': datetime.now(),
                'date': date_str
            }
            
            # 保存到文件
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"RPS数据已保存到: {cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存RPS数据失败: {str(e)}")
            return False
    
    def load_rps_data(self, date_str: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        从硬盘加载RPS数据
        
        Args:
            date_str: 日期字符串，默认使用今天日期
            
        Returns:
            Tuple[pd.DataFrame, Dict]: RPS时间序列数据和最新RPS值字典
        """
        try:
            if date_str is None:
                date_str = datetime.now().strftime('%Y-%m-%d')
            
            cache_file = self._get_cache_filename(date_str)
            
            if not os.path.exists(cache_file):
                logger.info(f"缓存文件不存在: {cache_file}")
                return pd.DataFrame(), {}
            
            # 检查文件是否是今天创建的
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            today = datetime.now().date()
            
            if file_mtime.date() != today:
                logger.info(f"缓存文件不是今天创建的，将重新获取数据")
                return pd.DataFrame(), {}
            
            # 加载数据
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            rps_df = cache_data.get('rps_df', pd.DataFrame())
            latest_rps = cache_data.get('latest_rps', {})
            save_time = cache_data.get('save_time')
            
            logger.info(f"成功加载RPS缓存数据，保存时间: {save_time}")
            logger.info(f"数据包含 {len(rps_df.columns)} 只股票，{len(rps_df)} 个交易日")
            
            return rps_df, latest_rps
            
        except Exception as e:
            logger.error(f"加载RPS数据失败: {str(e)}")
            return pd.DataFrame(), {}
    
    def clear_old_cache(self, days_to_keep: int = 7) -> None:
        """
        清理旧的缓存文件
        
        Args:
            days_to_keep: 保留最近几天的缓存文件
        """
        try:
            if not os.path.exists(self.cache_dir):
                return
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.startswith('rps_data_') and filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_mtime < cutoff_date:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.info(f"删除旧缓存文件: {filename}")
            
            if deleted_count > 0:
                logger.info(f"清理完成，删除了 {deleted_count} 个旧缓存文件")
            else:
                logger.info("没有需要清理的旧缓存文件")
                
        except Exception as e:
            logger.error(f"清理缓存失败: {str(e)}")
    
    def get_stock_list(self, day: str = None) -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            day: 查询日期，格式YYYY-MM-DD，默认为当前日期
            
        Returns:
            pd.DataFrame: 股票列表数据
        """
        if not self.bs_session:
            logger.error("请先登录Baostock")
            return pd.DataFrame()
            
        if day is None:
            # 使用一个较早的日期，确保有数据
            day = '2024-12-31'
            
        try:
            # 获取沪深A股列表
            rs = bs.query_all_stock(day=day)
            stock_list = []
            
            while (rs.error_code == '0') & rs.next():
                stock_list.append(rs.get_row_data())
            
            if not stock_list:
                logger.warning("未获取到股票列表")
                return pd.DataFrame()
                
            df = pd.DataFrame(stock_list, columns=rs.fields)
            
            # 打印字段信息用于调试
            logger.info(f"返回字段: {rs.fields}")
            logger.info(f"数据样本: {df.head() if not df.empty else '无数据'}")
            
            # 过滤A股股票（排除指数、B股等）
            # 根据实际字段调整过滤逻辑
            if 'code' in df.columns:
                # 过滤A股股票代码：sh.6开头（上海A股）或sz.0/3开头（深圳A股）
                pattern = r'^(sh\.6\d{5}|sz\.[03]\d{5})$'
                df = df[df['code'].str.match(pattern, na=False)]
            
            # 排除ST股票和退市股票
            if 'code_name' in df.columns:
                df = df[~df['code_name'].str.contains('ST|退市|暂停', na=False)]
            
            logger.info(f"获取到{len(df)}只A股股票")
            self.stock_list = df
            return df
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {str(e)}")
            return pd.DataFrame()
    
    def get_stock_data(self, code: str, start_date: str, end_date: str, 
                      frequency: str = 'd') -> pd.DataFrame:
        """
        获取单只股票的历史数据
        
        Args:
            code: 股票代码，如'sh.600000'
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            frequency: 数据频率，d=日线，w=周线，m=月线
            
        Returns:
            pd.DataFrame: 股票历史数据
        """
        if not self.bs_session:
            logger.error("请先登录Baostock")
            return pd.DataFrame()
            
        try:
            # 查询历史K线数据
            rs = bs.query_history_k_data_plus(
                code, 
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                start_date=start_date, 
                end_date=end_date,
                frequency=frequency, 
                adjustflag="3"  # 3=后复权
            )
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                return pd.DataFrame()
                
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 数据类型转换
            numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 过滤掉停牌和ST股票的数据
            df = df[(df['tradestatus'] == '1') & (df['isST'] == '0')]
            
            return df
            
        except Exception as e:
            logger.error(f"获取股票{code}数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_all_stocks_data(self, start_date: str, end_date: str, 
                           max_stocks: int = None) -> Dict[str, pd.DataFrame]:
        """
        获取所有股票的历史数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            max_stocks: 最大股票数量限制，用于测试
            
        Returns:
            Dict[str, pd.DataFrame]: 股票代码到数据的映射
        """
        if self.stock_list is None or self.stock_list.empty:
            logger.error("请先获取股票列表")
            return {}
            
        stock_data = {}
        stock_codes = self.stock_list['code'].tolist()
        
        if max_stocks:
            stock_codes = stock_codes[:max_stocks]
            
        total_stocks = len(stock_codes)
        logger.info(f"开始获取{total_stocks}只股票的历史数据")
        
        for i, code in enumerate(stock_codes, 1):
            try:
                df = self.get_stock_data(code, start_date, end_date)
                if not df.empty:
                    stock_data[code] = df
                    logger.info(f"进度: {i}/{total_stocks} - 成功获取{code}数据，共{len(df)}条记录")
                else:
                    logger.warning(f"进度: {i}/{total_stocks} - {code}数据为空")
                    
                # 避免请求过快
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"进度: {i}/{total_stocks} - 获取{code}数据失败: {str(e)}")
                continue
        
        logger.info(f"成功获取{len(stock_data)}只股票的数据")
        self.price_data = stock_data
        return stock_data
    
    def calculate_returns(self, period_days: int = 20) -> pd.DataFrame:
        """
        计算所有股票的收益率
        
        Args:
            period_days: 计算收益率的周期天数
            
        Returns:
            pd.DataFrame: 股票收益率数据，行为日期，列为股票代码
        """
        if not self.price_data:
            logger.error("请先获取股票数据")
            return pd.DataFrame()
            
        returns_data = {}
        
        for code, df in self.price_data.items():
            if len(df) > period_days:
                # 计算period_days天的收益率
                returns = df['close'].pct_change(periods=period_days).dropna()
                returns_data[code] = returns
        
        if not returns_data:
            logger.error("无法计算收益率数据")
            return pd.DataFrame()
            
        # 合并所有股票的收益率数据
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.fillna(0)  # 填充缺失值为0
        
        logger.info(f"计算了{len(returns_df.columns)}只股票的{period_days}日收益率")
        return returns_df
    
    def calculate_rps(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算RPS指标
        
        Args:
            returns_df: 收益率数据框
            
        Returns:
            pd.DataFrame: RPS数据，行为日期，列为股票代码
        """
        if returns_df.empty:
            logger.error("收益率数据为空")
            return pd.DataFrame()
            
        rps_data = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        
        for date in returns_df.index:
            # 获取当日所有股票的收益率
            daily_returns = returns_df.loc[date].dropna()
            
            if len(daily_returns) == 0:
                continue
                
            # 按收益率排序
            sorted_returns = daily_returns.sort_values(ascending=False)
            
            # 计算RPS值
            total_stocks = len(sorted_returns)
            for i, (code, ret) in enumerate(sorted_returns.items()):
                rank = i + 1
                rps_value = (1 - rank / total_stocks) * 100
                rps_data.loc[date, code] = rps_value
        
        logger.info(f"计算了{len(rps_data)}个交易日的RPS数据")
        return rps_data.astype(float)
    
    def get_latest_rps(self, stock_codes: List[str] = None, 
                      period_days: int = 20) -> Dict[str, float]:
        """
        获取最新的RPS值
        
        Args:
            stock_codes: 指定股票代码列表，如果为None则返回所有股票
            period_days: RPS计算周期
            
        Returns:
            Dict[str, float]: 股票代码到最新RPS值的映射
        """
        # 计算收益率
        returns_df = self.calculate_returns(period_days)
        if returns_df.empty:
            return {}
            
        # 计算RPS
        rps_df = self.calculate_rps(returns_df)
        if rps_df.empty:
            return {}
            
        # 获取最新日期的RPS值
        latest_date = rps_df.index[-1]
        latest_rps = rps_df.loc[latest_date].dropna().to_dict()
        
        # 如果指定了股票代码，则只返回指定股票的RPS
        if stock_codes:
            latest_rps = {code: latest_rps.get(code, 50.0) for code in stock_codes}
            
        logger.info(f"获取了{len(latest_rps)}只股票的最新RPS值")
        return latest_rps
    
    def collect_rps_data(self, start_date: str = None, end_date: str = None,
                        period_days: int = 20, max_stocks: int = None, 
                        use_cache: bool = True) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        完整的RPS数据采集流程，支持缓存功能
        
        Args:
            start_date: 开始日期，默认为60天前
            end_date: 结束日期，默认为今天
            period_days: RPS计算周期
            max_stocks: 最大股票数量限制
            use_cache: 是否使用缓存
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, float]]: (RPS时间序列数据, 最新RPS值)
        """
        try:
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=60 + period_days)).strftime('%Y-%m-%d')
            
            # 尝试从缓存加载数据
            if use_cache:
                logger.info("检查是否存在今日RPS缓存数据...")
                cached_rps_df, cached_latest_rps = self.load_rps_data()
                
                if not cached_rps_df.empty and cached_latest_rps:
                    logger.info("使用缓存的RPS数据")
                    return cached_rps_df, cached_latest_rps
                else:
                    logger.info("未找到有效缓存，将重新采集数据")
            
            logger.info(f"开始采集RPS数据，时间范围: {start_date} 到 {end_date}")
            
            # 登录
            if not self.login():
                return pd.DataFrame(), {}
            
            try:
                # 获取股票列表
                self.get_stock_list()
                
                # 获取股票数据
                self.get_all_stocks_data(start_date, end_date, max_stocks)
                
                # 计算收益率和RPS
                returns_df = self.calculate_returns(period_days)
                rps_df = self.calculate_rps(returns_df)
                
                # 获取最新RPS值
                latest_rps = {}
                if not rps_df.empty:
                    latest_date = rps_df.index[-1]
                    latest_rps = rps_df.loc[latest_date].dropna().to_dict()
                
                logger.info(f"RPS数据采集完成，共{len(rps_df.columns)}只股票，{len(rps_df)}个交易日")
                
                # 保存到缓存
                if use_cache and not rps_df.empty:
                    self.save_rps_data(rps_df, latest_rps)
                    # 清理旧缓存
                    self.clear_old_cache()
                
                return rps_df, latest_rps
                
            finally:
                # 确保登出
                self.logout()
                
        except Exception as e:
            logger.error(f"RPS数据采集失败: {str(e)}")
            return pd.DataFrame(), {}


def main():
    """
    测试函数
    """
    collector = BaostockRPSCollector()
    
    # 采集最近30天的RPS数据，限制10只股票用于测试
    rps_df, latest_rps = collector.collect_rps_data(
        period_days=20, 
        max_stocks=10
    )
    
    if not rps_df.empty:
        print("\n=== RPS时间序列数据 ===")
        print(rps_df.tail())
        
        print("\n=== 最新RPS值 ===")
        for code, rps in sorted(latest_rps.items(), key=lambda x: x[1], reverse=True):
            print(f"{code}: {rps:.2f}")
    else:
        print("未能获取到RPS数据")


if __name__ == "__main__":
    main()