#!/usr/bin/env python3
"""
数据获取模块
负责从各种数据源获取财务数据
"""
import requests
import baostock as bs
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time

from config_manager import config
from logger_manager import get_logger
from exception_handler import api_retry, safe_data_processing, APIError, DataError
from cache_manager import cache_manager

logger = get_logger('data_fetcher')

def format_stock_code(stock_code: str) -> str:
    """
    格式化股票代码为baostock API需要的9位格式
    
    Args:
        stock_code: 股票代码，可能是6位或9位格式
        
    Returns:
        str: 9位格式的股票代码 (如 sh.600000, sz.000001)
    """
    if not stock_code:
        return stock_code
    
    # 如果已经是9位格式，直接返回
    if len(stock_code) == 9 and ('sh.' in stock_code or 'sz.' in stock_code):
        return stock_code
    
    # 如果是6位数字，需要添加前缀
    if len(stock_code) == 6 and stock_code.isdigit():
        # 6开头的是上海股票
        if stock_code.startswith('6'):
            return f"sh.{stock_code}"
        # 0开头的是深圳股票
        elif stock_code.startswith('0') or stock_code.startswith('3'):
            return f"sz.{stock_code}"
        # 其他情况默认为深圳
        else:
            return f"sz.{stock_code}"
    
    # 如果格式不对，返回原值
    return stock_code

class DataFetcher:
    """数据获取器"""
    
    def __init__(self) -> None:
        self.session: requests.Session = requests.Session()
        self.session.headers.update(config.get_headers())
        
        # 设置代理
        proxy_config = config.get_proxy_config()
        if proxy_config:
            self.session.proxies.update(proxy_config)
        
        self._baostock_logged_in: bool = False
    
    def _ensure_baostock_login(self) -> None:
        """确保BaoStock已登录"""
        if not self._baostock_logged_in:
            lg = bs.login()
            if lg.error_code != '0':
                raise APIError(f"BaoStock登录失败: {lg.error_msg}", "baostock")
            self._baostock_logged_in = True
            logger.info("BaoStock登录成功")
    
    def __del__(self) -> None:
        """析构函数，确保BaoStock登出"""
        if self._baostock_logged_in:
            bs.logout()
    
    @api_retry(max_retries=3, delay=1.0)
    @safe_data_processing(default_value={})
    def get_company_basic_info(self, stock_code: str) -> Dict[str, Any]:
        """获取公司基本信息"""
        # 格式化股票代码
        formatted_code = format_stock_code(stock_code)
        
        cache_key = f"company_basic_{formatted_code}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        self._ensure_baostock_login()
        
        try:
            # 获取股票基本信息
            rs = bs.query_stock_basic(code=formatted_code)
            if rs.error_code != '0':
                raise APIError(f"获取股票基本信息失败: {rs.error_msg}", "baostock")
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                raise DataError(f"未找到股票 {formatted_code} 的基本信息")
            
            # 转换为字典格式
            columns = rs.fields
            basic_info = dict(zip(columns, data_list[0]))
            
            cache_manager.set(cache_key, basic_info)
            logger.info(f"获取公司基本信息成功: {formatted_code}")
            return basic_info
            
        except Exception as e:
            logger.error(f"获取公司基本信息失败: {formatted_code}, 错误: {e}")
            raise
    
    @api_retry(max_retries=3, delay=1.0)
    @safe_data_processing(default_value=pd.DataFrame())
    def get_stock_k_data(self, stock_code: str, start_date: str, end_date: str, 
                        frequency: str = "d") -> pd.DataFrame:
        """获取股票K线数据"""
        # 格式化股票代码
        formatted_code = format_stock_code(stock_code)
        
        cache_key = f"k_data_{formatted_code}_{start_date}_{end_date}_{frequency}"
        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        self._ensure_baostock_login()
        
        try:
            fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
            rs = bs.query_history_k_data_plus(
                formatted_code, fields, start_date=start_date, 
                end_date=end_date, frequency=frequency, adjustflag="3"
            )
            
            if rs.error_code != '0':
                raise APIError(f"获取K线数据失败: {rs.error_msg}", "baostock")
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                logger.warning(f"未获取到K线数据: {formatted_code}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 数据类型转换
            numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            cache_manager.set(cache_key, df)
            logger.info(f"获取K线数据成功: {formatted_code}, 共{len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取K线数据失败: {formatted_code}, 错误: {e}")
            raise
    
    @api_retry(max_retries=3, delay=1.0)
    @safe_data_processing(default_value=pd.DataFrame())
    def get_financial_data(self, stock_code: str, year: int, quarter: int, 
                          data_type: str = "profit") -> pd.DataFrame:
        """获取财务数据"""
        # 格式化股票代码
        formatted_code = format_stock_code(stock_code)
        
        cache_key = f"financial_{data_type}_{formatted_code}_{year}_{quarter}"
        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        self._ensure_baostock_login()
        
        try:
            # 根据数据类型选择查询函数
            query_functions = {
                "profit": bs.query_profit_data,
                "operation": bs.query_operation_data,
                "growth": bs.query_growth_data,
                "balance": bs.query_balance_data,
                "cash_flow": bs.query_cash_flow_data,
                "dupont": bs.query_dupont_data
            }
            
            if data_type not in query_functions:
                raise DataError(f"不支持的财务数据类型: {data_type}")
            
            query_func = query_functions[data_type]
            rs = query_func(code=formatted_code, year=year, quarter=quarter)
            
            if rs.error_code != '0':
                raise APIError(f"获取财务数据失败: {rs.error_msg}", "baostock")
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                logger.warning(f"未获取到财务数据: {formatted_code}, {year}Q{quarter}, {data_type}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            cache_manager.set(cache_key, df)
            logger.info(f"获取财务数据成功: {formatted_code}, {data_type}, 共{len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取财务数据失败: {formatted_code}, 错误: {e}")
            raise
    
    @api_retry(max_retries=3, delay=1.0)
    @safe_data_processing(default_value=[])
    def get_industry_stocks(self, industry: str) -> List[str]:
        """获取行业股票列表"""
        cache_key = f"industry_stocks_{industry}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        self._ensure_baostock_login()
        
        try:
            rs = bs.query_stock_industry()
            if rs.error_code != '0':
                raise APIError(f"获取行业分类失败: {rs.error_msg}", "baostock")
            
            industry_stocks = []
            while (rs.error_code == '0') & rs.next():
                row = rs.get_row_data()
                if len(row) >= 3 and industry.lower() in row[2].lower():
                    industry_stocks.append(row[0])  # 股票代码
            
            cache_manager.set(cache_key, industry_stocks)
            logger.info(f"获取行业股票列表成功: {industry}, 共{len(industry_stocks)}只股票")
            return industry_stocks
            
        except Exception as e:
            logger.error(f"获取行业股票列表失败: {industry}, 错误: {e}")
            raise
    
    @api_retry(max_retries=2, delay=0.5)
    def get_real_time_data(self, stock_code: str) -> Dict[str, Any]:
        """获取实时数据（模拟）"""
        # 格式化股票代码
        formatted_code = format_stock_code(stock_code)
        
        cache_key = f"realtime_{formatted_code}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 获取最近的K线数据作为实时数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            
            df = self.get_stock_k_data(formatted_code, start_date, end_date)
            if df.empty:
                raise DataError(f"无法获取实时数据: {formatted_code}")
            
            latest_data = df.iloc[-1].to_dict()
            
            # 设置较短的缓存时间（5分钟）
            cache_manager.set(cache_key, latest_data)
            logger.info(f"获取实时数据成功: {formatted_code}")
            return latest_data
            
        except Exception as e:
            logger.error(f"获取实时数据失败: {formatted_code}, 错误: {e}")
            raise
    
    def get_market_data(self, market_type: str = "all") -> Dict[str, Any]:
        """获取市场数据"""
        cache_key = f"market_data_{market_type}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            market_data = {
                "timestamp": datetime.now().isoformat(),
                "market_type": market_type,
                "indices": {}
            }
            
            # 获取主要指数数据
            indices = {
                "上证指数": "sh.000001",
                "深证成指": "sz.399001",
                "创业板指": "sz.399006"
            }
            
            for name, code in indices.items():
                try:
                    index_data = self.get_real_time_data(code)
                    market_data["indices"][name] = index_data
                except Exception as e:
                    logger.warning(f"获取指数数据失败: {name}, 错误: {e}")
            
            cache_manager.set(cache_key, market_data)
            logger.info(f"获取市场数据成功: {market_type}")
            return market_data
            
        except Exception as e:
            logger.error(f"获取市场数据失败: {market_type}, 错误: {e}")
            raise
    
    def get_all_stocks(self) -> Optional[pd.DataFrame]:
        """获取所有股票列表"""
        try:
            self._ensure_baostock_login()
            rs = bs.query_all_stock()
            if rs.error_code != '0':
                logger.error(f"获取股票列表失败: {rs.error_msg}")
                return None
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"获取股票列表成功，共{len(df)}只股票")
                return df
            else:
                logger.warning("未获取到股票数据")
                return None
                
        except Exception as e:
            logger.error(f"获取股票列表异常: {e}")
            return None
    
    def get_trade_dates(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取交易日期"""
        try:
            self._ensure_baostock_login()
            rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
            if rs.error_code != '0':
                logger.error(f"获取交易日期失败: {rs.error_msg}")
                return None
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"获取交易日期成功，共{len(df)}个交易日")
                return df
            else:
                logger.warning("未获取到交易日期数据")
                return None
                
        except Exception as e:
            logger.error(f"获取交易日期异常: {e}")
            return None

    def close(self) -> None:
        """关闭数据获取器"""
        if self._baostock_logged_in:
            bs.logout()
            self._baostock_logged_in = False
            logger.info("BaoStock登出成功")
        
        self.session.close()
        logger.info("数据获取器已关闭")