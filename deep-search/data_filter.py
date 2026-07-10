#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据过滤工具
用于过滤不合理的未来年份数据，确保分析的时效性和合理性
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFilter:
    def __init__(self, max_year: int = None):
        """
        初始化数据过滤器
        
        Args:
            max_year: 允许的最大年份，默认为当前年份
        """
        if max_year is None:
            max_year = datetime.now().year
        self.max_year = max_year
        logger.info(f"数据过滤器初始化，最大允许年份: {max_year}")
    
    def filter_financial_data(self, financial_data: Dict) -> Dict:
        """
        过滤财务数据，移除超出最大年份的数据
        
        Args:
            financial_data: 原始财务数据
            
        Returns:
            过滤后的财务数据
        """
        filtered_data = financial_data.copy()
        
        # 过滤财务指标数据
        if 'financial_ratios' in filtered_data:
            ratios = filtered_data['financial_ratios']
            if 'report_date' in ratios:
                report_date = str(ratios['report_date'])
                year = int(report_date[:4])
                if year > self.max_year:
                    logger.warning(f"财务指标数据年份 {year} 超出限制 {self.max_year}，将被过滤")
                    # 这里可以选择清空数据或使用默认值
                    filtered_data['financial_ratios'] = self._get_default_ratios()
        
        # 过滤现金流数据
        if 'cashflow_data' in filtered_data:
            cashflow = filtered_data['cashflow_data']
            if 'notice_date' in cashflow:
                notice_date = str(cashflow['notice_date'])
                year = int(notice_date[:4])
                if year > self.max_year:
                    logger.warning(f"现金流数据年份 {year} 超出限制 {self.max_year}，将被过滤")
                    filtered_data['cashflow_data'] = self._get_default_cashflow()
        
        return filtered_data
    
    def filter_historical_data(self, historical_data: Dict) -> Dict:
        """
        过滤历史数据，只保留合理年份的数据
        
        Args:
            historical_data: 原始历史数据
            
        Returns:
            过滤后的历史数据
        """
        if not historical_data:
            return historical_data
        
        filtered_data = {
            'revenue_history': [],
            'profit_history': [],
            'roe_history': [],
            'report_dates': [],
            'years': []
        }
        
        # 获取原始数据
        report_dates = historical_data.get('report_dates', [])
        revenue_history = historical_data.get('revenue_history', [])
        profit_history = historical_data.get('profit_history', [])
        roe_history = historical_data.get('roe_history', [])
        
        # 过滤数据
        for i, date in enumerate(report_dates):
            try:
                year = int(str(date)[:4])
                if year <= self.max_year:
                    filtered_data['report_dates'].append(date)
                    if year not in [int(y) for y in filtered_data['years']]:
                        filtered_data['years'].append(str(year))
                    
                    # 添加对应的财务数据
                    if i < len(revenue_history):
                        filtered_data['revenue_history'].append(revenue_history[i])
                    if i < len(profit_history):
                        filtered_data['profit_history'].append(profit_history[i])
                    if i < len(roe_history):
                        filtered_data['roe_history'].append(roe_history[i])
                else:
                    logger.info(f"过滤掉年份 {year} 的数据（报告期: {date}）")
            except (ValueError, IndexError) as e:
                logger.warning(f"解析日期失败: {date}, 错误: {e}")
                continue
        
        # 按年份排序（最新的在前）
        filtered_data['years'].sort(reverse=True)
        
        logger.info(f"历史数据过滤完成，保留 {len(filtered_data['report_dates'])} 期数据，涉及年份: {filtered_data['years']}")
        return filtered_data
    
    def get_latest_quarterly_data(self, historical_data: Dict) -> Optional[Dict]:
        """
        获取最新的季报数据（包括一季报、半年报、三季报、年报）
        
        Args:
            historical_data: 历史数据
            
        Returns:
            最新季报数据信息
        """
        if not historical_data or not historical_data.get('report_dates'):
            return None
        
        report_dates = historical_data.get('report_dates', [])
        revenue_history = historical_data.get('revenue_history', [])
        profit_history = historical_data.get('profit_history', [])
        
        # 找到最新的报告期
        latest_date = None
        latest_index = -1
        latest_year = 0
        
        for i, date in enumerate(report_dates):
            try:
                date_str = str(date)
                year = int(date_str[:4])
                
                if year <= self.max_year:
                    if year > latest_year or (year == latest_year and date > latest_date):
                        latest_date = date
                        latest_index = i
                        latest_year = year
            except (ValueError, IndexError):
                continue
        
        if latest_index >= 0:
            # 判断报告期类型
            date_str = str(latest_date)
            if date_str.endswith('03-31'):
                period_type = "一季报"
            elif date_str.endswith('06-30'):
                period_type = "半年报"
            elif date_str.endswith('09-30'):
                period_type = "三季报"
            elif date_str.endswith('12-31'):
                period_type = "年报"
            else:
                period_type = "其他"
            
            return {
                'year': latest_year,
                'date': latest_date,
                'period_type': period_type,
                'revenue': revenue_history[latest_index] if latest_index < len(revenue_history) else 0,
                'profit': profit_history[latest_index] if latest_index < len(profit_history) else 0,
                'index': latest_index
            }
        
        return None
    
    def get_latest_annual_data(self, historical_data: Dict) -> Optional[Dict]:
        """
        获取最新的年报数据（12-31结尾的数据）
        
        Args:
            historical_data: 历史数据
            
        Returns:
            最新年报数据或None
        """
        if not historical_data or not historical_data.get('report_dates'):
            return None
        
        report_dates = historical_data['report_dates']
        revenue_history = historical_data.get('revenue_history', [])
        profit_history = historical_data.get('profit_history', [])
        roe_history = historical_data.get('roe_history', [])
        
        # 查找最新的年报数据
        for i, date in enumerate(report_dates):
            date_str = str(date)
            year = int(date_str[:4])
            
            # 检查是否为年报数据（12-31）且年份合理
            if year <= self.max_year and date_str.endswith('12-31 00:00:00'):
                latest_data = {
                    'report_date': date,
                    'year': year,
                    'revenue': revenue_history[i] if i < len(revenue_history) else None,
                    'profit': profit_history[i] if i < len(profit_history) else None,
                    'roe': roe_history[i] if i < len(roe_history) else None
                }
                logger.info(f"找到最新年报数据: {year}年年报")
                return latest_data
        
        logger.warning("未找到合适的年报数据")
        return None
    
    def _get_default_ratios(self) -> Dict:
        """获取默认的财务指标数据"""
        return {
            'basic_eps': 0,
            'bps': 0,
            'operating_cashflow_per_share': 0,
            'total_operate_income': 0,
            'parent_netprofit': 0,
            'report_date': f'{self.max_year}-12-31',
            'security_name': '数据已过滤'
        }
    
    def _get_default_cashflow(self) -> Dict:
        """获取默认的现金流数据"""
        return {
            'operating_cashflow': 0,
            'investing_cashflow': 0,
            'financing_cashflow': 0,
            'notice_date': f'{self.max_year}-12-31'
        }

# 创建全局数据过滤器实例（使用当前年份）
data_filter = DataFilter()

def filter_financial_data(financial_data: Dict) -> Dict:
    """便捷函数：过滤财务数据"""
    return data_filter.filter_financial_data(financial_data)

def filter_historical_data(historical_data: Dict) -> Dict:
    """便捷函数：过滤历史数据"""
    return data_filter.filter_historical_data(historical_data)

def get_latest_annual_data(historical_data: Dict) -> Optional[Dict]:
    """获取最新年报数据的全局函数"""
    return data_filter.get_latest_annual_data(historical_data)

def get_latest_quarterly_data(historical_data: Dict) -> Optional[Dict]:
    """获取最新季报数据的全局函数"""
    return data_filter.get_latest_quarterly_data(historical_data)