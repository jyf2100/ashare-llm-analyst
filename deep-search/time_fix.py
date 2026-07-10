#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间修正工具
用于修正系统中的时间戳问题
"""

from datetime import datetime, timedelta, date
import time

class TimeFixer:
    """时间修正器"""
    
    def __init__(self):
        # 动态获取当前年份
        current_date = date.today()
        self.base_year = current_date.year
        # 使用当前日期，但保持固定的时间（19:00:00）以确保一致性
        self.current_real_time = datetime(current_date.year, current_date.month, current_date.day, 19, 0, 0)
    
    def get_corrected_time(self) -> datetime:
        """获取修正后的当前时间"""
        return self.current_real_time
    
    def get_corrected_timestamp(self) -> str:
        """获取修正后的时间戳字符串"""
        return self.current_real_time.strftime('%Y-%m-%d %H:%M:%S')
    
    def get_corrected_iso_timestamp(self) -> str:
        """获取修正后的ISO格式时间戳"""
        return self.current_real_time.isoformat()
    
    def get_corrected_filename_timestamp(self) -> str:
        """获取修正后的文件名时间戳"""
        return self.current_real_time.strftime('%Y%m%d_%H%M%S')
    
    def get_corrected_report_date(self) -> str:
        """获取修正后的报告日期"""
        return self.current_real_time.strftime('%Y年%m月%d日')
    
    def correct_financial_data_year(self, year_str: str) -> str:
        """修正财务数据中的年份"""
        # 动态判断：只修正明显不合理的未来年份（超过当前年份）
        try:
            year = int(year_str)
            if year > self.base_year:
                return str(self.base_year)
        except (ValueError, TypeError):
            pass
        return year_str
    
    def correct_report_date(self, report_date: str) -> str:
        """修正报告期日期"""
        # 动态判断：只修正明显不合理的未来年份（超过当前年份）
        if report_date:
            for year in range(self.base_year + 1, self.base_year + 5):  # 修正未来年份
                if str(year) in report_date:
                    return report_date.replace(str(year), str(self.base_year))
        return report_date

# 全局时间修正器实例
time_fixer = TimeFixer()

def get_current_time():
    """获取修正后的当前时间"""
    return time_fixer.get_corrected_time()

def get_current_timestamp():
    """获取修正后的当前时间戳"""
    return time_fixer.get_corrected_timestamp()

def get_current_iso_timestamp():
    """获取修正后的ISO时间戳"""
    return time_fixer.get_corrected_iso_timestamp()

def get_filename_timestamp():
    """获取修正后的文件名时间戳"""
    return time_fixer.get_corrected_filename_timestamp()

def get_report_date():
    """获取修正后的报告日期"""
    return time_fixer.get_corrected_report_date()

if __name__ == "__main__":
    print("时间修正工具测试:")
    print(f"修正后的当前时间: {get_current_timestamp()}")
    print(f"修正后的ISO时间戳: {get_current_iso_timestamp()}")
    print(f"修正后的文件名时间戳: {get_filename_timestamp()}")
    print(f"修正后的报告日期: {get_report_date()}")