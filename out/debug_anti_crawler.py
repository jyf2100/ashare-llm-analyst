#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试反爬虫机制的测试脚本
用于分析连续失败的原因
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import logging

# 导入增量下载器
from importlib import import_module
downloader_module = import_module('01-incremental_stock_data_downloader')
IncrementalStockDataDownloader = downloader_module.IncrementalStockDataDownloader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_anti_crawler_mechanism():
    """
    测试反爬虫机制，分析可能的失败原因
    """
    print("=== 反爬虫机制调试测试 ===")
    
    # 创建下载器实例
    downloader = IncrementalStockDataDownloader()
    
    # 测试场景1：正常请求
    print("\n1. 测试正常请求场景")
    try:
        # 建立连接
        if downloader.ensure_baostock_connection():
            print("✓ Baostock连接成功")
            
            # 测试下载一个存在的股票数据
            test_code = "sh.600000"
            start_date = "2024-01-01"
            end_date = "2024-01-05"
            
            print(f"测试下载股票 {test_code} 数据: {start_date} 到 {end_date}")
            data = downloader.download_stock_data_range(test_code, start_date, end_date)
            
            if data is not None:
                print(f"✓ 成功获取数据，共 {len(data)} 条记录")
            else:
                print("✗ 获取数据失败")
        else:
            print("✗ Baostock连接失败")
            
    except Exception as e:
        print(f"✗ 测试异常: {str(e)}")
    
    # 测试场景2：无效股票代码
    print("\n2. 测试无效股票代码场景")
    try:
        invalid_codes = ["sh.999999", "sz.999999", "invalid.code"]
        for code in invalid_codes:
            print(f"测试无效代码: {code}")
            data = downloader.download_stock_data_range(code, "2024-01-01", "2024-01-05")
            if data is None:
                print(f"  - {code}: 无数据返回（预期结果）")
            else:
                print(f"  - {code}: 意外获取到数据")
                
    except Exception as e:
        print(f"✗ 无效代码测试异常: {str(e)}")
    
    # 测试场景3：无效日期范围
    print("\n3. 测试无效日期范围场景")
    try:
        test_code = "sh.600000"
        invalid_dates = [
            ("2030-01-01", "2030-01-05"),  # 未来日期
            ("1990-01-01", "1990-01-05"),  # 过早日期
            ("2024-01-05", "2024-01-01"),  # 结束日期早于开始日期
        ]
        
        for start, end in invalid_dates:
            print(f"测试日期范围: {start} 到 {end}")
            data = downloader.download_stock_data_range(test_code, start, end)
            if data is None:
                print(f"  - 无数据返回")
            else:
                print(f"  - 获取到 {len(data)} 条数据")
                
    except Exception as e:
        print(f"✗ 无效日期测试异常: {str(e)}")
    
    # 显示反爬虫统计
    print("\n=== 反爬虫统计信息 ===")
    print(f"总请求数: {downloader.total_requests}")
    print(f"成功请求数: {downloader.successful_requests}")
    print(f"失败请求数: {downloader.failed_requests}")
    if downloader.total_requests > 0:
        success_rate = (downloader.successful_requests / downloader.total_requests) * 100
        print(f"成功率: {success_rate:.2f}%")
    
    print(f"连续失败次数: {downloader.anti_crawler.consecutive_failures}")
    print(f"是否需要暂停: {downloader.anti_crawler.should_pause()}")
    
    # 清理连接
    try:
        if hasattr(downloader, '_baostock_connected') and downloader._baostock_connected:
            downloader.logout_baostock()
            print("\n✓ 已清理Baostock连接")
    except Exception as e:
        print(f"清理连接时出错: {str(e)}")

def analyze_failure_scenarios():
    """
    分析可能导致连续失败的场景
    """
    print("\n=== 连续失败场景分析 ===")
    
    scenarios = [
        "1. 网络连接问题 - 无法访问Baostock服务器",
        "2. API限制 - 请求频率过高被限制",
        "3. 数据源问题 - Baostock服务暂时不可用",
        "4. 股票代码问题 - 大量无效或已退市股票代码",
        "5. 日期范围问题 - 请求的日期范围无数据",
        "6. 系统资源问题 - 内存不足或其他系统问题",
        "7. 反爬虫机制过于敏感 - 正常请求被误判为爬虫"
    ]
    
    for scenario in scenarios:
        print(scenario)
    
    print("\n建议的排查步骤:")
    steps = [
        "1. 检查网络连接和DNS解析",
        "2. 验证Baostock服务状态",
        "3. 检查请求的股票代码是否有效",
        "4. 验证请求的日期范围是否合理",
        "5. 监控系统资源使用情况",
        "6. 调整反爬虫参数（延时、重试次数等）",
        "7. 查看详细的错误日志和响应信息"
    ]
    
    for step in steps:
        print(step)

if __name__ == "__main__":
    test_anti_crawler_mechanism()
    analyze_failure_scenarios()