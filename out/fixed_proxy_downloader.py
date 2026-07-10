#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复代理问题的下载器
确保akshare正确使用代理设置
"""

import akshare as ak
import pandas as pd
import numpy as np
import os
import pickle
import json
import time
import random
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

class FixedProxyDownloader:
    """
    修复代理问题的下载器
    确保网络请求正确使用代理
    """
    
    def __init__(self, data_dir="market_data"):
        """
        初始化下载器
        
        参数:
        data_dir: str, 数据保存目录
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 设置代理环境变量
        self._setup_proxy()
        
        # 备用股票池
        self.backup_stocks = [
            '000001', '000002', '000063', '000069', '000100', '000157', '000166', '000333',
            '000338', '000402', '000413', '000415', '000423', '000425', '000503', '000538',
            '000540', '000559', '000568', '000623', '000625', '000627', '000630', '000651',
            '000661', '000671', '000709', '000725', '000728', '000729', '000738', '000750',
            '000768', '000776', '000783', '000792', '000826', '000839', '000858', '000876',
            '000895', '000898', '000938', '000959', '000961', '000963', '000977', '000983',
            '600000', '600036', '600519', '600887', '601318', '601398', '601857', '601988',
            '002415', '002594', '002714', '300014', '300015', '300033', '300059', '300122'
        ]
    
    def _setup_proxy(self):
        """
        设置代理环境变量和requests会话
        从.env文件中读取代理配置
        """
        # 加载.env文件
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path, override=True)  # override=True确保.env文件中的值覆盖现有环境变量
        
        # 从环境变量中获取代理URL，如果没有则使用默认值
        proxy_url = os.getenv('HTTP_PROXY', 'http://100.87.200.30:7890')
        
        # 设置环境变量
        os.environ['http_proxy'] = proxy_url
        os.environ['https_proxy'] = proxy_url
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        
        # 设置requests默认代理
        self.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        
        # 创建带代理的session
        self.session = requests.Session()
        self.session.proxies.update(self.proxies)
        
        print(f"✅ 代理已设置: {proxy_url}")
        if env_path and os.path.exists(env_path):
            print(f"✅ 从.env文件加载配置: {env_path}")
        else:
            print("⚠️ 未找到.env文件，使用默认代理配置")
    
    def test_network_connection(self):
        """
        测试网络连接
        
        返回:
        bool: 连接是否正常
        """
        try:
            print("🔍 测试网络连接...")
            
            # 测试基本连接
            response = self.session.get('http://www.baidu.com', timeout=10)
            if response.status_code == 200:
                print("✅ 基本网络连接正常")
            else:
                print(f"⚠️ 网络连接异常，状态码: {response.status_code}")
            
            # 测试akshare相关的API
            try:
                # 简单测试akshare功能
                df = ak.stock_info_a_code_name()
                if df is not None and len(df) > 0:
                    print("✅ AKShare API连接正常")
                    return True
                else:
                    print("⚠️ AKShare API返回空数据")
                    return False
            except Exception as e:
                print(f"❌ AKShare API连接失败: {e}")
                return False
                
        except Exception as e:
            print(f"❌ 网络连接测试失败: {e}")
            return False
    
    def get_stock_list_robust(self, target_count=None):
        """
        稳健的股票列表获取方法
        
        参数:
        target_count: int, 目标股票数量
        
        返回:
        list: 股票代码列表
        """
        print("\n=== 获取股票列表 ===")
        
        # 方法1: 尝试获取股票基本信息
        try:
            print("尝试方法1: 获取股票基本信息...")
            time.sleep(2)  # 增加延时
            
            df = ak.stock_info_a_code_name()
            if df is not None and len(df) > 0:
                stock_codes = [str(code).zfill(6) for code in df['code'].tolist()]
                # 过滤有效股票
                valid_codes = []
                for i, code in enumerate(stock_codes):
                    if len(code) == 6 and code.isdigit():
                        if i < len(df):
                            name = df.iloc[i]['name'] if 'name' in df.columns else ''
                            if not any(x in name for x in ['ST', '退', 'N ', '*ST']):
                                valid_codes.append(code)
                
                if len(valid_codes) > 100:
                    print(f"✅ 方法1成功，获取到 {len(valid_codes)} 只股票")
                    if target_count:
                        return valid_codes[:target_count]
                    return valid_codes
        except Exception as e:
            print(f"方法1失败: {e}")
        
        # 方法2: 使用备用股票池
        print("使用备用股票池...")
        if target_count and target_count <= len(self.backup_stocks):
            selected_stocks = self.backup_stocks[:target_count]
        else:
            selected_stocks = self.backup_stocks
        
        print(f"✅ 使用备用股票池，共 {len(selected_stocks)} 只股票")
        return selected_stocks
    
    def download_single_stock(self, stock_code, days=300, max_retries=3):
        """
        下载单只股票数据
        
        参数:
        stock_code: str, 股票代码
        days: int, 历史天数
        max_retries: int, 最大重试次数
        
        返回:
        dict: 下载结果
        """
        file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
        
        for attempt in range(max_retries):
            try:
                # 计算日期范围
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # 添加随机延时避免频率限制
                time.sleep(random.uniform(1, 3))
                
                # 获取股票数据
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"  # 前复权
                )
                
                if df is None or len(df) == 0:
                    return {
                        'success': False,
                        'error': '数据为空',
                        'stock_code': stock_code
                    }
                
                # 数据预处理
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.sort_values('日期')
                df.set_index('日期', inplace=True)
                
                # 列名标准化
                column_mapping = {
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }
                df = df.rename(columns=column_mapping)
                
                # 保存数据
                df.to_csv(file_path)
                
                return {
                    'success': True,
                    'stock_code': stock_code,
                    'rows': len(df),
                    'file_path': file_path
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'success': False,
                        'error': str(e),
                        'stock_code': stock_code
                    }
    
    def batch_download(self, target_count=100, days=300, batch_size=20):
        """
        批量下载股票数据
        
        参数:
        target_count: int, 目标股票数量
        days: int, 历史天数
        batch_size: int, 批次大小
        
        返回:
        dict: 下载结果统计
        """
        print(f"\n=== 开始批量下载 ===")
        print(f"目标股票数: {target_count}")
        print(f"历史天数: {days}")
        print(f"批次大小: {batch_size}")
        
        # 测试网络连接
        if not self.test_network_connection():
            print("❌ 网络连接异常，建议检查代理设置")
            return {'success': False, 'error': '网络连接失败'}
        
        # 获取股票列表
        stock_list = self.get_stock_list_robust(target_count)
        if not stock_list:
            return {'success': False, 'error': '无法获取股票列表'}
        
        print(f"\n开始下载 {len(stock_list)} 只股票...")
        
        # 统计变量
        success_count = 0
        failed_count = 0
        failed_stocks = []
        
        # 分批下载
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(stock_list) + batch_size - 1) // batch_size
            
            print(f"\n--- 批次 {batch_num}/{total_batches} ({len(batch)} 只股票) ---")
            
            # 使用进度条
            with tqdm(batch, desc=f"批次{batch_num}") as pbar:
                for stock_code in pbar:
                    result = self.download_single_stock(stock_code, days)
                    
                    if result['success']:
                        success_count += 1
                        pbar.set_postfix({'成功': success_count, '失败': failed_count})
                    else:
                        failed_count += 1
                        failed_stocks.append({
                            'stock_code': stock_code,
                            'error': result.get('error', '未知错误')
                        })
                        pbar.set_postfix({'成功': success_count, '失败': failed_count})
            
            # 批次间休息
            if i + batch_size < len(stock_list):
                print(f"批次完成，休息5秒...")
                time.sleep(5)
        
        # 返回结果
        result = {
            'success': True,
            'total_stocks': len(stock_list),
            'success_count': success_count,
            'failed_count': failed_count,
            'success_rate': success_count / len(stock_list) * 100,
            'failed_stocks': failed_stocks
        }
        
        print(f"\n=== 下载完成 ===")
        print(f"总股票数: {result['total_stocks']}")
        print(f"成功: {result['success_count']}")
        print(f"失败: {result['failed_count']}")
        print(f"成功率: {result['success_rate']:.1f}%")
        
        if failed_stocks:
            print(f"\n失败的股票:")
            for item in failed_stocks[:5]:  # 只显示前5个
                print(f"  {item['stock_code']}: {item['error']}")
            if len(failed_stocks) > 5:
                print(f"  ... 还有 {len(failed_stocks) - 5} 个失败")
        
        return result

def main():
    """
    主函数 - 测试修复后的下载器
    """
    print("=== 修复代理问题的下载器测试 ===")
    
    # 创建下载器
    downloader = FixedProxyDownloader()
    
    print("\n请选择测试模式:")
    print("1. 网络连接测试")
    print("2. 小规模下载测试 (10只股票)")
    print("3. 中等规模下载 (50只股票)")
    print("4. 大规模下载 (100只股票)")
    
    try:
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            # 网络连接测试
            downloader.test_network_connection()
            
        elif choice == '2':
            # 小规模测试
            result = downloader.batch_download(target_count=10, days=150, batch_size=5)
            
        elif choice == '3':
            # 中等规模
            result = downloader.batch_download(target_count=50, days=200, batch_size=10)
            
        elif choice == '4':
            # 大规模
            result = downloader.batch_download(target_count=100, days=300, batch_size=20)
            
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"程序错误: {e}")

if __name__ == "__main__":
    main()