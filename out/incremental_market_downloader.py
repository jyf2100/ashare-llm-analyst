#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量市场数据下载器
支持智能增量下载，避免重复下载已有数据
"""

import akshare as ak
import pandas as pd
import numpy as np
import os
import pickle
import json
import time
import random
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# 加载环境变量
load_dotenv(override=True)

class IncrementalMarketDownloader:
    """
    增量市场数据下载器类
    支持智能增量下载功能
    """
    
    def __init__(self, data_dir="market_data", proxy_config=None):
        """
        初始化增量下载器
        
        参数:
        data_dir: str, 数据保存目录
        proxy_config: dict, 代理配置
        """
        self.data_dir = data_dir
        self.proxy_config = proxy_config
        self.backup_stock_pools = self._get_backup_stock_pools()
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 增量下载配置
        self.min_file_size = 1000  # 最小文件大小（字节）
        self.min_data_rows = 50    # 最小数据行数
        self.max_data_age_days = 7 # 数据最大过期天数
    
    def _get_backup_stock_pools(self):
        """
        获取扩展的备用股票池
        
        返回:
        dict: 包含不同类型的股票池
        """
        return {
            # 主板蓝筹股（稳定性好）
            'main_board': [
                '000001', '000002', '000858', '000895', '000938', '002415', '002594', '002714',
                '600000', '600036', '600519', '600887', '601318', '601398', '601857', '601988',
                '000001', '000002', '000063', '000069', '000100', '000157', '000166', '000333',
                '000338', '000402', '000413', '000415', '000423', '000425', '000503', '000538',
                '000540', '000559', '000568', '000623', '000625', '000627', '000630', '000651',
                '000661', '000671', '000709', '000725', '000728', '000729', '000738', '000750',
                '000768', '000776', '000783', '000792', '000826', '000839', '000858', '000876',
                '000895', '000898', '000938', '000959', '000961', '000963', '000977', '000983'
            ],
            
            # 科技股
            'tech_stocks': [
                '000725', '002415', '002456', '002475', '300014', '300015', '300033', '300059',
                '300122', '300124', '300136', '300142', '300144', '300251', '300274', '300296',
                '300316', '300347', '300408', '300413', '300433', '300454', '300496', '300498'
            ],
            
            # 消费股
            'consumer_stocks': [
                '000568', '000596', '000858', '000895', '000930', '002304', '002507', '002568',
                '002714', '600519', '600887', '600999', '603288', '603369', '603589', '603899'
            ]
        }
    
    def check_file_status(self, stock_code):
        """
        检查股票数据文件状态
        
        参数:
        stock_code: str, 股票代码
        
        返回:
        dict: 文件状态信息
        """
        file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
        
        status = {
            'exists': False,
            'valid_size': False,
            'valid_data': False,
            'up_to_date': False,
            'needs_download': True,
            'file_path': file_path,
            'file_size': 0,
            'data_rows': 0,
            'last_date': None,
            'age_days': None
        }
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return status
        
        status['exists'] = True
        status['file_size'] = os.path.getsize(file_path)
        
        # 检查文件大小
        if status['file_size'] < self.min_file_size:
            return status
        
        status['valid_size'] = True
        
        try:
            # 读取数据检查内容
            df = pd.read_csv(file_path)
            status['data_rows'] = len(df)
            
            # 检查数据行数
            if status['data_rows'] < self.min_data_rows:
                return status
            
            status['valid_data'] = True
            
            # 检查数据时效性
            if 'date' in df.columns and len(df) > 0:
                # 获取最新数据日期
                df['date'] = pd.to_datetime(df['date'])
                last_date = df['date'].max()
                status['last_date'] = last_date.strftime('%Y-%m-%d')
                
                # 计算数据年龄
                today = datetime.now()
                age_days = (today - last_date).days
                status['age_days'] = age_days
                
                # 检查是否需要更新（考虑周末）
                if age_days <= self.max_data_age_days:
                    status['up_to_date'] = True
                    status['needs_download'] = False
                    
        except Exception as e:
            # 文件损坏，需要重新下载
            pass
        
        return status
    
    def get_download_list(self, stock_list, force_update=False):
        """
        获取需要下载的股票列表
        
        参数:
        stock_list: list, 股票代码列表
        force_update: bool, 是否强制更新所有文件
        
        返回:
        dict: 下载列表分类
        """
        download_lists = {
            'new_files': [],      # 新文件
            'update_files': [],   # 需要更新的文件
            'skip_files': [],     # 跳过的文件
            'invalid_files': []   # 无效文件
        }
        
        print("\n=== 检查现有数据文件状态 ===")
        
        for stock_code in tqdm(stock_list, desc="检查文件状态"):
            status = self.check_file_status(stock_code)
            
            if force_update:
                if status['exists']:
                    download_lists['update_files'].append(stock_code)
                else:
                    download_lists['new_files'].append(stock_code)
            else:
                if not status['exists']:
                    download_lists['new_files'].append(stock_code)
                elif not status['valid_size'] or not status['valid_data']:
                    download_lists['invalid_files'].append(stock_code)
                elif not status['up_to_date']:
                    download_lists['update_files'].append(stock_code)
                else:
                    download_lists['skip_files'].append(stock_code)
        
        # 打印统计信息
        print(f"\n文件状态统计:")
        print(f"  新文件: {len(download_lists['new_files'])}")
        print(f"  需更新: {len(download_lists['update_files'])}")
        print(f"  无效文件: {len(download_lists['invalid_files'])}")
        print(f"  跳过文件: {len(download_lists['skip_files'])}")
        
        return download_lists
    
    def get_stock_list_with_limit(self, target_count=None):
        """
        获取指定数量的股票列表
        
        参数:
        target_count: int, 目标股票数量，None表示获取所有
        
        返回:
        list: 股票代码列表
        """
        print("\n=== 获取股票列表 ===")
        
        # 尝试多种方法获取股票列表
        stock_list = []
        
        # 方法1: 尝试获取A股实时行情
        try:
            print("尝试方法1: 获取A股实时行情...")
            df = ak.stock_zh_a_spot_em()
            if df is not None and len(df) > 0:
                stock_list = df['代码'].tolist()
                print(f"✓ 通过实时行情获取到 {len(stock_list)} 只股票")
        except Exception as e:
            print(f"✗ 实时行情获取失败: {str(e)}")
        
        # 方法2: 如果方法1失败，尝试获取股票基本信息
        if not stock_list:
            try:
                print("尝试方法2: 获取股票基本信息...")
                df = ak.stock_info_a_code_name()
                if df is not None and len(df) > 0:
                    stock_list = df['code'].tolist()
                    print(f"✓ 通过基本信息获取到 {len(stock_list)} 只股票")
            except Exception as e:
                print(f"✗ 基本信息获取失败: {str(e)}")
        
        # 方法3: 使用备用股票池
        if not stock_list:
            print("使用备用股票池...")
            all_backup = []
            for pool_name, stocks in self.backup_stock_pools.items():
                all_backup.extend(stocks)
            stock_list = list(set(all_backup))  # 去重
            print(f"✓ 备用股票池包含 {len(stock_list)} 只股票")
        
        # 限制数量
        if target_count is not None and len(stock_list) > target_count:
            stock_list = stock_list[:target_count]
            print(f"✓ 限制为前 {target_count} 只股票")
        
        return stock_list
    
    def download_stock_data_with_retry(self, stock_code, days=300, max_retries=3):
        """
        带重试机制的股票数据下载
        
        参数:
        stock_code: str, 股票代码
        days: int, 获取天数
        max_retries: int, 最大重试次数
        
        返回:
        bool: 下载是否成功
        """
        file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
        
        for attempt in range(max_retries):
            try:
                # 计算开始日期
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
                
                # 下载数据
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                
                if df is not None and len(df) > 0:
                    # 数据预处理
                    df = self._preprocess_stock_data(df, stock_code)
                    
                    # 保存数据
                    df.to_csv(file_path, index=False, encoding='utf-8')
                    return True
                else:
                    if attempt < max_retries - 1:
                        time.sleep(random.uniform(1, 3))
                    continue
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(2, 5))
                    continue
                else:
                    return False
        
        return False
    
    def _preprocess_stock_data(self, df, stock_code):
        """
        数据预处理
        
        参数:
        df: DataFrame, 原始数据
        stock_code: str, 股票代码
        
        返回:
        DataFrame: 处理后的数据
        """
        # 重命名列
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_chg',
            '涨跌额': 'change',
            '换手率': 'turnover'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 添加股票代码列
        df['stock_code'] = stock_code
        
        # 确保日期格式正确
        df['date'] = pd.to_datetime(df['date'])
        
        # 按日期排序
        df = df.sort_values('date')
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df
    
    def incremental_batch_download(self, target_count=None, days=300, batch_size=50, force_update=False):
        """
        增量批量下载股票数据
        
        参数:
        target_count: int, 目标股票数量，None表示下载所有
        days: int, 获取天数
        batch_size: int, 批次大小
        force_update: bool, 是否强制更新所有文件
        
        返回:
        dict: 下载结果统计
        """
        if target_count is None:
            print("\n=== 开始增量下载全市场股票数据 ===")
        else:
            print(f"\n=== 开始增量下载 {target_count} 只股票数据 ===")
        
        # 获取股票列表
        stock_list = self.get_stock_list_with_limit(target_count)
        
        if not stock_list:
            print("❌ 无法获取股票列表")
            return None
        
        # 获取需要下载的文件列表
        download_lists = self.get_download_list(stock_list, force_update)
        
        # 合并需要下载的股票
        stocks_to_download = (
            download_lists['new_files'] + 
            download_lists['update_files'] + 
            download_lists['invalid_files']
        )
        
        total_stocks = len(stock_list)
        download_count = len(stocks_to_download)
        skip_count = len(download_lists['skip_files'])
        
        print(f"\n下载计划:")
        print(f"  总股票数: {total_stocks}")
        print(f"  需下载: {download_count}")
        print(f"  跳过: {skip_count}")
        print(f"  数据天数: {days}")
        print(f"  批次大小: {batch_size}")
        
        if download_count == 0:
            print("\n✓ 所有数据都是最新的，无需下载")
            return {
                'total': total_stocks,
                'success': skip_count,
                'failed': 0,
                'skipped': skip_count,
                'success_rate': 100.0,
                'failed_stocks': []
            }
        
        success_count = 0
        failed_stocks = []
        
        # 分批下载
        for i in range(0, download_count, batch_size):
            batch = stocks_to_download[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (download_count + batch_size - 1) // batch_size
            
            print(f"\n--- 批次 {batch_num}/{total_batches} ({len(batch)} 只股票) ---")
            
            # 下载当前批次
            for stock_code in tqdm(batch, desc=f"批次{batch_num}"):
                if self.download_stock_data_with_retry(stock_code, days):
                    success_count += 1
                else:
                    failed_stocks.append(stock_code)
                
                # 随机延时，避免请求过于频繁
                time.sleep(random.uniform(0.1, 0.5))
            
            # 批次间延时
            if i + batch_size < download_count:
                time.sleep(random.uniform(1, 3))
        
        # 统计结果
        total_success = success_count + skip_count
        success_rate = (total_success / total_stocks) * 100
        
        result = {
            'total': total_stocks,
            'success': total_success,
            'downloaded': success_count,
            'skipped': skip_count,
            'failed': len(failed_stocks),
            'success_rate': success_rate,
            'failed_stocks': failed_stocks
        }
        
        print(f"\n=== 增量下载完成 ===")
        print(f"总股票数: {total_stocks}")
        print(f"成功处理: {total_success} (下载: {success_count}, 跳过: {skip_count})")
        print(f"下载失败: {len(failed_stocks)}")
        print(f"成功率: {success_rate:.1f}%")
        
        if failed_stocks:
            print(f"\n失败的股票: {failed_stocks[:10]}{'...' if len(failed_stocks) > 10 else ''}")
        
        # 保存下载结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if target_count is None:
            result_file = os.path.join(self.data_dir, f'incremental_download_result_full_{timestamp}.json')
        else:
            result_file = os.path.join(self.data_dir, f'incremental_download_result_{target_count}_{timestamp}.json')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
    
    def _calculate_stock_rps(self, stock_code, all_data, valid_stocks, rps_period):
        """
        计算单只股票的RPS（多线程辅助函数）
        
        参数:
        stock_code: str, 股票代码
        all_data: dict, 所有股票数据
        valid_stocks: list, 有效股票列表
        rps_period: int, RPS计算周期
        
        返回:
        tuple: (stock_code, rps_values)
        """
        df = all_data[stock_code]
        rps_values = []
        
        for i in range(rps_period - 1, len(df)):
            current_date = df.iloc[i]['date']
            current_price = df.iloc[i]['close']
            past_price = df.iloc[i - rps_period + 1]['close']
            
            # 计算当前股票的涨幅
            current_return = (current_price - past_price) / past_price
            
            # 计算同期其他股票的涨幅
            other_returns = []
            for other_code in valid_stocks:
                if other_code == stock_code:
                    continue
                
                other_df = all_data[other_code]
                # 找到相同日期的数据
                date_mask = other_df['date'] <= current_date
                if date_mask.sum() >= rps_period:
                    other_current_idx = date_mask.sum() - 1
                    other_past_idx = other_current_idx - rps_period + 1
                    
                    if other_past_idx >= 0:
                        other_current_price = other_df.iloc[other_current_idx]['close']
                        other_past_price = other_df.iloc[other_past_idx]['close']
                        other_return = (other_current_price - other_past_price) / other_past_price
                        other_returns.append(other_return)
            
            # 计算RPS
            if len(other_returns) > 0:
                better_count = sum(1 for r in other_returns if current_return > r)
                rps = (better_count / len(other_returns)) * 100
            else:
                rps = 50  # 默认值
            
            rps_values.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'rps': round(rps, 2)
            })
        
        return stock_code, rps_values
    
    def calculate_rps(self, rps_period=50, max_workers=None):
        """
        计算RPS（相对价格强度）- 多线程版本
        
        参数:
        rps_period: int, RPS计算周期
        max_workers: int, 最大线程数（默认为CPU核心数）
        
        返回:
        dict: RPS数据
        """
        print(f"\n=== 开始计算 {rps_period} 日RPS（多线程版本）===")
        
        # 获取所有CSV文件
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if len(csv_files) == 0:
            print("❌ 没有找到股票数据文件")
            return None
        
        print(f"找到 {len(csv_files)} 个股票数据文件")
        
        # 读取所有股票数据
        all_data = {}
        valid_stocks = []
        
        for csv_file in tqdm(csv_files, desc="读取数据"):
            stock_code = csv_file.replace('.csv', '')
            file_path = os.path.join(self.data_dir, csv_file)
            
            try:
                df = pd.read_csv(file_path)
                if len(df) >= rps_period:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    all_data[stock_code] = df
                    valid_stocks.append(stock_code)
            except Exception as e:
                continue
        
        if len(valid_stocks) == 0:
            print("❌ 没有有效的股票数据")
            return None
        
        print(f"有效股票数据: {len(valid_stocks)} 只")
        
        # 设置线程数
        if max_workers is None:
            max_workers = min(20, (os.cpu_count() or 1) + 4)  # 默认线程数
        
        print(f"使用 {max_workers} 个线程进行并行计算")
        
        # 使用多线程计算每只股票的RPS
        rps_data = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(self._calculate_stock_rps, stock_code, all_data, valid_stocks, rps_period): stock_code
                for stock_code in valid_stocks
            }
            
            # 收集结果
            with tqdm(total=len(valid_stocks), desc="计算RPS") as pbar:
                for future in as_completed(future_to_stock):
                    try:
                        stock_code, rps_values = future.result()
                        rps_data[stock_code] = rps_values
                        pbar.update(1)
                    except Exception as e:
                        stock_code = future_to_stock[future]
                        print(f"\n⚠️ 计算股票 {stock_code} RPS时出错: {e}")
                        pbar.update(1)
        
        # 保存RPS数据
        rps_file = os.path.join(self.data_dir, f'rps_{rps_period}d_incremental.pkl')
        with open(rps_file, 'wb') as f:
            pickle.dump(rps_data, f)
        
        print(f"✓ RPS计算完成，数据已保存到 {rps_file}")
        print(f"✓ 覆盖 {len(rps_data)} 只股票")
        
        return rps_data

def main():
    """
    主函数 - 演示增量下载功能
    """
    print("=== 增量市场数据下载器演示 ===")
    
    # 创建下载器实例
    proxy_config = None
    if os.getenv('HTTP_PROXY') and os.getenv('HTTPS_PROXY'):
        proxy_config = {
            'http_proxy': os.getenv('HTTP_PROXY'),
            'https_proxy': os.getenv('HTTPS_PROXY')
        }
    
    downloader = IncrementalMarketDownloader(
        data_dir="market_data",
        proxy_config=proxy_config
    )
    
    # 演示增量下载
    print("\n选择下载模式:")
    print("1. 增量下载 100 只股票")
    print("2. 增量下载 500 只股票")
    print("3. 增量下载全市场")
    print("4. 强制更新所有文件")
    
    try:
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == '1':
            result = downloader.incremental_batch_download(target_count=100, days=300, batch_size=20)
        elif choice == '2':
            result = downloader.incremental_batch_download(target_count=500, days=300, batch_size=30)
        elif choice == '3':
            result = downloader.incremental_batch_download(target_count=None, days=300, batch_size=50)
        elif choice == '4':
            result = downloader.incremental_batch_download(target_count=100, days=300, batch_size=20, force_update=True)
        else:
            print("无效选择")
            return
        
        if result:
            print("\n=== 下载结果 ===")
            print(f"总股票数: {result['total']}")
            print(f"成功处理: {result['success']}")
            print(f"新下载: {result.get('downloaded', 0)}")
            print(f"跳过: {result.get('skipped', 0)}")
            print(f"失败: {result['failed']}")
            print(f"成功率: {result['success_rate']:.1f}%")
            
            # 如果下载成功，计算RPS
            if result['success'] >= 10:
                print("\n开始计算RPS...")
                rps_data = downloader.calculate_rps(rps_period=50)
                if rps_data:
                    print(f"✓ RPS计算完成，覆盖 {len(rps_data)} 只股票")
        
    except KeyboardInterrupt:
        print("\n下载被用户中断")
    except Exception as e:
        print(f"\n下载过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()