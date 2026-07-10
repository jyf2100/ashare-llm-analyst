#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的市场数据下载器
解决API限制问题，提供多种数据获取策略
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
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# 加载环境变量
load_dotenv(override=True)

class OptimizedMarketDownloader:
    """
    优化的市场数据下载器类
    提供多种数据获取策略和容错机制
    """
    
    def __init__(self, data_dir="market_data", proxy_config=None):
        """
        初始化下载器
        
        参数:
        data_dir: str, 数据保存目录
        proxy_config: dict, 代理配置
        """
        self.data_dir = data_dir
        self.proxy_config = proxy_config
        self.backup_stock_pools = self._get_backup_stock_pools()
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_backup_stock_pools(self):
        """
        获取多个备用股票池
        
        返回:
        dict: 包含不同类型的股票池
        """
        return {
            # 主板蓝筹股（稳定性好）
            "blue_chips": [
                '000001', '000002', '000858', '600000', '600036', '600519',
                '600887', '000063', '000069', '000100', '000157', '000166',
                '000333', '000338', '000402', '000425', '000503', '000538',
                '000540', '000559', '000568', '000596', '000625', '000629',
                '000630', '000651', '000661', '000671', '000709', '000725'
            ],
            
            # 科技成长股
            "tech_growth": [
                '000977', '002415', '002487', '300059', '300142', '300274',
                '300408', '300433', '300496', '300750', '688036', '688111',
                '688169', '688188', '688223', '688256', '688303', '688981',
                '300014', '300015', '300017', '300024', '300027', '300033',
                '300037', '300058', '300070', '300072', '300073', '300088'
            ],
            
            # 传统行业龙头
            "traditional": [
                '600028', '600030', '600048', '600050', '600104', '600111',
                '600150', '600170', '600196', '600276', '600309', '600362',
                '600383', '600406', '600436', '600438', '600482', '600489',
                '600498', '600516', '600547', '600570', '600585', '600588',
                '600606', '600637', '600660', '600674', '600690', '600703'
            ],
            
            # 中小板活跃股
            "mid_cap": [
                '002001', '002007', '002008', '002024', '002027', '002032',
                '002044', '002049', '002050', '002065', '002074', '002081',
                '002092', '002120', '002129', '002142', '002146', '002153',
                '002174', '002179', '002202', '002230', '002236', '002241',
                '002252', '002271', '002304', '002311', '002352', '002372'
            ],
            
            # 创业板代表股
            "growth_board": [
                '300001', '300002', '300003', '300009', '300012', '300013',
                '300016', '300018', '300019', '300020', '300021', '300022',
                '300025', '300026', '300028', '300029', '300030', '300031',
                '300032', '300034', '300035', '300036', '300038', '300039',
                '300040', '300041', '300042', '300043', '300044', '300045'
            ]
        }
    
    def get_stock_list_multiple_methods(self, target_count=500):
        """
        使用多种方法获取股票列表
        
        参数:
        target_count: int, 目标股票数量
        
        返回:
        list: 股票代码列表
        """
        print("=== 尝试多种方法获取股票列表 ===")
        
        # 方法1：尝试获取A股实时行情
        stock_list = self._try_get_spot_data()
        if stock_list and len(stock_list) >= target_count:
            print(f"✓ 方法1成功：获取到 {len(stock_list)} 只股票")
            return stock_list[:target_count]
        
        # 方法2：尝试获取股票基本信息
        stock_list = self._try_get_stock_info()
        if stock_list and len(stock_list) >= target_count:
            print(f"✓ 方法2成功：获取到 {len(stock_list)} 只股票")
            return stock_list[:target_count]
        
        # 方法3：尝试获取指数成分股
        stock_list = self._try_get_index_components()
        if stock_list and len(stock_list) >= target_count:
            print(f"✓ 方法3成功：获取到 {len(stock_list)} 只股票")
            return stock_list[:target_count]
        
        # 方法4：使用扩展的备用股票池
        print("✗ 所有API方法失败，使用扩展备用股票池")
        return self._get_extended_backup_list(target_count)
    
    def _try_get_spot_data(self):
        """
        尝试获取A股实时行情数据
        
        返回:
        list: 股票代码列表或None
        """
        try:
            print("尝试方法1：获取A股实时行情...")
            df = ak.stock_zh_a_spot_em()
            
            if df is None or len(df) == 0:
                return None
            
            # 过滤股票代码
            stock_codes = df['代码'].tolist()
            filtered_codes = []
            
            for code in stock_codes:
                if (len(code) == 6 and code.isdigit() and 
                    not any(x in df[df['代码']==code]['名称'].iloc[0] for x in ['ST', '退', 'N '])):
                    filtered_codes.append(code)
            
            return filtered_codes
            
        except Exception as e:
            print(f"方法1失败: {e}")
            return None
    
    def _try_get_stock_info(self):
        """
        尝试获取股票基本信息
        
        返回:
        list: 股票代码列表或None
        """
        try:
            print("尝试方法2：获取股票基本信息...")
            # 添加随机延时避免频率限制
            time.sleep(random.uniform(1, 3))
            
            df = ak.stock_info_a_code_name()
            
            if df is None or len(df) == 0:
                return None
            
            # 过滤股票代码
            stock_codes = df['code'].tolist()
            filtered_codes = []
            
            for i, code in enumerate(stock_codes):
                if i < len(df) and len(code) == 6 and code.isdigit():
                    name = df.iloc[i]['name'] if 'name' in df.columns else ''
                    if not any(x in name for x in ['ST', '退', 'N ']):
                        filtered_codes.append(code)
            
            return filtered_codes
            
        except Exception as e:
            print(f"方法2失败: {e}")
            return None
    
    def _try_get_index_components(self):
        """
        尝试获取主要指数成分股
        
        返回:
        list: 股票代码列表或None
        """
        try:
            print("尝试方法3：获取指数成分股...")
            all_stocks = set()
            
            # 主要指数列表
            indices = [
                ('沪深300', 'hs300'),
                ('中证500', 'zz500'),
                ('上证50', 'sz50'),
                ('创业板指', 'cyb')
            ]
            
            for index_name, index_code in indices:
                try:
                    time.sleep(random.uniform(0.5, 1.5))
                    
                    if index_code == 'hs300':
                        df = ak.index_stock_cons_weight_csindex(symbol="000300")
                    elif index_code == 'zz500':
                        df = ak.index_stock_cons_weight_csindex(symbol="000905")
                    elif index_code == 'sz50':
                        df = ak.index_stock_cons_weight_csindex(symbol="000016")
                    else:
                        continue
                    
                    if df is not None and len(df) > 0:
                        codes = df['成分券代码'].tolist() if '成分券代码' in df.columns else df.iloc[:, 0].tolist()
                        for code in codes:
                            if len(str(code)) == 6 and str(code).isdigit():
                                all_stocks.add(str(code))
                        print(f"  {index_name}: 获取到 {len(codes)} 只成分股")
                    
                except Exception as e:
                    print(f"  {index_name} 获取失败: {e}")
                    continue
            
            return list(all_stocks) if all_stocks else None
            
        except Exception as e:
            print(f"方法3失败: {e}")
            return None
    
    def _get_extended_backup_list(self, target_count):
        """
        获取扩展的备用股票列表
        
        参数:
        target_count: int, 目标数量
        
        返回:
        list: 股票代码列表
        """
        print(f"使用扩展备用股票池，目标数量: {target_count}")
        
        # 合并所有备用股票池
        all_backup_stocks = []
        for pool_name, stocks in self.backup_stock_pools.items():
            all_backup_stocks.extend(stocks)
            print(f"  {pool_name}: {len(stocks)} 只股票")
        
        # 去重并随机打乱
        unique_stocks = list(set(all_backup_stocks))
        random.shuffle(unique_stocks)
        
        print(f"备用股票池总计: {len(unique_stocks)} 只股票")
        
        # 如果备用股票不够，生成一些常见的股票代码
        if len(unique_stocks) < target_count:
            print("备用股票不足，生成补充股票代码...")
            
            # 生成一些常见的股票代码范围
            additional_stocks = []
            
            # 000001-000999 (深市主板)
            for i in range(1, 1000):
                code = f"{i:06d}"
                if code not in unique_stocks:
                    additional_stocks.append(code)
                    if len(unique_stocks) + len(additional_stocks) >= target_count:
                        break
            
            # 600000-600999 (沪市主板)
            if len(unique_stocks) + len(additional_stocks) < target_count:
                for i in range(600000, 601000):
                    code = str(i)
                    if code not in unique_stocks and code not in additional_stocks:
                        additional_stocks.append(code)
                        if len(unique_stocks) + len(additional_stocks) >= target_count:
                            break
            
            unique_stocks.extend(additional_stocks)
            print(f"补充后股票池: {len(unique_stocks)} 只股票")
        
        return unique_stocks[:target_count]
    
    def download_stock_data_with_retry(self, stock_code, days=300, max_retries=3):
        """
        带重试机制的股票数据下载
        
        参数:
        stock_code: str, 股票代码
        days: int, 数据天数
        max_retries: int, 最大重试次数
        
        返回:
        DataFrame: 股票数据或None
        """
        for attempt in range(max_retries):
            try:
                # 添加随机延时避免频率限制
                if attempt > 0:
                    delay = random.uniform(1, 3) * (attempt + 1)
                    time.sleep(delay)
                
                # 计算日期范围
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days*2)
                
                # 获取股票历史数据
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"  # 前复权
                )
                
                if df is None or len(df) == 0:
                    if attempt < max_retries - 1:
                        print(f"  股票 {stock_code} 数据为空，重试 {attempt + 1}/{max_retries}")
                        continue
                    return None
                
                # 数据预处理
                df = self._preprocess_stock_data(df, stock_code)
                
                if df is not None and len(df) >= days // 2:  # 至少要有一半的数据
                    # 保存数据
                    file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
                    df.to_csv(file_path)
                    return df
                else:
                    if attempt < max_retries - 1:
                        print(f"  股票 {stock_code} 数据不足，重试 {attempt + 1}/{max_retries}")
                        continue
                    return None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  股票 {stock_code} 下载失败: {e}，重试 {attempt + 1}/{max_retries}")
                    continue
                else:
                    print(f"  股票 {stock_code} 最终下载失败: {e}")
                    return None
        
        return None
    
    def _preprocess_stock_data(self, df, stock_code):
        """
        预处理股票数据
        
        参数:
        df: DataFrame, 原始数据
        stock_code: str, 股票代码
        
        返回:
        DataFrame: 处理后的数据
        """
        try:
            # 处理列名
            if len(df.columns) == 12 and '股票代码' in df.columns:
                df = df.drop('股票代码', axis=1)
            
            # 重命名列
            expected_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 
                              'amount', 'amplitude', 'change_pct', 'change_amount', 'turnover']
            
            if len(df.columns) == len(expected_columns):
                df.columns = expected_columns
            else:
                # 如果列数不匹配，使用基本列名
                basic_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
                if len(df.columns) >= len(basic_columns):
                    new_columns = basic_columns + [f'col_{i}' for i in range(len(basic_columns), len(df.columns))]
                    df.columns = new_columns
                else:
                    print(f"警告: 股票 {stock_code} 列数异常: {len(df.columns)}")
                    return None
            
            # 转换数据类型
            df['date'] = pd.to_datetime(df['date'])
            
            # 转换数值列
            numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'amount']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 设置索引
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # 删除无效数据
            df = df.dropna(subset=['open', 'close', 'high', 'low'])
            
            return df
            
        except Exception as e:
            print(f"数据预处理失败 {stock_code}: {e}")
            return None
    
    def batch_download(self, target_count=500, days=300, batch_size=50):
        """
        批量下载股票数据
        
        参数:
        target_count: int, 目标股票数量
        days: int, 数据天数
        batch_size: int, 批次大小
        
        返回:
        dict: 下载结果统计
        """
        print(f"=== 开始批量下载 {target_count} 只股票数据 ===")
        
        # 获取股票列表
        stock_list = self.get_stock_list_multiple_methods(target_count)
        
        if not stock_list:
            print("✗ 无法获取股票列表")
            return None
        
        print(f"准备下载 {len(stock_list)} 只股票的数据")
        
        # 分批下载
        success_count = 0
        failed_count = 0
        failed_stocks = []
        
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i:i+batch_size]
            print(f"\n=== 批次 {i//batch_size + 1}: 下载 {len(batch)} 只股票 ===")
            
            for stock_code in tqdm(batch, desc=f"批次{i//batch_size + 1}"):
                df = self.download_stock_data_with_retry(stock_code, days)
                
                if df is not None:
                    success_count += 1
                else:
                    failed_count += 1
                    failed_stocks.append(stock_code)
                
                # 批次内延时
                time.sleep(random.uniform(0.1, 0.3))
            
            # 批次间延时
            if i + batch_size < len(stock_list):
                print(f"批次完成，休息 {random.uniform(2, 5):.1f} 秒...")
                time.sleep(random.uniform(2, 5))
        
        # 保存下载结果
        result = {
            'total_target': target_count,
            'total_attempted': len(stock_list),
            'success': success_count,
            'failed': failed_count,
            'success_rate': success_count / len(stock_list) * 100,
            'failed_stocks': failed_stocks,
            'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_days': days
        }
        
        # 保存结果到JSON文件
        with open(os.path.join(self.data_dir, 'download_result_optimized.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 下载完成 ===")
        print(f"目标数量: {target_count}")
        print(f"尝试下载: {len(stock_list)}")
        print(f"成功: {success_count} 只")
        print(f"失败: {failed_count} 只")
        print(f"成功率: {result['success_rate']:.1f}%")
        
        return result
    
    def calculate_rps(self, rps_period=50):
        """
        计算RPS指标
        
        参数:
        rps_period: int, RPS计算周期
        
        返回:
        dict: RPS数据
        """
        print(f"\n=== 开始计算 {rps_period} 日RPS ===")
        
        # 加载所有股票数据
        all_returns = {}
        loaded_count = 0
        
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv') and len(file_name) == 10:  # 格式: 000001.csv
                stock_code = file_name[:-4]
                
                try:
                    file_path = os.path.join(self.data_dir, file_name)
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    if 'close' in df.columns and len(df) >= rps_period:
                        # 计算收益率
                        returns = df['close'].pct_change(rps_period).dropna()
                        if len(returns) > 0:
                            all_returns[stock_code] = returns
                            loaded_count += 1
                    
                except Exception as e:
                    print(f"加载股票 {stock_code} 数据失败: {e}")
                    continue
        
        print(f"成功加载 {loaded_count} 只股票的收益率数据")
        
        if loaded_count < 10:
            print("✗ 股票数量不足，无法计算RPS")
            return None
        
        # 计算RPS
        rps_data = {}
        
        # 获取所有日期的并集
        all_dates = set()
        for returns in all_returns.values():
            all_dates.update(returns.index)
        all_dates = sorted(list(all_dates))
        
        print(f"共有 {len(all_dates)} 个交易日期")
        
        for date in tqdm(all_dates, desc="计算RPS"):
            # 获取该日期所有股票的收益率
            date_returns = {}
            for stock_code, returns in all_returns.items():
                if date in returns.index:
                    date_returns[stock_code] = returns[date]
            
            if len(date_returns) < 5:  # 至少需要5只股票
                continue
            
            # 计算RPS
            returns_list = list(date_returns.values())
            valid_returns = [r for r in returns_list if not np.isnan(r)]
            
            if len(valid_returns) >= 3:  # 至少需要3只股票的有效数据
                for stock_code, return_val in date_returns.items():
                    if not np.isnan(return_val):
                        # 计算该股票收益率在所有股票中的排名百分比
                        rank = sum(1 for r in valid_returns if r < return_val)
                        rps_value = (rank / len(valid_returns)) * 100
                        
                        if stock_code not in rps_data:
                            rps_data[stock_code] = {}
                        rps_data[stock_code][date] = rps_value
        
        # 保存RPS数据
        rps_file = os.path.join(self.data_dir, f'rps_{rps_period}d_optimized.pkl')
        with open(rps_file, 'wb') as f:
            pickle.dump(rps_data, f)
        
        print(f"✓ RPS计算完成，数据已保存到 {rps_file}")
        print(f"计算了 {len(rps_data)} 只股票的RPS数据")
        
        return rps_data

def main():
    """
    主函数
    """
    print("=== 优化的A股市场数据下载器 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建下载器实例
    proxy_config = None
    if os.getenv('HTTP_PROXY') and os.getenv('HTTPS_PROXY'):
        proxy_config = {
            'http_proxy': os.getenv('HTTP_PROXY'),
            'https_proxy': os.getenv('HTTPS_PROXY')
        }
    
    downloader = OptimizedMarketDownloader(
        data_dir="market_data",
        proxy_config=proxy_config
    )
    
    # 配置参数
    target_count = 300  # 目标下载300只股票
    days = 300  # 获取300天数据
    batch_size = 30  # 每批30只股票
    
    print(f"\n配置参数:")
    print(f"  目标股票数量: {target_count}")
    print(f"  数据天数: {days}")
    print(f"  批次大小: {batch_size}")
    
    # 批量下载数据
    result = downloader.batch_download(
        target_count=target_count,
        days=days,
        batch_size=batch_size
    )
    
    if result and result['success'] >= 10:
        print("\n=== 开始计算RPS ===")
        rps_data = downloader.calculate_rps(rps_period=50)
        
        if rps_data:
            print("\n=== 数据下载和RPS计算全部完成 ===")
            print("现在可以使用优化后的数据进行选股分析了！")
        else:
            print("\n=== 数据下载完成，但RPS计算失败 ===")
    else:
        print("\n=== 下载的股票数量不足，无法计算RPS ===")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return result

if __name__ == "__main__":
    main()