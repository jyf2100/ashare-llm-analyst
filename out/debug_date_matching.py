#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试日期匹配问题
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

def debug_date_matching():
    """调试日期匹配问题"""
    print("🔍 调试日期匹配问题")
    print("=" * 50)
    
    # 1. 加载市场数据样本
    market_data_dir = "market_data"
    sample_files = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')][:3]
    
    market_data = {}
    for file in sample_files:
        stock_code = file.replace('.csv', '')
        df = pd.read_csv(os.path.join(market_data_dir, file))
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        market_data[stock_code] = df
        print(f"📈 {stock_code}: {len(df)} 条记录, 日期范围: {df.index.min()} 到 {df.index.max()}")
    
    # 2. 加载RPS数据
    rps_dir = "rps_results"
    rps_files = [f for f in os.listdir(rps_dir) if f.endswith('.pkl')]
    
    rps_data = {}
    for rps_file in rps_files:
        print(f"\n📁 处理RPS文件: {rps_file}")
        file_path = os.path.join(rps_dir, rps_file)
        
        with open(file_path, 'rb') as f:
            raw_rps_data = pickle.load(f)
        
        # 处理RPS数据结构
        for stock_code, rps_list in raw_rps_data.items():
            if stock_code not in rps_data:
                rps_data[stock_code] = {}
            
            if isinstance(rps_list, list) and len(rps_list) > 0:
                rps_dict = {}
                for item in rps_list:
                    if isinstance(item, dict) and 'date' in item and 'rps' in item:
                        rps_dict[item['date']] = float(item['rps'])
                
                if rps_dict:
                    rps_series = pd.Series(rps_dict)
                    rps_series.index = pd.to_datetime(rps_series.index)
                    rps_data[stock_code]['rps'] = rps_series.sort_index()
        
        print(f"  处理后股票数量: {len(rps_data)}")
        break  # 只处理第一个文件
    
    # 3. 详细调试日期匹配
    print("\n🔍 详细调试日期匹配")
    print("=" * 50)
    
    for stock_code in list(market_data.keys())[:2]:  # 只检查前2只股票
        print(f"\n📊 股票: {stock_code}")
        
        if stock_code not in market_data:
            print(f"  ❌ 市场数据中没有 {stock_code}")
            continue
            
        market_df = market_data[stock_code]
        print(f"  市场数据: {len(market_df)} 条记录")
        print(f"  市场数据日期范围: {market_df.index.min()} 到 {market_df.index.max()}")
        print(f"  市场数据前5个日期: {market_df.index[:5].tolist()}")
        print(f"  市场数据后5个日期: {market_df.index[-5:].tolist()}")
        
        if stock_code in rps_data and 'rps' in rps_data[stock_code]:
            rps_series = rps_data[stock_code]['rps']
            print(f"  ✅ RPS数据: {len(rps_series)} 条记录")
            print(f"  RPS数据日期范围: {rps_series.index.min()} 到 {rps_series.index.max()}")
            print(f"  RPS数据前5个日期: {rps_series.index[:5].tolist()}")
            print(f"  RPS数据后5个日期: {rps_series.index[-5:].tolist()}")
            
            # 检查日期重叠
            market_dates = set(market_df.index)
            rps_dates = set(rps_series.index)
            overlap_dates = market_dates.intersection(rps_dates)
            
            print(f"  📅 日期重叠分析:")
            print(f"    市场数据日期数: {len(market_dates)}")
            print(f"    RPS数据日期数: {len(rps_dates)}")
            print(f"    重叠日期数: {len(overlap_dates)}")
            print(f"    重叠比例: {len(overlap_dates)/len(market_dates)*100:.1f}%")
            
            if len(overlap_dates) > 0:
                overlap_list = sorted(list(overlap_dates))
                print(f"    前5个重叠日期: {overlap_list[:5]}")
                print(f"    后5个重叠日期: {overlap_list[-5:]}")
                
                # 测试_get_rps_value方法
                print(f"  🧪 测试RPS值获取:")
                test_dates = overlap_list[:5]
                for test_date in test_dates:
                    rps_value = get_rps_value(rps_series, test_date)
                    actual_rps = rps_series.get(test_date, 'N/A')
                    print(f"    {test_date}: 获取值={rps_value}, 实际值={actual_rps}")
            else:
                print(f"    ❌ 没有重叠日期")
                
                # 分析日期差异
                market_sample = sorted(list(market_dates))[:5]
                rps_sample = sorted(list(rps_dates))[:5]
                print(f"    市场数据样本日期: {market_sample}")
                print(f"    RPS数据样本日期: {rps_sample}")
                
                # 检查日期格式
                print(f"    市场数据日期类型: {type(market_df.index[0])}")
                print(f"    RPS数据日期类型: {type(rps_series.index[0])}")
        else:
            print(f"  ❌ RPS数据中没有 {stock_code}")

def get_rps_value(rps_series, target_date):
    """模拟_get_rps_value方法"""
    try:
        valid_dates = rps_series.index[rps_series.index <= target_date]
        if len(valid_dates) > 0:
            rps_value = float(rps_series[valid_dates.max()])
            # 确保RPS值在合理范围内
            if 0 <= rps_value <= 100:
                return rps_value
    except Exception as e:
        print(f"      错误: {e}")
    return 50.0  # 返回默认值

def suggest_solutions():
    """建议解决方案"""
    print("\n🔧 解决方案建议")
    print("=" * 50)
    print("基于调试结果，可能的解决方案:")
    print("1. 如果日期格式不匹配，需要统一日期格式")
    print("2. 如果日期范围不重叠，需要检查数据时间范围")
    print("3. 如果RPS数据缺失，需要重新生成RPS数据")
    print("4. 如果日期匹配逻辑有问题，需要修改匹配算法")

if __name__ == "__main__":
    debug_date_matching()
    suggest_solutions()