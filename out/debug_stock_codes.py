#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试股票代码匹配问题
"""

import os
import pickle
import pandas as pd

def debug_stock_codes():
    """调试股票代码匹配问题"""
    print("🔍 调试股票代码匹配问题")
    print("=" * 50)
    
    # 1. 检查市场数据中的股票代码
    market_data_dir = "market_data"
    market_files = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')]
    market_codes = [f.replace('.csv', '') for f in market_files]
    
    print(f"📈 市场数据股票代码 (总数: {len(market_codes)})")
    print(f"  前10个: {market_codes[:10]}")
    print(f"  后10个: {market_codes[-10:]}")
    
    # 分析股票代码格式
    sh_codes = [code for code in market_codes if code.startswith('sh.')]
    sz_codes = [code for code in market_codes if code.startswith('sz.')]
    other_codes = [code for code in market_codes if not code.startswith(('sh.', 'sz.'))]
    
    print(f"  上海股票 (sh.): {len(sh_codes)} 个")
    print(f"  深圳股票 (sz.): {len(sz_codes)} 个")
    print(f"  其他格式: {len(other_codes)} 个")
    if other_codes:
        print(f"    其他格式样本: {other_codes[:5]}")
    
    # 2. 检查RPS数据中的股票代码
    rps_dir = "rps_results"
    rps_files = [f for f in os.listdir(rps_dir) if f.endswith('.pkl')]
    
    for rps_file in rps_files:
        print(f"\n📁 RPS文件: {rps_file}")
        file_path = os.path.join(rps_dir, rps_file)
        
        with open(file_path, 'rb') as f:
            rps_data = pickle.load(f)
        
        rps_codes = list(rps_data.keys())
        print(f"📊 RPS数据股票代码 (总数: {len(rps_codes)})")
        print(f"  前10个: {rps_codes[:10]}")
        print(f"  后10个: {rps_codes[-10:]}")
        
        # 分析RPS股票代码格式
        rps_sh_codes = [code for code in rps_codes if code.startswith('sh.')]
        rps_sz_codes = [code for code in rps_codes if code.startswith('sz.')]
        rps_other_codes = [code for code in rps_codes if not code.startswith(('sh.', 'sz.'))]
        
        print(f"  上海股票 (sh.): {len(rps_sh_codes)} 个")
        print(f"  深圳股票 (sz.): {len(rps_sz_codes)} 个")
        print(f"  其他格式: {len(rps_other_codes)} 个")
        if rps_other_codes:
            print(f"    其他格式样本: {rps_other_codes[:5]}")
        
        # 3. 检查代码匹配情况
        print(f"\n🔍 代码匹配分析")
        market_set = set(market_codes)
        rps_set = set(rps_codes)
        
        intersection = market_set.intersection(rps_set)
        market_only = market_set - rps_set
        rps_only = rps_set - market_set
        
        print(f"  市场数据股票数: {len(market_set)}")
        print(f"  RPS数据股票数: {len(rps_set)}")
        print(f"  共同股票数: {len(intersection)}")
        print(f"  仅在市场数据中: {len(market_only)}")
        print(f"  仅在RPS数据中: {len(rps_only)}")
        print(f"  匹配率: {len(intersection)/len(market_set)*100:.1f}%")
        
        if len(intersection) > 0:
            print(f"  共同股票样本: {list(intersection)[:10]}")
        
        if len(market_only) > 0:
            print(f"  仅在市场数据中的样本: {list(market_only)[:10]}")
        
        if len(rps_only) > 0:
            print(f"  仅在RPS数据中的样本: {list(rps_only)[:10]}")
        
        # 4. 检查具体的股票代码格式差异
        print(f"\n🔍 股票代码格式分析")
        
        # 检查是否存在格式转换问题
        sample_market_codes = market_codes[:5]
        for market_code in sample_market_codes:
            print(f"  市场代码: {market_code}")
            
            # 尝试不同的格式匹配
            possible_formats = [
                market_code,  # 原格式
                market_code.upper(),  # 大写
                market_code.lower(),  # 小写
                market_code.replace('sh.', '').replace('sz.', ''),  # 去掉前缀
                market_code.replace('.', ''),  # 去掉点
            ]
            
            found_matches = []
            for fmt in possible_formats:
                if fmt in rps_set:
                    found_matches.append(fmt)
            
            if found_matches:
                print(f"    ✅ 找到匹配: {found_matches}")
            else:
                print(f"    ❌ 未找到匹配")
                # 查找相似的代码
                similar_codes = []
                base_code = market_code.replace('sh.', '').replace('sz.', '')
                for rps_code in rps_codes:
                    if base_code in rps_code or rps_code in base_code:
                        similar_codes.append(rps_code)
                if similar_codes:
                    print(f"    🔍 相似代码: {similar_codes[:3]}")
        
        break  # 只处理第一个RPS文件

def suggest_fix():
    """建议修复方案"""
    print(f"\n🔧 修复建议")
    print("=" * 50)
    print("基于分析结果，可能的修复方案:")
    print("1. 如果股票代码格式不匹配，需要在加载时进行格式转换")
    print("2. 如果RPS数据中缺少某些股票，需要重新生成RPS数据")
    print("3. 如果存在大小写问题，需要统一大小写格式")
    print("4. 如果存在前缀问题，需要统一前缀格式")
    print("\n建议的修复步骤:")
    print("1. 修改load_rps_data方法，添加股票代码格式转换")
    print("2. 在匹配时尝试多种格式")
    print("3. 记录匹配失败的股票代码，便于调试")

if __name__ == "__main__":
    debug_stock_codes()
    suggest_fix()