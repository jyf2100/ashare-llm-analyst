#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""测试价格模式识别功能"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detailed_daily_analyzer import DetailedDailyAnalyzer
import pandas as pd
import numpy as np

def test_pattern_recognition():
    """测试模式识别功能"""
    print("=== 测试价格模式识别功能 ===")
    
    # 创建分析器实例
    analyzer = DetailedDailyAnalyzer()
    
    # 创建测试数据 - 模拟头肩顶模式
    print("\n1. 测试头肩顶模式识别...")
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    # 模拟头肩顶价格数据
    prices = [
        100, 102, 105, 108, 110,  # 左肩上升
        112, 115, 118, 120, 125,   # 头部上升
        122, 118, 115, 112, 110,   # 头部下降
        108, 105, 102, 100, 98,    # 右肩形成
        95, 92, 88, 85, 82,        # 颈线突破
        80, 78, 75, 72, 70         # 下跌趋势
    ]
    
    test_df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * 1.02 for p in prices],  # 高点略高于收盘价
        'low': [p * 0.98 for p in prices],   # 低点略低于收盘价
        'close': prices,
        'volume': [1000000] * len(prices)
    })
    
    # 添加必要的技术指标列（简化版本）
    test_df['MA20'] = test_df['close'].rolling(20).mean()
    test_df['MA60'] = test_df['close'].rolling(60).mean()
    test_df['MA120'] = test_df['close'].rolling(120).mean()
    test_df['Volatility_20d'] = test_df['close'].rolling(20).std()
    test_df['ATR'] = test_df['high'] - test_df['low']
    
    # 测试模式识别
    patterns = analyzer._identify_price_patterns(test_df)
    
    print(f"识别到的模式数量: {len(patterns)}")
    for i, pattern in enumerate(patterns, 1):
        print(f"\n模式 {i}:")
        for key, value in pattern.items():
            print(f"  {key}: {value}")
    
    # 测试其他模式
    print("\n2. 测试其他模式识别方法...")
    
    # 测试双顶模式
    print("\n测试双顶模式:")
    double_top_data = [
        100, 105, 110, 115, 120, 125, 120, 115, 110, 105,  # 第一个顶
        100, 105, 110, 115, 120, 125, 120, 115, 110, 105   # 第二个顶
    ]
    
    test_df2 = pd.DataFrame({
        'high': double_top_data,
        'low': [p * 0.95 for p in double_top_data],
        'close': double_top_data
    })
    
    double_top = analyzer._detect_double_top(test_df2)
    if double_top:
        print("双顶模式识别成功!")
        for key, value in double_top.items():
            print(f"  {key}: {value}")
    else:
        print("未识别到双顶模式")
    
    # 测试双底模式
    print("\n测试双底模式:")
    double_bottom_data = [
        125, 120, 115, 110, 105, 100, 105, 110, 115, 120,  # 第一个底
        125, 120, 115, 110, 105, 100, 105, 110, 115, 120   # 第二个底
    ]
    
    test_df3 = pd.DataFrame({
        'high': [p * 1.05 for p in double_bottom_data],
        'low': double_bottom_data,
        'close': double_bottom_data
    })
    
    double_bottom = analyzer._detect_double_bottom(test_df3)
    if double_bottom:
        print("双底模式识别成功!")
        for key, value in double_bottom.items():
            print(f"  {key}: {value}")
    else:
        print("未识别到双底模式")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_pattern_recognition()