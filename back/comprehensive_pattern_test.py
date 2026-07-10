#!/usr/bin/env python3
"""
综合测试所有价格模式识别功能
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detailed_daily_analyzer import DetailedDailyAnalyzer

def create_head_shoulder_top_data():
    """创建头肩顶模式测试数据"""
    # 左肩
    left_shoulder = np.linspace(90, 110, 5)
    # 头部
    head = np.linspace(115, 125, 7)
    # 右肩
    right_shoulder = np.linspace(110, 95, 5)
    
    prices = np.concatenate([left_shoulder, head, right_shoulder])
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(prices), freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(10000, 50000, len(prices))
    })
    df.set_index('date', inplace=True)
    return df

def create_head_shoulder_bottom_data():
    """创建头肩底模式测试数据"""
    # 左肩
    left_shoulder = np.linspace(110, 90, 5)
    # 头部
    head = np.linspace(85, 75, 7)
    # 右肩
    right_shoulder = np.linspace(90, 100, 5)
    
    prices = np.concatenate([left_shoulder, head, right_shoulder])
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(prices), freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * 1.01,
        'high': prices * 1.03,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(10000, 50000, len(prices))
    })
    df.set_index('date', inplace=True)
    return df

def create_ascending_triangle_data():
    """创建上升三角形测试数据"""
    # 水平阻力线
    resistance = 120
    # 上升支撑线
    support = np.linspace(100, 115, 15)
    
    prices = []
    for s in support:
        # 价格在支撑和阻力之间波动
        price = np.random.uniform(s, resistance)
        prices.append(price)
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(prices), freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'open': np.array(prices) * 0.99,
        'high': np.array([resistance] * len(prices)),
        'low': np.array(prices) * 0.98,
        'close': prices,
        'volume': np.random.randint(10000, 50000, len(prices))
    })
    df.set_index('date', inplace=True)
    return df

def create_descending_triangle_data():
    """创建下降三角形测试数据"""
    # 水平支撑线
    support = 100
    # 下降阻力线
    resistance = np.linspace(120, 105, 15)
    
    prices = []
    for r in resistance:
        # 价格在支撑和阻力之间波动
        price = np.random.uniform(support, r)
        prices.append(price)
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(prices), freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'open': np.array(prices) * 1.01,
        'high': np.array(resistance),
        'low': np.array([support] * len(prices)),
        'close': prices,
        'volume': np.random.randint(10000, 50000, len(prices))
    })
    df.set_index('date', inplace=True)
    return df

def create_flag_pattern_data():
    """创建旗形模式测试数据"""
    # 快速上涨阶段
    uptrend = np.linspace(80, 120, 10)
    # 整理阶段（旗形）
    consolidation = np.random.uniform(115, 125, 10)
    
    prices = np.concatenate([uptrend, consolidation])
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(prices), freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.concatenate([
            np.random.randint(50000, 100000, 10),  # 上涨阶段高成交量
            np.random.randint(20000, 40000, 10)   # 整理阶段低成交量
        ])
    })
    df.set_index('date', inplace=True)
    return df

def test_all_patterns():
    """测试所有模式识别功能"""
    print("=== 综合测试所有价格模式识别功能 ===\n")
    
    analyzer = DetailedDailyAnalyzer()
    
    # 测试头肩顶
    print("1. 测试头肩顶模式识别...")
    hs_top_data = create_head_shoulder_top_data()
    result = analyzer._detect_head_shoulder_top(hs_top_data)
    if result:
        print(f"  识别成功: {result['pattern']} - {result['description']}")
        print(f"  头部价格: {result['head_price']:.2f}")
        print(f"  左肩价格: {result['left_shoulder_price']:.2f}")
        print(f"  右肩价格: {result['right_shoulder_price']:.2f}")
    else:
        print("  未识别到头肩顶模式")
    print()
    
    # 测试头肩底
    print("2. 测试头肩底模式识别...")
    hs_bottom_data = create_head_shoulder_bottom_data()
    result = analyzer._detect_head_shoulder_bottom(hs_bottom_data)
    if result:
        print(f"  识别成功: {result['pattern']} - {result['description']}")
        print(f"  头部价格: {result['head_price']:.2f}")
        print(f"  左肩价格: {result['left_shoulder_price']:.2f}")
        print(f"  右肩价格: {result['right_shoulder_price']:.2f}")
    else:
        print("  未识别到头肩底模式")
    print()
    
    # 测试上升三角形
    print("3. 测试上升三角形模式识别...")
    asc_triangle_data = create_ascending_triangle_data()
    result = analyzer._detect_ascending_triangle(asc_triangle_data)
    if result:
        print(f"  识别成功: {result['pattern']} - {result['description']}")
        print(f"  阻力位: {result['resistance_level']:.2f}")
        print(f"  支撑趋势: {result['support_trend']}")
    else:
        print("  未识别到上升三角形模式")
    print()
    
    # 测试下降三角形
    print("4. 测试下降三角形模式识别...")
    desc_triangle_data = create_descending_triangle_data()
    result = analyzer._detect_descending_triangle(desc_triangle_data)
    if result:
        print(f"  识别成功: {result['pattern']} - {result['description']}")
        print(f"  支撑位: {result['support_level']:.2f}")
        print(f"  阻力趋势: {result['resistance_trend']}")
    else:
        print("  未识别到下降三角形模式")
    print()
    
    # 测试旗形模式
    print("5. 测试旗形模式识别...")
    flag_data = create_flag_pattern_data()
    result = analyzer._detect_flag_pattern(flag_data)
    if result:
        print(f"  识别成功: {result['pattern']} - {result['description']}")
        print(f"  趋势强度: {result['trend_strength']:.4f}")
        print(f"  整理强度: {result['consolidation_strength']:.4f}")
    else:
        print("  未识别到旗形模式")
    print()
    
    # 测试综合模式识别
    print("6. 测试综合模式识别(_identify_price_patterns)...")
    all_patterns = analyzer._identify_price_patterns(hs_top_data)
    print(f"  识别到的模式数量: {len(all_patterns)}")
    for pattern in all_patterns:
        print(f"  - {pattern['pattern']}: {pattern['description']}")
    print()

if __name__ == "__main__":
    test_all_patterns()