#!/usr/bin/env python3
"""
简单测试模式识别功能
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detailed_daily_analyzer import DetailedDailyAnalyzer

def create_test_data_with_indicators():
    """创建包含技术指标的测试数据"""
    # 创建价格数据
    np.random.seed(42)
    n_days = 100
    
    # 基础趋势
    trend = np.linspace(100, 120, n_days)
    # 随机波动
    noise = np.random.normal(0, 2, n_days)
    
    close_prices = trend + noise
    
    # 计算高点和低点
    high_prices = close_prices + np.abs(np.random.normal(1, 0.5, n_days))
    low_prices = close_prices - np.abs(np.random.normal(1, 0.5, n_days))
    open_prices = close_prices - np.random.normal(0, 0.5, n_days)
    
    # 创建DataFrame
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(10000, 50000, n_days)
    })
    df.set_index('date', inplace=True)
    
    # 添加技术指标（简化版本）
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()
    df['MA120'] = df['close'].rolling(window=120).mean()
    df['Volatility_20d'] = df['close'].rolling(window=20).std()
    df['ATR'] = (df['high'] - df['low']).rolling(window=14).mean()
    
    # 布林带
    df['BB_Middle_20'] = df['close'].rolling(window=20).mean()
    df['BB_Upper_20'] = df['BB_Middle_20'] + 2 * df['close'].rolling(window=20).std()
    df['BB_Lower_20'] = df['BB_Middle_20'] - 2 * df['close'].rolling(window=20).std()
    
    # 填充NaN值
    df = df.fillna(method='bfill')
    
    return df

def test_individual_pattern_methods():
    """测试单个模式识别方法"""
    print("=== 测试单个模式识别方法 ===\n")
    
    analyzer = DetailedDailyAnalyzer()
    test_data = create_test_data_with_indicators()
    
    # 测试每个模式识别方法
    methods = [
        ('头肩顶', analyzer._detect_head_shoulder_top),
        ('头肩底', analyzer._detect_head_shoulder_bottom),
        ('双顶', analyzer._detect_double_top),
        ('双底', analyzer._detect_double_bottom),
        ('上升三角形', analyzer._detect_ascending_triangle),
        ('下降三角形', analyzer._detect_descending_triangle),
        ('对称三角形', analyzer._detect_symmetrical_triangle),
        ('旗形', analyzer._detect_flag_pattern)
    ]
    
    for pattern_name, method in methods:
        result = method(test_data)
        if result:
            print(f"✓ {pattern_name}: 识别成功")
            print(f"  类型: {result['type']}, 置信度: {result['confidence']}")
        else:
            print(f"✗ {pattern_name}: 未识别")
    print()

def test_comprehensive_pattern_recognition():
    """测试综合模式识别"""
    print("=== 测试综合模式识别 ===\n")
    
    analyzer = DetailedDailyAnalyzer()
    test_data = create_test_data_with_indicators()
    
    # 测试综合模式识别
    patterns = analyzer._identify_price_patterns(test_data)
    
    print(f"识别到的模式数量: {len(patterns)}")
    for pattern in patterns:
        print(f"- {pattern['pattern']} ({pattern['type']}): {pattern['description']}")
    print()

def create_double_top_data():
    """创建双顶模式测试数据"""
    # 第一个顶
    first_top = np.linspace(90, 120, 10)
    # 回调
    correction = np.linspace(120, 100, 5)
    # 第二个顶
    second_top = np.linspace(100, 120, 10)
    # 下跌
    decline = np.linspace(120, 90, 10)
    
    prices = np.concatenate([first_top, correction, second_top, decline])
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
    
    # 添加技术指标
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()
    df['Volatility_20d'] = df['close'].rolling(window=20).std()
    df['ATR'] = (df['high'] - df['low']).rolling(window=14).mean()
    df = df.fillna(method='bfill')
    
    return df

def test_with_double_top():
    """使用双顶模式数据测试"""
    print("=== 使用双顶模式数据测试 ===\n")
    
    analyzer = DetailedDailyAnalyzer()
    double_top_data = create_double_top_data()
    
    # 测试双顶模式识别
    result = analyzer._detect_double_top(double_top_data)
    if result:
        print("✓ 双顶模式识别成功:")
        print(f"  第一个顶: {result['top1_price']:.2f}")
        print(f"  第二个顶: {result['top2_price']:.2f}")
        print(f"  颈线: {result['neckline_price']:.2f}")
    else:
        print("✗ 双顶模式未识别")
    print()
    
    # 测试综合模式识别
    patterns = analyzer._identify_price_patterns(double_top_data)
    print(f"综合识别到的模式数量: {len(patterns)}")
    for pattern in patterns:
        print(f"- {pattern['pattern']}")

if __name__ == "__main__":
    test_individual_pattern_methods()
    test_comprehensive_pattern_recognition()
    test_with_double_top()