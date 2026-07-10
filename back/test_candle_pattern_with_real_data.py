#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实股票数据测试K线形态分析器
"""

import pandas as pd
import numpy as np
import os
import sys
from independent_candle_pattern_analyzer import IndependentCandlePatternAnalyzer


def create_test_data_with_patterns():
    """创建包含特定K线形态的测试数据"""
    
    # 测试数据1: 锤子线形态
    hammer_data = {
        'date': pd.date_range('2024-01-01', periods=5),
        'open': [100, 102, 98, 95, 100],
        'high': [105, 104, 100, 97, 102],
        'low': [95, 96, 94, 92, 95],  # 最后一根K线有长下影线
        'close': [102, 98, 95, 96, 101],  # 最后一根收盘接近高点
        'volume': [1000, 1200, 800, 900, 1500]
    }
    
    # 测试数据2: 看涨吞没形态
    engulfing_data = {
        'date': pd.date_range('2024-01-01', periods=5),
        'open': [100, 102, 98, 95, 96],  # 第4根阴线
        'high': [105, 104, 100, 97, 102],
        'low': [95, 96, 94, 92, 95],
        'close': [102, 98, 95, 94, 101],  # 第5根阳线吞没前一根
        'volume': [1000, 1200, 800, 900, 2000]
    }
    
    # 测试数据3: 十字星形态
    doji_data = {
        'date': pd.date_range('2024-01-01', periods=5),
        'open': [100, 102, 98, 95, 100],
        'high': [105, 104, 100, 97, 101],
        'low': [95, 96, 94, 92, 99],
        'close': [102, 98, 95, 96, 100.1],  # 最后一根实体很小
        'volume': [1000, 1200, 800, 900, 1200]
    }
    
    return {
        'hammer': pd.DataFrame(hammer_data),
        'engulfing': pd.DataFrame(engulfing_data),
        'doji': pd.DataFrame(doji_data)
    }


def test_with_market_data():
    """使用市场数据目录中的真实数据进行测试"""
    market_data_dir = 'out/market_data/'
    
    if not os.path.exists(market_data_dir):
        print(f"❌ 市场数据目录不存在: {market_data_dir}")
        return
    
    # 获取前几个CSV文件
    csv_files = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')][:5]
    
    if not csv_files:
        print("❌ 没有找到CSV文件")
        return
    
    print(f"📊 找到 {len(csv_files)} 个CSV文件，开始测试...")
    
    for csv_file in csv_files:
        file_path = os.path.join(market_data_dir, csv_file)
        try:
            print(f"\n🔍 分析文件: {csv_file}")
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查必要列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  跳过 - 缺少列: {missing_cols}")
                continue
            
            # 使用最后10根K线进行分析
            recent_data = df.tail(10).copy()
            
            # 创建分析器
            analyzer = IndependentCandlePatternAnalyzer(recent_data)
            
            # 分析形态
            patterns = analyzer.analyze_all_patterns()
            
            # 获取交易建议
            recommendation = analyzer.get_trading_recommendation()
            
            print(f"  📈 检测到 {len(patterns)} 个形态:")
            for pattern in patterns:
                print(f"    • {pattern}")
            
            print(f"  💡 建议: {recommendation['recommendation']} "
                  f"(置信度: {recommendation['confidence']}%)")
            
        except Exception as e:
            print(f"  ❌ 分析出错: {e}")


def main():
    """主测试函数"""
    print("=" * 70)
    print("📊 K线形态分析器 - 综合测试")
    print("=" * 70)
    
    # 测试1: 使用模拟数据
    print("\n1. 🧪 使用模拟数据测试:")
    test_data = create_test_data_with_patterns()
    
    for pattern_name, df in test_data.items():
        print(f"\n  测试 {pattern_name} 形态:")
        try:
            analyzer = IndependentCandlePatternAnalyzer(df)
            patterns = analyzer.analyze_all_patterns()
            
            print(f"    检测到 {len(patterns)} 个形态:")
            for p in patterns:
                print(f"      • {p}")
                
        except Exception as e:
            print(f"    ❌ 错误: {e}")
    
    # 测试2: 使用真实市场数据
    print("\n2. 📈 使用真实市场数据测试:")
    test_with_market_data()
    
    print("\n" + "=" * 70)
    print("✅ 测试完成!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())