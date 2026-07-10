#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面的ML策略测试脚本
测试ML增强策略在不同场景下的表现
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入ML策略
import importlib.util
spec = importlib.util.spec_from_file_location("optimized_stock_selection_v2", "-02-optimized_stock_selection_v2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MLEnhancedStrategy = module.MLEnhancedStrategy

def create_realistic_stock_data(stock_code, days=252, trend='random'):
    """
    创建更真实的股票数据
    
    Args:
        stock_code: 股票代码
        days: 天数
        trend: 趋势类型 ('up', 'down', 'sideways', 'random')
    """
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # 基础价格
    base_price = 50.0
    
    if trend == 'up':
        # 上升趋势
        trend_component = np.linspace(0, 20, days)
        volatility = 0.02
    elif trend == 'down':
        # 下降趋势
        trend_component = np.linspace(0, -15, days)
        volatility = 0.025
    elif trend == 'sideways':
        # 横盘整理
        trend_component = np.sin(np.linspace(0, 4*np.pi, days)) * 2
        volatility = 0.015
    else:
        # 随机走势
        trend_component = np.cumsum(np.random.randn(days) * 0.5)
        volatility = 0.02
    
    # 生成价格序列
    random_component = np.random.randn(days) * volatility * base_price
    prices = base_price + trend_component + random_component
    prices = np.maximum(prices, 1.0)  # 确保价格为正
    
    # 生成成交量
    volumes = np.random.lognormal(mean=15, sigma=0.5, size=days)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(days) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(days)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(days)) * 0.02),
        'close': prices,
        'volume': volumes
    })
    
    # 确保high >= max(open, close), low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # 设置日期为索引
    data.set_index('date', inplace=True)
    
    return data

def test_ml_strategy_scenarios():
    """
    测试ML策略在不同市场场景下的表现
    """
    print("🔬 开始全面ML策略测试\n")
    
    # 初始化ML策略
    print("=== 初始化ML策略 ===")
    config = {
        'ml_threshold': 0.6,
        'use_ensemble': True,
        'feature_importance_threshold': 0.01
    }
    ml_strategy = MLEnhancedStrategy(config)
    print(f"模型已训练: {ml_strategy.is_trained}")
    print(f"标准化器可用: {ml_strategy.scaler is not None}")
    
    # 测试不同趋势的股票
    scenarios = [
        ('上升趋势', 'up'),
        ('下降趋势', 'down'),
        ('横盘整理', 'sideways'),
        ('随机走势', 'random')
    ]
    
    results = []
    
    for scenario_name, trend in scenarios:
        print(f"\n=== 测试场景: {scenario_name} ===")
        
        # 创建测试数据
        stock_code = f"TEST_{trend.upper()}"
        test_data = create_realistic_stock_data(stock_code, days=252, trend=trend)
        
        print(f"股票代码: {stock_code}")
        print(f"数据期间: {test_data.index.min().strftime('%Y-%m-%d')} 到 {test_data.index.max().strftime('%Y-%m-%d')}")
        print(f"价格范围: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
        print(f"总收益率: {((test_data['close'].iloc[-1] / test_data['close'].iloc[0]) - 1) * 100:.2f}%")
        
        # 测试特征提取
        features = ml_strategy.extract_features(test_data)
        if features is not None:
            print(f"特征提取成功，形状: {features.shape}")
            
            # 测试选股
            target_date = test_data.index[-1]
            selection_result = ml_strategy.select(stock_code, test_data, target_date)
            
            if selection_result:
                print(f"选股结果:")
                print(f"  - 是否选中: {selection_result.get('selected', False)}")
                print(f"  - ML预测概率: {selection_result.get('ml_probability', 0):.4f}")
                print(f"  - ML预测结果: {selection_result.get('ml_prediction', 0)}")
                print(f"  - 当前价格: {selection_result.get('close_price', 0):.2f}")
                print(f"  - RPS值: {selection_result.get('rps_value', 0):.1f}")
                print(f"  - 综合得分: {selection_result.get('score', 0):.4f}")
                
                results.append({
                    'scenario': scenario_name,
                    'trend': trend,
                    'selected': selection_result.get('selected', False),
                    'ml_probability': selection_result.get('ml_probability', 0),
                    'ml_prediction': selection_result.get('ml_prediction', 0),
                    'score': selection_result.get('score', 0),
                    'return_rate': ((test_data['close'].iloc[-1] / test_data['close'].iloc[0]) - 1) * 100
                })
            else:
                print("  - 选股失败")
        else:
            print("特征提取失败")
    
    # 汇总结果
    print("\n" + "="*50)
    print("📊 测试结果汇总")
    print("="*50)
    
    if results:
        df_results = pd.DataFrame(results)
        print("\n场景对比:")
        for _, row in df_results.iterrows():
            status = "✅ 选中" if row['selected'] else "❌ 未选中"
            print(f"{row['scenario']:8} | {status} | 概率: {row['ml_probability']:.3f} | 收益: {row['return_rate']:6.2f}% | 得分: {row['score']:.3f}")
        
        # 统计分析
        print(f"\n统计分析:")
        print(f"总测试场景: {len(results)}")
        print(f"选中场景数: {sum(df_results['selected'])}")
        print(f"选中率: {sum(df_results['selected'])/len(results)*100:.1f}%")
        print(f"平均ML概率: {df_results['ml_probability'].mean():.3f}")
        print(f"平均得分: {df_results['score'].mean():.3f}")
        
        # 按趋势分析
        selected_scenarios = df_results[df_results['selected']]
        if len(selected_scenarios) > 0:
            print(f"\n被选中场景的特征:")
            print(f"平均收益率: {selected_scenarios['return_rate'].mean():.2f}%")
            print(f"平均ML概率: {selected_scenarios['ml_probability'].mean():.3f}")
    
    print("\n🎉 全面测试完成！")
    return results

def test_batch_selection():
    """
    测试批量选股功能
    """
    print("\n" + "="*50)
    print("🔄 测试批量选股功能")
    print("="*50)
    
    # 初始化ML策略
    config = {
        'ml_threshold': 0.6,
        'use_ensemble': True,
        'feature_importance_threshold': 0.01
    }
    ml_strategy = MLEnhancedStrategy(config)
    
    # 创建多只股票数据
    stock_codes = ['000001', '000002', '000858', '002415', '600036']
    stock_data = {}
    
    trends = ['up', 'down', 'sideways', 'random', 'up']
    
    for i, code in enumerate(stock_codes):
        stock_data[code] = create_realistic_stock_data(code, days=300, trend=trends[i])
        print(f"创建股票 {code} 数据 ({trends[i]} 趋势)")
    
    # 执行批量选股 - 使用数据范围内的日期
    target_date = stock_data[stock_codes[0]].index[200]  # 使用第200个交易日作为目标日期
    print(f"\n目标选股日期: {target_date.strftime('%Y-%m-%d')}")
    
    # 对每只股票单独执行选股
    selection_results = []
    for code in stock_codes:
        if target_date in stock_data[code].index:
            result = ml_strategy.select(code, stock_data[code], target_date)
            result['code'] = code
            selection_results.append(result)
        else:
            print(f"警告: 股票 {code} 在目标日期 {target_date} 没有数据")
    
    print(f"\n批量选股结果:")
    if selection_results:
        selected_count = 0
        for result in selection_results:
            code = result.get('code', 'Unknown')
            selected = result.get('selected', False)
            probability = result.get('ml_probability', 0)
            score = result.get('score', 0)
            
            status = "✅" if selected else "❌"
            print(f"{code}: {status} | 概率: {probability:.3f} | 得分: {score:.3f}")
            
            if selected:
                selected_count += 1
        
        print(f"\n批量选股统计:")
        print(f"总股票数: {len(stock_codes)}")
        print(f"选中股票数: {selected_count}")
        print(f"选中率: {selected_count/len(stock_codes)*100:.1f}%")
    else:
        print("批量选股失败")

if __name__ == "__main__":
    try:
        # 运行场景测试
        scenario_results = test_ml_strategy_scenarios()
        
        # 运行批量选股测试
        test_batch_selection()
        
        print("\n" + "="*50)
        print("🏆 所有测试完成！ML策略功能正常")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()