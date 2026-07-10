#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的ML策略测试脚本
测试MLEnhancedStrategy类的基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入ML策略
import importlib.util
spec = importlib.util.spec_from_file_location("optimized_stock_selection_v2", "-02-optimized_stock_selection_v2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MLEnhancedStrategy = module.MLEnhancedStrategy

def create_sample_data(days=100):
    """创建示例股票数据"""
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # 创建模拟股票数据
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, days)  # 日收益率
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 创建成交量数据
    volumes = np.random.randint(1000000, 10000000, days)
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.03)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.03)) for p in prices],
        'close': prices,
        'volume': volumes,
        'amount': [p * v for p, v in zip(prices, volumes)]
    })
    
    data.set_index('date', inplace=True)
    return data

def test_ml_strategy_basic():
    """测试ML策略的基本功能"""
    print("\n=== 测试ML策略基本功能 ===")
    
    # 创建配置
    config = {
        'rsi_threshold': 30,
        'volume_threshold': 1.5,
        'price_change_threshold': 0.05
    }
    
    try:
        # 初始化ML策略
        print("1. 初始化ML策略...")
        ml_config = {
            'ml_threshold': 0.6,
            'use_ensemble': True,
            'feature_importance_threshold': 0.01
        }
        ml_strategy = MLEnhancedStrategy(ml_config)
        print(f"   - 模型已训练: {ml_strategy.is_trained}")
        print(f"   - 标准化器可用: {ml_strategy.scaler is not None}")
        
        # 创建测试数据
        print("\n2. 创建测试数据...")
        test_data = create_sample_data(100)
        print(f"   - 数据形状: {test_data.shape}")
        print(f"   - 日期范围: {test_data.index[0]} 到 {test_data.index[-1]}")
        
        # 测试特征提取
        print("\n3. 测试特征提取...")
        features = ml_strategy.extract_features(test_data)
        if features is not None:
            print(f"   - 特征形状: {features.shape}")
            if hasattr(features, 'columns'):
                print(f"   - 特征列: {list(features.columns)}")
            else:
                print(f"   - 特征类型: {type(features)}")
        else:
            print("   - 特征提取失败")
        
        # 测试选股功能
        print("\n4. 测试选股功能...")
        test_date = test_data.index[-10]  # 选择倒数第10天
        result = ml_strategy.select('TEST001', test_data, test_date)
        
        print(f"   - 选股结果: {result}")
        print(f"   - 是否选中: {result.get('selected', False)}")
        print(f"   - 选择原因: {result.get('reason', 'N/A')}")
        
        if 'ml_probability' in result:
            print(f"   - ML预测概率: {result['ml_probability']}")
        
        print("\n✅ ML策略基本功能测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """测试模型加载功能"""
    print("\n=== 测试模型加载功能 ===")
    
    # 检查模型文件是否存在
    models_dir = 'models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.startswith('ml_model_') and f.endswith('.pkl')]
        scaler_files = [f for f in os.listdir(models_dir) if f.startswith('scaler_') and f.endswith('.pkl')]
        
        print(f"找到模型文件: {len(model_files)} 个")
        print(f"找到标准化器文件: {len(scaler_files)} 个")
        
        if model_files:
            print(f"最新模型文件: {sorted(model_files)[-1]}")
        if scaler_files:
            print(f"最新标准化器文件: {sorted(scaler_files)[-1]}")
    else:
        print("models目录不存在")

def main():
    """主函数"""
    print("🧪 开始ML策略简单测试")
    
    # 测试模型加载
    test_model_loading()
    
    # 测试基本功能
    success = test_ml_strategy_basic()
    
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n💥 测试失败！")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())