#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的ML策略测试脚本
测试使用概率阈值的ML增强策略
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
    """
    # 创建日期索引（只包含工作日）
    start_date = datetime(2023, 1, 1)
    end_date = start_date + timedelta(days=int(days * 1.5))  # 考虑周末
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    business_days = date_range[date_range.weekday < 5][:days]
    
    # 设置随机种子以获得可重复的结果
    np.random.seed(hash(stock_code) % 2**32)
    
    # 基础价格
    base_price = 50.0
    
    # 根据趋势生成价格数据
    if trend == 'up':
        # 上升趋势：正向漂移
        drift = 0.0008
        volatility = 0.015
    elif trend == 'down':
        # 下降趋势：负向漂移
        drift = -0.0008
        volatility = 0.018
    elif trend == 'sideways':
        # 横盘：无漂移，低波动
        drift = 0.0001
        volatility = 0.012
    else:  # random
        # 随机：随机漂移
        drift = np.random.normal(0, 0.0003)
        volatility = 0.020
    
    # 生成价格序列（几何布朗运动）
    returns = np.random.normal(drift, volatility, days)
    prices = [base_price]
    
    for i in range(1, days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1.0))  # 确保价格不为负
    
    # 生成其他数据
    high_prices = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    low_prices = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    volumes = np.random.lognormal(15, 0.5, days)  # 对数正态分布的成交量
    
    # 创建DataFrame
    data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': volumes,
        'amount': [v * p for v, p in zip(volumes, prices)]
    }, index=business_days)
    
    # 添加RPS数据
    data['rps'] = np.random.uniform(30, 90, days)
    data['rps_value'] = data['rps']
    
    return data

class ImprovedMLStrategy(MLEnhancedStrategy):
    """
    改进的ML策略，使用概率阈值进行选股
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.probability_threshold = config.get('probability_threshold', 0.6)
        
    def select(self, stock_code: str, data: pd.DataFrame, date: pd.Timestamp):
        """
        改进的选股方法，使用概率阈值
        """
        try:
            if not self.is_trained or not hasattr(self, 'model') or self.model is None:
                # 如果模型未训练，使用模拟预测
                return self._simulate_prediction(stock_code, data, date)
            
            if date not in data.index:
                return {'selected': False, 'reason': '日期不存在'}
            
            # 截取到指定日期的数据
            date_index = data.index.get_loc(date)
            data_subset = data.iloc[:date_index + 1]
            
            # 提取特征
            features = self.extract_features(data_subset)
            
            if features is None:
                return {'selected': False, 'reason': '特征计算失败'}
            
            # 标准化特征
            if hasattr(self, 'scaler') and self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # 预测
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # 使用概率阈值决定是否选中
            prob_positive = float(probability[1])  # 正类概率
            selected = prob_positive >= self.probability_threshold
            
            # 获取基础信息
            row = data.loc[date]
            
            return {
                'selected': selected,
                'ml_probability': prob_positive,
                'ml_prediction': int(prediction),
                'close_price': row['close'],
                'rps_value': row.get('rps_value', row.get('rps', 50.0)),
                'score': prob_positive,
                'threshold': self.probability_threshold
            }
            
        except Exception as e:
            print(f"ML策略执行失败: {e}")
            return self._simulate_prediction(stock_code, data, date)
    
    def _simulate_prediction(self, stock_code: str, data: pd.DataFrame, date: pd.Timestamp):
        """
        模拟预测（当模型不可用时）
        """
        try:
            row = data.loc[date]
            
            # 基于简单规则生成模拟概率
            rps_value = row.get('rps_value', row.get('rps', 50.0))
            price_change = (row['close'] / data['close'].iloc[-20:].mean() - 1) if len(data) >= 20 else 0
            
            # 简单的评分逻辑
            score = 0.3
            if rps_value > 70:
                score += 0.2
            if price_change > 0.05:
                score += 0.3
            if row.get('volume', 0) > data['volume'].mean():
                score += 0.2
            
            # 添加随机性
            score += np.random.normal(0, 0.1)
            score = max(0, min(1, score))  # 限制在0-1之间
            
            selected = score >= self.probability_threshold
            
            return {
                'selected': selected,
                'ml_probability': score,
                'ml_prediction': 1 if selected else 0,
                'close_price': row['close'],
                'rps_value': rps_value,
                'score': score,
                'threshold': self.probability_threshold,
                'simulated': True
            }
            
        except Exception as e:
            return {
                'selected': False,
                'reason': f'模拟预测失败: {e}',
                'simulated': True
            }

def test_improved_ml_strategy():
    """
    测试改进的ML策略
    """
    print("🚀 测试改进的ML策略（使用概率阈值）\n")
    
    # 测试不同的概率阈值
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\n=== 测试概率阈值: {threshold} ===")
        
        # 初始化改进的ML策略
        config = {
            'probability_threshold': threshold,
            'use_ensemble': True,
            'feature_importance_threshold': 0.01
        }
        ml_strategy = ImprovedMLStrategy(config)
        
        # 测试不同趋势的股票
        scenarios = [
            ('强势上涨', 'up'),
            ('弱势下跌', 'down'),
            ('横盘整理', 'sideways'),
            ('随机波动', 'random')
        ]
        
        selected_count = 0
        total_count = len(scenarios)
        
        for scenario_name, trend in scenarios:
            # 创建测试数据
            stock_code = f"TEST_{trend.upper()}"
            test_data = create_realistic_stock_data(stock_code, days=200, trend=trend)
            target_date = test_data.index[150]  # 使用第150个交易日
            
            # 执行选股
            result = ml_strategy.select(stock_code, test_data, target_date)
            
            if result['selected']:
                selected_count += 1
            
            status = "✅ 选中" if result['selected'] else "❌ 未选中"
            simulated = " (模拟)" if result.get('simulated', False) else ""
            
            print(f"  {scenario_name:8} | {status} | 概率: {result['ml_probability']:.3f} | 阈值: {threshold}{simulated}")
        
        selection_rate = selected_count / total_count * 100
        print(f"\n  选中率: {selection_rate:.1f}% ({selected_count}/{total_count})")
        
        # 评估阈值效果
        if selection_rate == 0:
            print("  ⚠️  阈值过高，没有股票被选中")
        elif selection_rate == 100:
            print("  ⚠️  阈值过低，所有股票都被选中")
        else:
            print(f"  ✅ 阈值合适，选择性适中")

def test_batch_selection_with_threshold():
    """
    测试批量选股（使用概率阈值）
    """
    print("\n" + "="*60)
    print("🔄 测试批量选股（概率阈值版本）")
    print("="*60)
    
    # 使用适中的阈值
    config = {
        'probability_threshold': 0.5,
        'use_ensemble': True,
        'feature_importance_threshold': 0.01
    }
    ml_strategy = ImprovedMLStrategy(config)
    
    # 创建多只股票数据
    stock_codes = ['000001', '000002', '000858', '002415', '600036']
    stock_data = {}
    trends = ['up', 'down', 'sideways', 'random', 'up']
    
    for i, code in enumerate(stock_codes):
        stock_data[code] = create_realistic_stock_data(code, days=200, trend=trends[i])
        print(f"创建股票 {code} 数据 ({trends[i]} 趋势)")
    
    # 执行批量选股
    target_date = stock_data[stock_codes[0]].index[150]
    print(f"\n目标选股日期: {target_date.strftime('%Y-%m-%d')}")
    print(f"概率阈值: {config['probability_threshold']}")
    
    selection_results = []
    for code in stock_codes:
        if target_date in stock_data[code].index:
            result = ml_strategy.select(code, stock_data[code], target_date)
            result['code'] = code
            selection_results.append(result)
    
    print(f"\n批量选股结果:")
    selected_count = 0
    for result in selection_results:
        status = "✅" if result['selected'] else "❌"
        simulated = " (模拟)" if result.get('simulated', False) else ""
        print(f"{result['code']}: {status} | 概率: {result['ml_probability']:.3f} | 得分: {result['score']:.3f}{simulated}")
        if result['selected']:
            selected_count += 1
    
    print(f"\n批量选股统计:")
    print(f"总股票数: {len(selection_results)}")
    print(f"选中股票数: {selected_count}")
    print(f"选中率: {selected_count/len(selection_results)*100:.1f}%")
    
    if selected_count > 0:
        print(f"\n✅ 成功选出 {selected_count} 只股票！")
    else:
        print(f"\n⚠️  没有股票被选中，可能需要调整阈值")

if __name__ == "__main__":
    try:
        # 运行改进的策略测试
        test_improved_ml_strategy()
        
        # 运行批量选股测试
        test_batch_selection_with_threshold()
        
        print("\n" + "="*60)
        print("🎉 改进的ML策略测试完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()