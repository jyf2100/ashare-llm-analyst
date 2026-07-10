#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终ML策略测试脚本
强制使用模拟预测来验证选股逻辑
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
        drift = 0.0015
        volatility = 0.015
    elif trend == 'down':
        # 下降趋势：负向漂移
        drift = -0.0012
        volatility = 0.018
    elif trend == 'sideways':
        # 横盘：无漂移，低波动
        drift = 0.0001
        volatility = 0.012
    else:  # random
        # 随机：随机漂移
        drift = np.random.normal(0, 0.0005)
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
    
    # 添加RPS数据（根据趋势调整）
    if trend == 'up':
        base_rps = 75
        rps_noise = 10
    elif trend == 'down':
        base_rps = 35
        rps_noise = 15
    elif trend == 'sideways':
        base_rps = 50
        rps_noise = 8
    else:  # random
        base_rps = 55
        rps_noise = 20
    
    data['rps'] = np.random.normal(base_rps, rps_noise, days)
    data['rps'] = np.clip(data['rps'], 1, 99)  # 限制在1-99之间
    data['rps_value'] = data['rps']
    
    return data

class SimulatedMLStrategy(MLEnhancedStrategy):
    """
    模拟ML策略，强制使用智能模拟预测
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.probability_threshold = config.get('probability_threshold', 0.6)
        self.force_simulation = config.get('force_simulation', True)
        
    def select(self, stock_code: str, data: pd.DataFrame, date: pd.Timestamp):
        """
        强制使用模拟预测的选股方法
        """
        try:
            if date not in data.index:
                return {'selected': False, 'reason': '日期不存在'}
            
            # 强制使用模拟预测
            if self.force_simulation:
                return self._intelligent_simulation(stock_code, data, date)
            
            # 原始ML预测逻辑（如果需要）
            return super().select(stock_code, data, date)
            
        except Exception as e:
            print(f"选股失败: {e}")
            return self._intelligent_simulation(stock_code, data, date)
    
    def _intelligent_simulation(self, stock_code: str, data: pd.DataFrame, date: pd.Timestamp):
        """
        智能模拟预测（基于技术指标和趋势分析）
        """
        try:
            # 获取当前日期的数据
            date_index = data.index.get_loc(date)
            current_data = data.iloc[:date_index + 1]
            row = current_data.iloc[-1]
            
            # 计算技术指标
            score = 0.0
            factors = []
            
            # 1. RPS评分 (权重: 0.25)
            rps_value = row.get('rps_value', row.get('rps', 50.0))
            if rps_value > 80:
                rps_score = 0.9
            elif rps_value > 70:
                rps_score = 0.7
            elif rps_value > 60:
                rps_score = 0.5
            elif rps_value > 40:
                rps_score = 0.3
            else:
                rps_score = 0.1
            
            score += rps_score * 0.25
            factors.append(f"RPS({rps_value:.1f}): {rps_score:.2f}")
            
            # 2. 价格趋势评分 (权重: 0.3)
            if len(current_data) >= 20:
                recent_prices = current_data['close'].tail(20)
                price_trend = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1)
                
                if price_trend > 0.15:  # 上涨超过15%
                    trend_score = 0.9
                elif price_trend > 0.05:  # 上涨超过5%
                    trend_score = 0.7
                elif price_trend > -0.05:  # 小幅波动
                    trend_score = 0.4
                elif price_trend > -0.15:  # 下跌不超过15%
                    trend_score = 0.2
                else:  # 大幅下跌
                    trend_score = 0.1
                
                score += trend_score * 0.3
                factors.append(f"趋势({price_trend*100:.1f}%): {trend_score:.2f}")
            else:
                score += 0.5 * 0.3  # 默认中性评分
                factors.append("趋势(数据不足): 0.50")
            
            # 3. 成交量评分 (权重: 0.2)
            if len(current_data) >= 10:
                recent_volume = current_data['volume'].tail(5).mean()
                avg_volume = current_data['volume'].tail(20).mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio > 2.0:  # 成交量放大2倍以上
                    volume_score = 0.9
                elif volume_ratio > 1.5:  # 成交量放大1.5倍以上
                    volume_score = 0.7
                elif volume_ratio > 1.2:  # 成交量略有放大
                    volume_score = 0.6
                elif volume_ratio > 0.8:  # 成交量正常
                    volume_score = 0.4
                else:  # 成交量萎缩
                    volume_score = 0.2
                
                score += volume_score * 0.2
                factors.append(f"成交量({volume_ratio:.1f}x): {volume_score:.2f}")
            else:
                score += 0.5 * 0.2
                factors.append("成交量(数据不足): 0.50")
            
            # 4. 价格位置评分 (权重: 0.25)
            if len(current_data) >= 60:
                high_60d = current_data['high'].tail(60).max()
                low_60d = current_data['low'].tail(60).min()
                current_price = row['close']
                
                if high_60d > low_60d:
                    price_position = (current_price - low_60d) / (high_60d - low_60d)
                    
                    if price_position > 0.8:  # 接近60日高点
                        position_score = 0.9
                    elif price_position > 0.6:  # 中高位
                        position_score = 0.7
                    elif price_position > 0.4:  # 中位
                        position_score = 0.5
                    elif price_position > 0.2:  # 中低位
                        position_score = 0.3
                    else:  # 接近60日低点
                        position_score = 0.1
                else:
                    position_score = 0.5
                
                score += position_score * 0.25
                factors.append(f"价格位置({price_position*100:.1f}%): {position_score:.2f}")
            else:
                score += 0.5 * 0.25
                factors.append("价格位置(数据不足): 0.50")
            
            # 添加一些随机性（模拟模型的不确定性）
            noise = np.random.normal(0, 0.05)
            score = max(0, min(1, score + noise))
            
            # 根据阈值决定是否选中
            selected = score >= self.probability_threshold
            
            return {
                'selected': selected,
                'ml_probability': score,
                'ml_prediction': 1 if selected else 0,
                'close_price': row['close'],
                'rps_value': rps_value,
                'score': score,
                'threshold': self.probability_threshold,
                'simulated': True,
                'factors': factors
            }
            
        except Exception as e:
            return {
                'selected': False,
                'reason': f'智能模拟失败: {e}',
                'simulated': True
            }

def test_simulated_ml_strategy():
    """
    测试模拟ML策略
    """
    print("🎯 测试智能模拟ML策略\n")
    
    # 测试不同的概率阈值
    thresholds = [0.4, 0.6, 0.8]
    
    for threshold in thresholds:
        print(f"\n=== 测试概率阈值: {threshold} ===")
        
        # 初始化模拟ML策略
        config = {
            'probability_threshold': threshold,
            'force_simulation': True
        }
        ml_strategy = SimulatedMLStrategy(config)
        
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
            
            print(f"  {scenario_name:8} | {status} | 概率: {result['ml_probability']:.3f} | RPS: {result.get('rps_value', 0):.1f}")
            
            # 显示评分因子（仅对选中的股票）
            if result['selected'] and 'factors' in result:
                print(f"    评分因子: {', '.join(result['factors'])}")
        
        selection_rate = selected_count / total_count * 100
        print(f"\n  选中率: {selection_rate:.1f}% ({selected_count}/{total_count})")
        
        # 评估阈值效果
        if selection_rate == 0:
            print("  ⚠️  阈值过高，没有股票被选中")
        elif selection_rate == 100:
            print("  ⚠️  阈值过低，所有股票都被选中")
        else:
            print(f"  ✅ 阈值合适，选择性适中")

def test_batch_selection_simulated():
    """
    测试批量选股（模拟版本）
    """
    print("\n" + "="*60)
    print("🔄 测试批量选股（智能模拟版本）")
    print("="*60)
    
    # 使用适中的阈值
    config = {
        'probability_threshold': 0.6,
        'force_simulation': True
    }
    ml_strategy = SimulatedMLStrategy(config)
    
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
            result['trend'] = trends[stock_codes.index(code)]
            selection_results.append(result)
    
    print(f"\n批量选股结果:")
    selected_count = 0
    for result in selection_results:
        status = "✅" if result['selected'] else "❌"
        trend_info = f"({result['trend']})"
        print(f"{result['code']} {trend_info:12} | {status} | 概率: {result['ml_probability']:.3f} | RPS: {result.get('rps_value', 0):.1f}")
        
        if result['selected']:
            selected_count += 1
            # 显示选中股票的详细信息
            if 'factors' in result:
                print(f"    评分详情: {', '.join(result['factors'])}")
    
    print(f"\n批量选股统计:")
    print(f"总股票数: {len(selection_results)}")
    print(f"选中股票数: {selected_count}")
    print(f"选中率: {selected_count/len(selection_results)*100:.1f}%")
    
    if selected_count > 0:
        print(f"\n✅ 成功选出 {selected_count} 只股票！")
        selected_stocks = [r for r in selection_results if r['selected']]
        print("选中的股票:")
        for stock in selected_stocks:
            print(f"  - {stock['code']} ({stock['trend']} 趋势): 概率 {stock['ml_probability']:.3f}, RPS {stock.get('rps_value', 0):.1f}")
    else:
        print(f"\n⚠️  没有股票被选中，可能需要调整阈值")

if __name__ == "__main__":
    try:
        # 运行模拟策略测试
        test_simulated_ml_strategy()
        
        # 运行批量选股测试
        test_batch_selection_simulated()
        
        print("\n" + "="*60)
        print("🎉 智能模拟ML策略测试完成！")
        print("说明: 此测试使用智能模拟预测，基于技术指标和趋势分析")
        print("包含: RPS强度、价格趋势、成交量变化、价格位置等因子")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()