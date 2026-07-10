#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资组合优化演示脚本
基于模型组合使用指南中的投资组合优化方法
"""

import sys
import os
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from practical_model_usage import StockPredictor

def generate_mock_stock_data():
    """生成模拟股票数据"""
    # 生成100天的模拟数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # 模拟价格走势
    base_price = 10.0
    price_changes = np.random.randn(100) * 0.02  # 2%的日波动
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # 价格不能为负
    
    # 创建DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': [p * (1 + np.random.randn() * 0.005) for p in prices],
        'high': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
        'low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    return data

def calculate_investment_score(result: Dict[str, any]) -> float:
    """
    计算投资评分
    
    Args:
        result: 预测结果字典，包含各个任务的预测结果
        
    Returns:
        投资评分 (0-10分)
    """
    if not result:
        return 0.0
    
    score = 0.0
    
    # 趋势评分（权重：30%）
    if 'short_term_trend' in result and result['short_term_trend']:
        trend_result = result['short_term_trend']
        if isinstance(trend_result, dict):
            trend = trend_result.get('prediction', 0)
            confidence = trend_result.get('confidence', 0)
            if trend == 2:  # 上涨
                score += 3 * confidence
            elif trend == 1:  # 震荡
                score += 1 * confidence
            # 下跌不加分
    
    # 收益概率评分（权重：40%）
    if 'high_return_probability' in result and result['high_return_probability']:
        prob_result = result['high_return_probability']
        if isinstance(prob_result, dict) and 'probabilities' in prob_result:
            probs = prob_result['probabilities']
            if isinstance(probs, list) and len(probs) > 1:
                high_return_prob = probs[1]  # 正类概率
                score += 4 * high_return_prob
    
    # 预期收益评分（权重：30%）
    if 'expected_return' in result and result['expected_return']:
        return_result = result['expected_return']
        if isinstance(return_result, dict):
            expected_return = return_result.get('prediction', 0)
            if expected_return > 0.05:  # >5%
                score += 3
            elif expected_return > 0:
                score += 1
    
    return min(score, 10.0)  # 最高10分

def select_best_stocks(batch_results: Dict[str, Dict[str, float]], top_n: int = 10) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    从批量分析结果中选择最佳股票
    
    参数:
    batch_results: Dict[str, Dict[str, float]], 批量预测结果
    top_n: int, 返回前N只股票
    
    返回:
    List[Tuple[str, float, Dict[str, float]]]: (股票代码, 评分, 预测结果)
    """
    scored_stocks = []
    
    for code, result in batch_results.items():
        if result:
            score = calculate_investment_score(result)
            scored_stocks.append((code, score, result))
    
    # 按评分排序
    scored_stocks.sort(key=lambda x: x[1], reverse=True)
    
    return scored_stocks[:top_n]

def print_investment_recommendations(best_stocks: List[Tuple[str, float, Dict[str, float]]]):
    """
    打印投资建议
    
    参数:
    best_stocks: List[Tuple[str, float, Dict[str, float]]], 最佳股票列表
    """
    print("\n" + "="*80)
    print("💎 投资组合优化建议")
    print("="*80)
    
    if not best_stocks:
        print("❌ 没有找到符合条件的投资标的")
        return
    
    print(f"\n🏆 推荐投资组合 (前{len(best_stocks)}只股票):")
    print("-" * 80)
    print(f"{'排名':<4} {'股票代码':<12} {'投资评分':<10} {'预期收益率':<12} {'置信度':<10} {'投资建议':<10}")
    print("-" * 80)
    
    for i, (code, score, result) in enumerate(best_stocks, 1):
        predicted_return = result.get('predicted_return', 0)
        confidence = result.get('confidence', 0)
        
        # 生成投资建议
        if score > 1.0 and predicted_return > 3:
            recommendation = "强烈买入"
        elif score > 0.5 and predicted_return > 1:
            recommendation = "买入"
        elif score > 0 and predicted_return > 0:
            recommendation = "关注"
        else:
            recommendation = "回避"
        
        print(f"{i:<4} {code:<12} {score:<10.3f} {predicted_return:<12.3f}% {confidence:<10.3f} {recommendation:<10}")
    
    # 投资组合统计
    total_stocks = len(best_stocks)
    avg_return = sum(result['predicted_return'] for _, _, result in best_stocks) / total_stocks
    avg_confidence = sum(result['confidence'] for _, _, result in best_stocks) / total_stocks
    avg_score = sum(score for _, score, _ in best_stocks) / total_stocks
    
    print("-" * 80)
    print(f"📊 投资组合统计:")
    print(f"   总股票数: {total_stocks}")
    print(f"   平均预期收益率: {avg_return:.3f}%")
    print(f"   平均置信度: {avg_confidence:.3f}")
    print(f"   平均投资评分: {avg_score:.3f}")
    
    # 风险提示
    high_risk_count = sum(1 for _, _, result in best_stocks if result['confidence'] < 0.5)
    if high_risk_count > 0:
        print(f"\n⚠️  风险提示: {high_risk_count} 只股票置信度较低，请谨慎投资")
    
    print("="*80)

def main():
    """
    主函数：演示投资组合优化功能
    """
    print("🚀 投资组合优化演示")
    print("基于模型组合使用指南中的投资组合优化方法")
    
    # 创建预测器
    predictor = StockPredictor()
    
    # 示例股票列表
    stock_codes = [
        '000001.SZ',  # 平安银行
        '000002.SZ',  # 万科A
        '600000.SH',  # 浦发银行
        '600036.SH',  # 招商银行
        '000858.SZ',  # 五粮液
        '600519.SH',  # 贵州茅台
        '000725.SZ',  # 京东方A
        '002415.SZ',  # 海康威视
    ]
    
    print(f"\n📊 分析股票列表: {stock_codes}")
    
    # 批量预测
    print("\n🔄 正在进行批量预测...")
    batch_results = {}
    
    for code in stock_codes:
        try:
            # 生成模拟股票数据（实际使用时替换为真实数据）
            mock_data = generate_mock_stock_data()
            result = predictor.predict_stock_analysis(code, mock_data)
            batch_results[code] = result
            print(f"  ✅ {code}: 预测完成，结果={result is not None}")
            if result:
                print(f"    结果类型: {type(result)}, 键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        except Exception as e:
            print(f"  ❌ {code}: 预测失败 - {e}")
            batch_results[code] = None
    
    # 显示原始预测结果
    print("\n📈 原始预测结果:")
    print(f"batch_results类型: {type(batch_results)}, 长度: {len(batch_results)}")
    for code, result in batch_results.items():
        print(f"  {code}: result={result}, type={type(result)}, bool={bool(result)}")
        if result:
            # 检查结果格式并提取信息
            print(f"    预测成功，结果类型={type(result)}")
            if isinstance(result, dict):
                print(f"    结果键: {list(result.keys())}")
                # 显示每个键的值
                for key, value in result.items():
                    print(f"      {key}: {value}")
        else:
            print(f"    预测失败或结果为空")
    
    # 投资组合优化
    print("\n🎯 开始投资组合优化...")
    best_stocks = select_best_stocks(batch_results, top_n=5)
    
    # 打印投资建议
    print_investment_recommendations(best_stocks)
    
    # 使用说明
    print("\n💡 使用说明:")
    print("1. 投资评分 = 预期收益率 × 置信度 × 风险调整系数")
    print("2. 评分越高，投资价值越大")
    print("3. 建议优先关注评分 > 0.5 且预期收益率 > 1% 的股票")
    print("4. 置信度 < 0.5 的股票风险较高，请谨慎投资")
    print("5. 本演示基于历史数据训练的模型，仅供参考，不构成投资建议")

if __name__ == "__main__":
    main()