#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速模型演示脚本

这是一个简化的演示脚本，展示如何快速使用模型组合进行股票预测
适合初学者快速上手
"""

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入我们的预测器
from practical_model_usage import StockPredictor

def quick_demo():
    """快速演示"""
    print("🚀 股票预测模型快速演示")
    print("=" * 50)
    
    # 1. 初始化预测器
    print("📊 正在加载模型...")
    predictor = StockPredictor()
    
    if not predictor.load_best_models():
        print("❌ 模型加载失败，请先运行 model_compatibility_test.py 创建演示模型")
        return
    
    print("✅ 模型加载成功！")
    
    # 2. 演示不同类型的预测
    demo_stocks = [
        {"code": "000001.SZ", "name": "平安银行"},
        {"code": "600000.SH", "name": "浦发银行"},
        {"code": "000002.SZ", "name": "万科A"}
    ]
    
    print(f"\n📈 开始分析 {len(demo_stocks)} 只股票...")
    
    results = []
    for stock in demo_stocks:
        print(f"\n{'='*60}")
        print(f"🏢 分析股票：{stock['name']} ({stock['code']})")
        print(f"{'='*60}")
        
        # 生成模拟数据（实际使用时替换为真实数据）
        mock_data = generate_mock_stock_data()
        
        # 执行预测
        result = predictor.predict_stock_analysis(stock['code'], mock_data)
        
        # 保存结果用于后续排序
        if result:
            score = calculate_simple_score(result)
            results.append({
                'code': stock['code'],
                'name': stock['name'],
                'score': score,
                'result': result
            })
    
    # 3. 生成投资建议排序
    print(f"\n\n🏆 投资建议排序")
    print("=" * 50)
    
    if results:
        # 按评分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print("排名 | 股票代码 | 股票名称 | 综合评分 | 建议")
        print("-" * 55)
        
        for i, item in enumerate(results, 1):
            recommendation = get_recommendation(item['score'])
            print(f"{i:2d}   | {item['code']:8s} | {item['name']:8s} | {item['score']:6.2f}   | {recommendation}")
        
        # 详细分析最佳股票
        best_stock = results[0]
        print(f"\n🎯 最佳选择详细分析：{best_stock['name']} ({best_stock['code']})")
        print("-" * 50)
        show_detailed_analysis(best_stock['result'])
    
    print("\n✨ 演示完成！")
    print("\n📝 使用提示：")
    print("1. 这是演示数据，实际使用时请替换为真实股票数据")
    print("2. 模型预测仅供参考，投资需谨慎")
    print("3. 建议结合基本面分析和市场环境")
    print("4. 可以修改 generate_mock_stock_data() 函数来测试不同的数据")

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

def calculate_simple_score(prediction_result):
    """计算简单的投资评分"""
    score = 0.0
    
    # 短期趋势评分（0-3分）
    if 'short_term_trend' in prediction_result and prediction_result['short_term_trend']:
        trend_result = prediction_result['short_term_trend']
        trend_pred = int(trend_result['prediction'])
        confidence = trend_result.get('confidence', 0)
        
        if trend_pred == 2:  # 上涨
            score += 3 * confidence
        elif trend_pred == 1:  # 震荡
            score += 1 * confidence
        # 下跌不加分
    
    # 高收益概率评分（0-4分）
    if 'high_return_probability' in prediction_result and prediction_result['high_return_probability']:
        prob_result = prediction_result['high_return_probability']
        if 'probabilities' in prob_result and len(prob_result['probabilities']) > 1:
            high_return_prob = prob_result['probabilities'][1]
            score += 4 * high_return_prob
    
    # 预期收益评分（0-3分）
    if 'expected_return' in prediction_result and prediction_result['expected_return']:
        expected_return = prediction_result['expected_return']['prediction']
        if expected_return > 0.08:  # >8%
            score += 3
        elif expected_return > 0.05:  # >5%
            score += 2
        elif expected_return > 0:  # >0%
            score += 1
    
    return min(score, 10)  # 最高10分

def get_recommendation(score):
    """根据评分获取投资建议"""
    if score >= 7:
        return "🟢 强烈推荐"
    elif score >= 5:
        return "🟡 推荐"
    elif score >= 3:
        return "🟠 观望"
    else:
        return "🔴 回避"

def show_detailed_analysis(result):
    """显示详细分析"""
    # 趋势分析
    if 'short_term_trend' in result and result['short_term_trend']:
        trend_result = result['short_term_trend']
        trend_names = ['📉 下跌', '📊 震荡', '📈 上涨']
        trend_pred = int(trend_result['prediction'])
        confidence = trend_result.get('confidence', 0)
        print(f"趋势预测: {trend_names[trend_pred]} (置信度: {confidence:.2f})")
    
    # 收益概率
    if 'high_return_probability' in result and result['high_return_probability']:
        prob_result = result['high_return_probability']
        if 'probabilities' in prob_result and len(prob_result['probabilities']) > 1:
            high_return_prob = prob_result['probabilities'][1]
            print(f"高收益概率: {high_return_prob:.2f} (>8%收益的可能性)")
    
    # 预期收益
    if 'expected_return' in result and result['expected_return']:
        expected_return = result['expected_return']['prediction']
        print(f"预期收益率: {expected_return:.2f} ({expected_return*100:.1f}%)")
    
    # 投资建议
    score = calculate_simple_score(result)
    recommendation = get_recommendation(score)
    print(f"\n💡 综合建议: {recommendation} (评分: {score:.1f}/10)")

def interactive_demo():
    """交互式演示"""
    print("🎮 交互式股票预测演示")
    print("=" * 50)
    
    predictor = StockPredictor()
    if not predictor.load_best_models():
        print("❌ 模型加载失败")
        return
    
    while True:
        print("\n请选择操作：")
        print("1. 分析单只股票")
        print("2. 批量分析对比")
        print("3. 查看模型信息")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            stock_code = input("请输入股票代码 (如 000001.SZ): ").strip()
            if stock_code:
                mock_data = generate_mock_stock_data()
                result = predictor.predict_stock_analysis(stock_code, mock_data)
                if result:
                    print(f"\n📊 {stock_code} 详细分析：")
                    show_detailed_analysis(result)
        
        elif choice == '2':
            print("\n批量分析演示（使用预设股票）...")
            quick_demo()
            break
        
        elif choice == '3':
            print("\n📋 已加载的模型信息：")
            for task_name, model_info in predictor.best_models.items():
                print(f"- {task_name}: {model_info['description']}")
        
        elif choice == '4':
            print("👋 再见！")
            break
        
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_demo()
    else:
        quick_demo()