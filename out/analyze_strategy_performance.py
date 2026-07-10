#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略表现分析脚本
"""

import json
import os

def analyze_strategy_performance():
    """分析当前策略表现"""
    
    # 读取分析结果
    results_file = 'analysis_results_v2.json'
    if not os.path.exists(results_file):
        print(f"错误: 找不到分析结果文件 {results_file}")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== 当前策略表现分析 ===")
    print(f"分析股票总数: {data['summary']['total_stocks']}")
    print(f"分析时间段: {data['summary']['analysis_period']['total_days']}天")
    print(f"分析期间: {data['summary']['analysis_period']['start_date'][:10]} 到 {data['summary']['analysis_period']['end_date'][:10]}")
    
    print("\n各策略表现:")
    strategies = data['summary']['strategies']
    
    for strategy, metrics in strategies.items():
        print(f"\n{strategy.upper()}策略:")
        print(f"  平均选中率: {metrics['avg_selection_rate']:.2f}%")
        print(f"  最大选中率: {metrics['max_selection_rate']:.2f}%")
        print(f"  平均评分: {metrics['avg_score']:.3f}")
        print(f"  选中股票数: {metrics['selected_stocks']}")
        print(f"  高质量股票数: {metrics['high_quality_stocks']}")
        print(f"  选中率标准差: {metrics['std_selection_rate']:.2f}%")
    
    # 策略对比分析
    print("\n=== 策略对比分析 ===")
    
    # 按平均选中率排序
    sorted_by_selection = sorted(strategies.items(), key=lambda x: x[1]['avg_selection_rate'], reverse=True)
    print("\n按平均选中率排序:")
    for i, (strategy, metrics) in enumerate(sorted_by_selection, 1):
        print(f"{i}. {strategy}: {metrics['avg_selection_rate']:.2f}%")
    
    # 按平均评分排序
    sorted_by_score = sorted(strategies.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    print("\n按平均评分排序:")
    for i, (strategy, metrics) in enumerate(sorted_by_score, 1):
        print(f"{i}. {strategy}: {metrics['avg_score']:.3f}")
    
    # 按选中股票数排序
    sorted_by_count = sorted(strategies.items(), key=lambda x: x[1]['selected_stocks'], reverse=True)
    print("\n按选中股票数排序:")
    for i, (strategy, metrics) in enumerate(sorted_by_count, 1):
        print(f"{i}. {strategy}: {metrics['selected_stocks']}只")
    
    # 优化建议
    print("\n=== 优化建议 ===")
    
    # 检查各策略的问题
    for strategy, metrics in strategies.items():
        print(f"\n{strategy.upper()}策略分析:")
        
        if metrics['avg_selection_rate'] < 1.0:
            print(f"  ⚠️  选中率过低 ({metrics['avg_selection_rate']:.2f}%)，可能过于严格")
        elif metrics['avg_selection_rate'] > 20.0:
            print(f"  ⚠️  选中率过高 ({metrics['avg_selection_rate']:.2f}%)，可能过于宽松")
        else:
            print(f"  ✅ 选中率适中 ({metrics['avg_selection_rate']:.2f}%)")
        
        if metrics['high_quality_stocks'] == 0:
            print(f"  ⚠️  没有高质量股票，策略可能需要调整")
        else:
            print(f"  ✅ 发现 {metrics['high_quality_stocks']} 只高质量股票")
        
        if metrics['std_selection_rate'] > 10.0:
            print(f"  ⚠️  选中率波动较大 (标准差: {metrics['std_selection_rate']:.2f}%)，策略稳定性有待提高")
        else:
            print(f"  ✅ 选中率波动适中 (标准差: {metrics['std_selection_rate']:.2f}%)")
    
    print("\n=== 总体评估 ===")
    total_selected = sum(metrics['selected_stocks'] for metrics in strategies.values())
    total_stocks = data['summary']['total_stocks']
    overall_selection_rate = (total_selected / total_stocks / len(strategies)) * 100
    
    print(f"总体平均选中率: {overall_selection_rate:.2f}%")
    print(f"策略覆盖度: {len([s for s in strategies.values() if s['selected_stocks'] > 0])}/{len(strategies)} 个策略有效")
    
    # 具体优化建议
    print("\n=== 具体优化建议 ===")
    print("1. 参数调优:")
    print("   - Basic策略选中率较高，可适当提高阈值")
    print("   - Advanced和Momentum策略选中率较低，可适当降低阈值")
    print("   - ML策略选中率极低，需要重新训练模型或调整特征")
    
    print("\n2. 策略组合:")
    print("   - 可以考虑多策略加权组合")
    print("   - 根据市场环境动态调整策略权重")
    
    print("\n3. 风险控制:")
    print("   - 增加止损机制")
    print("   - 添加仓位管理")
    print("   - 考虑市场情绪指标")

if __name__ == "__main__":
    analyze_strategy_performance()