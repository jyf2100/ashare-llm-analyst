#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试RPS数据质量
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_rps_data():
    """分析RPS数据质量"""
    print("RPS数据质量分析")
    print("=" * 50)
    
    # 1. 检查RPS原始数据
    rps_dir = "rps_results"
    if not os.path.exists(rps_dir):
        print(f"❌ RPS目录不存在: {rps_dir}")
        return
    
    rps_files = [f for f in os.listdir(rps_dir) if f.endswith('.pkl')]
    print(f"📁 找到 {len(rps_files)} 个RPS文件")
    
    for rps_file in rps_files:
        print(f"\n📊 分析文件: {rps_file}")
        file_path = os.path.join(rps_dir, rps_file)
        
        try:
            with open(file_path, 'rb') as f:
                rps_data = pickle.load(f)
            
            print(f"  股票数量: {len(rps_data)}")
            
            # 分析RPS数据结构
            sample_stocks = list(rps_data.keys())[:5]
            for stock_code in sample_stocks:
                stock_rps = rps_data[stock_code]
                print(f"  {stock_code}: {type(stock_rps)}, 长度: {len(stock_rps) if hasattr(stock_rps, '__len__') else 'N/A'}")
                
                if isinstance(stock_rps, list) and len(stock_rps) > 0:
                    sample_item = stock_rps[0]
                    print(f"    样本数据: {sample_item}")
                    
                    # 提取RPS值进行统计
                    rps_values = []
                    for item in stock_rps:
                        if isinstance(item, dict) and 'rps' in item:
                            rps_values.append(float(item['rps']))
                    
                    if rps_values:
                        print(f"    RPS统计: 最小值={min(rps_values):.2f}, 最大值={max(rps_values):.2f}, 平均值={np.mean(rps_values):.2f}")
                        print(f"    RPS分布: {np.percentile(rps_values, [25, 50, 75])}")
                        
                        # 检查是否所有值都相同
                        unique_values = len(set(rps_values))
                        print(f"    唯一值数量: {unique_values}")
                        if unique_values == 1:
                            print(f"    ⚠️  警告: 所有RPS值都相同 ({rps_values[0]})")
                
        except Exception as e:
            print(f"  ❌ 读取文件失败: {e}")

def analyze_training_data_rps():
    """分析训练数据中的RPS特征"""
    print("\n训练数据中的RPS特征分析")
    print("=" * 50)
    
    # 查找最新的训练数据
    training_dir = "training_data"
    if not os.path.exists(training_dir):
        print(f"❌ 训练数据目录不存在: {training_dir}")
        return
    
    feature_files = [f for f in os.listdir(training_dir) if f.startswith('features_') and f.endswith('.npy')]
    if not feature_files:
        print("❌ 未找到特征文件")
        return
    
    # 使用最新的特征文件
    latest_feature_file = sorted(feature_files)[-1]
    feature_path = os.path.join(training_dir, latest_feature_file)
    
    print(f"📊 分析特征文件: {latest_feature_file}")
    
    try:
        X = np.load(feature_path)
        print(f"特征矩阵形状: {X.shape}")
        
        # RPS是第11个特征（索引10）
        rps_feature = X[:, 10]  # RPS特征列
        
        print(f"\nRPS特征统计:")
        print(f"  样本数: {len(rps_feature)}")
        print(f"  最小值: {np.min(rps_feature):.4f}")
        print(f"  最大值: {np.max(rps_feature):.4f}")
        print(f"  平均值: {np.mean(rps_feature):.4f}")
        print(f"  标准差: {np.std(rps_feature):.4f}")
        print(f"  分位数: {np.percentile(rps_feature, [25, 50, 75])}")
        
        # 检查唯一值
        unique_values = np.unique(rps_feature)
        print(f"  唯一值数量: {len(unique_values)}")
        print(f"  前10个唯一值: {unique_values[:10]}")
        
        # 检查是否大部分值都是默认值
        default_count = np.sum(rps_feature == 50.0)
        negative_count = np.sum(rps_feature == -99.0)
        print(f"  默认值(50.0)数量: {default_count} ({default_count/len(rps_feature)*100:.2f}%)")
        print(f"  缺失值(-99.0)数量: {negative_count} ({negative_count/len(rps_feature)*100:.2f}%)")
        
        if default_count > len(rps_feature) * 0.8:
            print("  ⚠️  警告: 超过80%的RPS值是默认值，可能存在数据质量问题")
        
        if negative_count > len(rps_feature) * 0.1:
            print("  ⚠️  警告: 超过10%的RPS值是缺失值")
            
        # 分析RPS值的分布
        valid_rps = rps_feature[(rps_feature != 50.0) & (rps_feature != -99.0)]
        if len(valid_rps) > 0:
            print(f"\n有效RPS值分析 (排除默认值和缺失值):")
            print(f"  有效样本数: {len(valid_rps)} ({len(valid_rps)/len(rps_feature)*100:.2f}%)")
            print(f"  最小值: {np.min(valid_rps):.4f}")
            print(f"  最大值: {np.max(valid_rps):.4f}")
            print(f"  平均值: {np.mean(valid_rps):.4f}")
            print(f"  标准差: {np.std(valid_rps):.4f}")
        else:
            print("  ❌ 没有有效的RPS值")
            
    except Exception as e:
        print(f"❌ 分析训练数据失败: {e}")

def check_rps_calculation_logic():
    """检查RPS计算逻辑"""
    print("\nRPS计算逻辑检查")
    print("=" * 50)
    
    # 模拟RPS计算
    print("模拟RPS计算过程:")
    
    # 假设有5只股票的收益率
    returns = [0.05, 0.02, -0.01, 0.08, 0.03]
    stock_codes = ['A', 'B', 'C', 'D', 'E']
    
    print(f"股票收益率: {dict(zip(stock_codes, returns))}")
    
    # 按收益率排序
    sorted_data = sorted(zip(stock_codes, returns), key=lambda x: x[1], reverse=True)
    print(f"排序后: {sorted_data}")
    
    # 计算RPS
    total_stocks = len(returns)
    for i, (code, ret) in enumerate(sorted_data):
        rank = i + 1
        rps_value = (1 - rank / total_stocks) * 100
        print(f"  {code}: 收益率={ret:.3f}, 排名={rank}, RPS={rps_value:.2f}")
    
    print("\n预期RPS范围: 0-100")
    print("预期RPS分布: 应该相对均匀分布，不应该集中在某个值")

def main():
    """主函数"""
    analyze_rps_data()
    analyze_training_data_rps()
    check_rps_calculation_logic()
    
    print("\n🔍 RPS问题诊断建议:")
    print("1. 检查RPS数据是否正确加载")
    print("2. 检查RPS计算逻辑是否正确")
    print("3. 检查RPS数据的时间对齐")
    print("4. 检查是否存在大量默认值或缺失值")
    print("5. 验证RPS值的分布是否合理")

if __name__ == "__main__":
    main()