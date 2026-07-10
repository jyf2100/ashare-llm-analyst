#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细检查RPS数据结构
"""

import os
import pickle
import pandas as pd

def inspect_rps_structure_detailed():
    """详细检查RPS数据结构"""
    print("🔍 详细检查RPS数据结构")
    print("=" * 50)
    
    rps_dir = "rps_results"
    rps_files = [f for f in os.listdir(rps_dir) if f.endswith('.pkl')]
    
    for rps_file in rps_files:
        print(f"\n📁 检查文件: {rps_file}")
        file_path = os.path.join(rps_dir, rps_file)
        
        with open(file_path, 'rb') as f:
            rps_data = pickle.load(f)
        
        print(f"数据类型: {type(rps_data)}")
        print(f"顶级键: {list(rps_data.keys())}")
        
        # 详细检查每个RPS周期
        for period_key in rps_data.keys():
            print(f"\n🔍 检查周期: {period_key}")
            period_data = rps_data[period_key]
            print(f"  数据类型: {type(period_data)}")
            
            if isinstance(period_data, dict):
                print(f"  字典键数量: {len(period_data)}")
                sample_keys = list(period_data.keys())[:10]
                print(f"  前10个键: {sample_keys}")
                
                # 检查字典中的值
                for sample_key in sample_keys[:3]:
                    sample_value = period_data[sample_key]
                    print(f"    {sample_key}: {type(sample_value)}")
                    
                    if isinstance(sample_value, (list, pd.Series)):
                        print(f"      长度: {len(sample_value)}")
                        if len(sample_value) > 0:
                            print(f"      第一个元素: {sample_value[0]}")
                            print(f"      第一个元素类型: {type(sample_value[0])}")
                            if len(sample_value) > 5:
                                print(f"      前5个元素: {sample_value[:5]}")
                    
                    elif isinstance(sample_value, pd.DataFrame):
                        print(f"      DataFrame形状: {sample_value.shape}")
                        print(f"      列名: {list(sample_value.columns)[:10]}")
                        print(f"      索引类型: {type(sample_value.index)}")
                        if len(sample_value) > 0:
                            print(f"      前3行:")
                            print(sample_value.head(3))
                            
                            # 如果列是股票代码，检查几只股票的数据
                            if len(sample_value.columns) > 0:
                                stock_samples = list(sample_value.columns)[:3]
                                for stock_code in stock_samples:
                                    stock_data = sample_value[stock_code].dropna()
                                    if len(stock_data) > 0:
                                        print(f"        {stock_code}: {len(stock_data)} 个有效值")
                                        print(f"          范围: {stock_data.min():.2f} - {stock_data.max():.2f}")
                                        print(f"          平均: {stock_data.mean():.2f}")
                                        print(f"          前3个值: {stock_data.head(3).tolist()}")
                    
                    elif isinstance(sample_value, dict):
                        print(f"      嵌套字典键数量: {len(sample_value)}")
                        nested_keys = list(sample_value.keys())[:5]
                        print(f"      嵌套字典前5个键: {nested_keys}")
                        
                        for nested_key in nested_keys[:2]:
                            nested_value = sample_value[nested_key]
                            print(f"        {nested_key}: {type(nested_value)}")
                            if isinstance(nested_value, (list, pd.Series)):
                                print(f"          长度: {len(nested_value)}")
                                if len(nested_value) > 0:
                                    print(f"          第一个元素: {nested_value[0]}")
            
            elif isinstance(period_data, pd.DataFrame):
                print(f"  DataFrame形状: {period_data.shape}")
                print(f"  列名: {list(period_data.columns)[:10]}")
                print(f"  索引类型: {type(period_data.index)}")
                if len(period_data) > 0:
                    print(f"  前3行:")
                    print(period_data.head(3))
                    
                    # 检查是否列是股票代码
                    if len(period_data.columns) > 0:
                        stock_samples = [col for col in period_data.columns if isinstance(col, str) and ('.' in col or len(col) == 6)][:5]
                        if stock_samples:
                            print(f"  疑似股票代码列: {stock_samples}")
                            for stock_code in stock_samples[:3]:
                                stock_data = period_data[stock_code].dropna()
                                if len(stock_data) > 0:
                                    print(f"    {stock_code}: {len(stock_data)} 个有效值")
                                    print(f"      范围: {stock_data.min():.2f} - {stock_data.max():.2f}")
                                    print(f"      平均: {stock_data.mean():.2f}")
                                    print(f"      前3个值: {stock_data.head(3).tolist()}")
                                    print(f"      索引前3个: {stock_data.index[:3].tolist()}")
            
            elif isinstance(period_data, list):
                print(f"  列表长度: {len(period_data)}")
                if len(period_data) > 0:
                    print(f"  第一个元素: {period_data[0]}")
                    print(f"  第一个元素类型: {type(period_data[0])}")
                    if len(period_data) > 5:
                        print(f"  前5个元素: {period_data[:5]}")
        
        break  # 只检查第一个文件

def suggest_correct_usage():
    """建议正确的使用方法"""
    print(f"\n🔧 正确使用建议")
    print("=" * 50)
    print("基于RPS数据结构分析，建议:")
    print("1. 如果RPS数据是多周期结构 (RPS20, RPS60, RPS120)")
    print("2. 每个周期可能包含DataFrame，行为日期，列为股票代码")
    print("3. 需要修改load_rps_data方法以正确解析这种结构")
    print("4. 选择合适的RPS周期进行分析")
    print("\n修复步骤:")
    print("1. 修改load_rps_data方法")
    print("2. 选择RPS20作为默认周期")
    print("3. 从DataFrame中提取股票代码和时间序列")
    print("4. 重新生成训练数据")

if __name__ == "__main__":
    inspect_rps_structure_detailed()
    suggest_correct_usage()