#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查RPS文件的实际结构
"""

import os
import pickle
import pandas as pd

def inspect_rps_file():
    """检查RPS文件结构"""
    print("RPS文件结构检查")
    print("=" * 50)
    
    rps_dir = "rps_results"
    if not os.path.exists(rps_dir):
        print(f"❌ RPS目录不存在: {rps_dir}")
        return
    
    rps_files = [f for f in os.listdir(rps_dir) if f.endswith('.pkl')]
    
    for rps_file in rps_files:
        print(f"\n📁 检查文件: {rps_file}")
        file_path = os.path.join(rps_dir, rps_file)
        
        try:
            with open(file_path, 'rb') as f:
                rps_data = pickle.load(f)
            
            print(f"数据类型: {type(rps_data)}")
            print(f"顶级键数量: {len(rps_data)}")
            print(f"顶级键: {list(rps_data.keys())}")
            
            # 检查每个顶级键的内容
            for key in list(rps_data.keys())[:3]:  # 只检查前3个
                print(f"\n🔍 检查键: {key}")
                value = rps_data[key]
                print(f"  值类型: {type(value)}")
                
                if isinstance(value, dict):
                    print(f"  字典键数量: {len(value)}")
                    sample_keys = list(value.keys())[:5]
                    print(f"  样本键: {sample_keys}")
                    
                    # 检查字典中的值
                    for sample_key in sample_keys[:2]:
                        sample_value = value[sample_key]
                        print(f"    {sample_key}: {type(sample_value)}")
                        
                        if isinstance(sample_value, (list, pd.Series)):
                            print(f"      长度: {len(sample_value)}")
                            if len(sample_value) > 0:
                                print(f"      第一个元素: {sample_value[0] if hasattr(sample_value, '__getitem__') else 'N/A'}")
                                print(f"      第一个元素类型: {type(sample_value[0]) if hasattr(sample_value, '__getitem__') and len(sample_value) > 0 else 'N/A'}")
                        elif isinstance(sample_value, pd.DataFrame):
                            print(f"      DataFrame形状: {sample_value.shape}")
                            print(f"      列名: {list(sample_value.columns)}")
                            print(f"      索引类型: {type(sample_value.index)}")
                            if len(sample_value) > 0:
                                print(f"      前几行:")
                                print(sample_value.head())
                
                elif isinstance(value, (list, pd.Series)):
                    print(f"  长度: {len(value)}")
                    if len(value) > 0:
                        print(f"  第一个元素: {value[0]}")
                        print(f"  第一个元素类型: {type(value[0])}")
                        
                elif isinstance(value, pd.DataFrame):
                    print(f"  DataFrame形状: {value.shape}")
                    print(f"  列名: {list(value.columns)}")
                    print(f"  索引类型: {type(value.index)}")
                    if len(value) > 0:
                        print(f"  前几行:")
                        print(value.head())
                        
                        # 如果DataFrame的列是股票代码
                        if len(value.columns) > 0:
                            sample_stocks = list(value.columns)[:5]
                            print(f"  样本股票代码: {sample_stocks}")
                            
                            # 检查某只股票的RPS数据
                            if len(sample_stocks) > 0:
                                stock_code = sample_stocks[0]
                                stock_rps = value[stock_code].dropna()
                                print(f"  {stock_code} RPS数据:")
                                print(f"    有效数据点: {len(stock_rps)}")
                                if len(stock_rps) > 0:
                                    print(f"    最小值: {stock_rps.min():.2f}")
                                    print(f"    最大值: {stock_rps.max():.2f}")
                                    print(f"    平均值: {stock_rps.mean():.2f}")
                                    print(f"    前5个值: {stock_rps.head().tolist()}")
                
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            import traceback
            traceback.print_exc()

def suggest_fix():
    """建议修复方案"""
    print("\n🔧 修复建议")
    print("=" * 50)
    print("基于检查结果，RPS数据结构可能是:")
    print("1. 多周期RPS数据 (RPS20, RPS60, RPS120)")
    print("2. 每个周期包含DataFrame，行为日期，列为股票代码")
    print("3. 需要修改训练数据生成器以正确解析这种结构")
    print("\n建议的修复步骤:")
    print("1. 修改load_rps_data方法以处理多周期结构")
    print("2. 选择合适的RPS周期（如RPS20）")
    print("3. 正确提取股票代码和对应的RPS时间序列")

if __name__ == "__main__":
    inspect_rps_file()
    suggest_fix()