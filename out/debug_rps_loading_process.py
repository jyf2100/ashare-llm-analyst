#!/usr/bin/env python3
"""
调试RPS数据加载过程
检查RPS数据的实际加载情况和数据结构
"""

import os
import pickle
import pandas as pd
from datetime import datetime

def debug_rps_loading():
    """调试RPS数据加载过程"""
    print("=== RPS数据加载过程调试 ===")
    
    rps_data_dir = "rps_results"
    
    if not os.path.exists(rps_data_dir):
        print(f"❌ RPS目录不存在: {rps_data_dir}")
        return
    
    # 查找RPS文件
    rps_files = [f for f in os.listdir(rps_data_dir) if f.endswith('.pkl')]
    print(f"📁 找到RPS文件: {rps_files}")
    
    if not rps_files:
        print("❌ 没有找到RPS文件")
        return
    
    # 加载第一个RPS文件
    rps_file = rps_files[0]
    file_path = os.path.join(rps_data_dir, rps_file)
    
    print(f"\n📂 加载文件: {rps_file}")
    
    try:
        with open(file_path, 'rb') as f:
            raw_rps_data = pickle.load(f)
        
        print(f"✅ 文件加载成功")
        print(f"📊 原始数据类型: {type(raw_rps_data)}")
        print(f"📊 原始数据键数量: {len(raw_rps_data)}")
        
        # 检查前几个键
        sample_keys = list(raw_rps_data.keys())[:5]
        print(f"📊 前5个键: {sample_keys}")
        
        # 模拟load_rps_data方法的处理过程
        processed_rps_data = {}
        
        for stock_code, rps_list in raw_rps_data.items():
            if stock_code not in processed_rps_data:
                processed_rps_data[stock_code] = {}
            
            print(f"\n🔍 处理股票: {stock_code}")
            print(f"   数据类型: {type(rps_list)}")
            
            if isinstance(rps_list, list):
                print(f"   列表长度: {len(rps_list)}")
                if len(rps_list) > 0:
                    print(f"   第一个元素: {rps_list[0]}")
                    print(f"   第一个元素类型: {type(rps_list[0])}")
                    
                    # 处理列表数据
                    rps_dict = {}
                    valid_items = 0
                    
                    for item in rps_list:
                        if isinstance(item, dict) and 'date' in item and 'rps' in item:
                            rps_dict[item['date']] = float(item['rps'])
                            valid_items += 1
                    
                    print(f"   有效RPS项目数: {valid_items}")
                    
                    if rps_dict:
                        rps_series = pd.Series(rps_dict)
                        rps_series.index = pd.to_datetime(rps_series.index)
                        processed_rps_data[stock_code]['rps'] = rps_series.sort_index()
                        
                        print(f"   RPS序列长度: {len(rps_series)}")
                        print(f"   RPS日期范围: {rps_series.index.min()} 到 {rps_series.index.max()}")
                        print(f"   RPS值范围: {rps_series.min():.2f} 到 {rps_series.max():.2f}")
                    else:
                        print(f"   ❌ 没有有效的RPS数据")
            else:
                print(f"   ❌ 数据不是列表格式")
            
            # 只处理前3个股票作为示例
            if len(processed_rps_data) >= 3:
                break
        
        print(f"\n📈 处理后的RPS数据统计:")
        print(f"   股票数量: {len(processed_rps_data)}")
        
        # 检查是否有有效的RPS数据
        valid_stocks = 0
        for stock_code, data in processed_rps_data.items():
            if 'rps' in data and len(data['rps']) > 0:
                valid_stocks += 1
        
        print(f"   有效RPS数据的股票数: {valid_stocks}")
        
        # 测试RPS值获取
        print(f"\n🧪 测试RPS值获取:")
        test_date = pd.Timestamp('2024-12-01')
        
        for stock_code, data in list(processed_rps_data.items())[:2]:
            if 'rps' in data:
                rps_series = data['rps']
                print(f"\n   股票: {stock_code}")
                print(f"   测试日期: {test_date}")
                
                # 模拟_get_rps_value方法
                try:
                    valid_dates = rps_series.index[rps_series.index <= test_date]
                    if len(valid_dates) > 0:
                        rps_value = float(rps_series[valid_dates.max()])
                        print(f"   获取到的RPS值: {rps_value}")
                        if 0 <= rps_value <= 100:
                            print(f"   ✅ RPS值有效")
                        else:
                            print(f"   ❌ RPS值超出范围")
                    else:
                        print(f"   ❌ 没有找到有效日期")
                except Exception as e:
                    print(f"   ❌ 获取RPS值失败: {e}")
        
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_rps_loading()