#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查RPS pkl文件的内容结构
"""

import pickle
import os
import pandas as pd

def check_rps_pkl():
    """检查RPS pkl文件内容"""
    rps_file = "rps_results/multi_period_rps_20250810_084703.pkl"
    
    if not os.path.exists(rps_file):
        print(f"文件不存在: {rps_file}")
        return
        
    try:
        with open(rps_file, 'rb') as f:
            data = pickle.load(f)
            
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"字典键: {list(data.keys())}")
            for key, value in data.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    形状: {value.shape}")
                elif hasattr(value, '__len__'):
                    print(f"    长度: {len(value)}")
        elif isinstance(data, pd.DataFrame):
            print(f"DataFrame形状: {data.shape}")
            print(f"列名: {list(data.columns)}")
            print(f"前5行:")
            print(data.head())
        else:
            print(f"数据内容: {data}")
            
    except Exception as e:
        print(f"加载文件失败: {e}")

if __name__ == "__main__":
    check_rps_pkl()