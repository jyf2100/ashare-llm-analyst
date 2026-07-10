#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试RPS数据匹配过程
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RPSMatchingDebugger:
    """RPS数据匹配调试器"""
    
    def __init__(self, market_data_dir="market_data", rps_data_dir="rps_results"):
        self.market_data_dir = market_data_dir
        self.rps_data_dir = rps_data_dir
        self.stock_data = {}
        self.rps_data = {}
    
    def load_sample_market_data(self, sample_size=3):
        """加载样本市场数据"""
        print(f"📊 加载样本市场数据 (样本数: {sample_size})")
        
        if not os.path.exists(self.market_data_dir):
            print(f"❌ 市场数据目录不存在: {self.market_data_dir}")
            return False
        
        csv_files = [f for f in os.listdir(self.market_data_dir) if f.endswith('.csv')]
        
        for i, csv_file in enumerate(csv_files[:sample_size]):
            try:
                file_path = os.path.join(self.market_data_dir, csv_file)
                df = pd.read_csv(file_path)
                
                # 标准化列名
                df.columns = df.columns.str.lower()
                
                # 确保有必要的列
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    continue
                
                # 处理日期
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # 提取股票代码
                stock_code = csv_file.replace('.csv', '')
                self.stock_data[stock_code] = df
                
                print(f"  ✅ {stock_code}: {len(df)} 条记录, 日期范围: {df.index.min()} 到 {df.index.max()}")
                
            except Exception as e:
                print(f"  ❌ 加载 {csv_file} 失败: {e}")
        
        print(f"成功加载 {len(self.stock_data)} 只股票数据")
        return len(self.stock_data) > 0
    
    def load_rps_data(self):
        """加载RPS数据"""
        print(f"\n📈 加载RPS数据")
        
        if not os.path.exists(self.rps_data_dir):
            print(f"❌ RPS目录不存在: {self.rps_data_dir}")
            return False
        
        rps_files = [f for f in os.listdir(self.rps_data_dir) if f.endswith('.pkl')]
        
        for rps_file in rps_files:
            try:
                file_path = os.path.join(self.rps_data_dir, rps_file)
                with open(file_path, 'rb') as f:
                    rps_data = pickle.load(f)
                
                print(f"  📁 处理文件: {rps_file}")
                print(f"     股票数量: {len(rps_data)}")
                
                # 处理RPS数据格式
                for stock_code, rps_list in rps_data.items():
                    if stock_code not in self.rps_data:
                        self.rps_data[stock_code] = {}
                    
                    if isinstance(rps_list, list) and len(rps_list) > 0:
                        rps_dict = {}
                        for item in rps_list:
                            if isinstance(item, dict) and 'date' in item and 'rps' in item:
                                rps_dict[item['date']] = float(item['rps'])
                        
                        if rps_dict:
                            rps_series = pd.Series(rps_dict)
                            rps_series.index = pd.to_datetime(rps_series.index)
                            self.rps_data[stock_code]['rps'] = rps_series.sort_index()
                
                print(f"     处理后股票数量: {len(self.rps_data)}")
                break  # 只处理第一个文件
                
            except Exception as e:
                print(f"  ❌ 加载RPS文件 {rps_file} 失败: {e}")
        
        return len(self.rps_data) > 0
    
    def debug_rps_matching(self):
        """调试RPS匹配过程"""
        print(f"\n🔍 调试RPS匹配过程")
        print("=" * 50)
        
        for stock_code, market_data in self.stock_data.items():
            print(f"\n📊 股票: {stock_code}")
            print(f"  市场数据: {len(market_data)} 条记录")
            print(f"  日期范围: {market_data.index.min()} 到 {market_data.index.max()}")
            
            # 检查RPS数据是否存在
            if stock_code in self.rps_data and 'rps' in self.rps_data[stock_code]:
                rps_series = self.rps_data[stock_code]['rps']
                print(f"  RPS数据: {len(rps_series)} 条记录")
                print(f"  RPS日期范围: {rps_series.index.min()} 到 {rps_series.index.max()}")
                
                # 检查日期重叠
                market_dates = set(market_data.index.date)
                rps_dates = set(rps_series.index.date)
                overlap_dates = market_dates.intersection(rps_dates)
                
                print(f"  日期重叠: {len(overlap_dates)} 天")
                print(f"  重叠比例: {len(overlap_dates)/len(market_dates)*100:.2f}%")
                
                if len(overlap_dates) > 0:
                    # 测试RPS值获取
                    sample_dates = sorted(list(overlap_dates))[:5]
                    print(f"  样本日期RPS值:")
                    
                    for date in sample_dates:
                        target_date = pd.Timestamp(date)
                        rps_value = self._get_rps_value(rps_series, target_date)
                        print(f"    {date}: {rps_value}")
                else:
                    print(f"  ⚠️  警告: 没有日期重叠")
                    
                    # 显示最近的日期
                    print(f"  最近的市场数据日期: {max(market_dates)}")
                    print(f"  最近的RPS数据日期: {max(rps_dates)}")
            else:
                print(f"  ❌ 没有RPS数据")
                
                # 检查股票代码格式
                print(f"  股票代码格式检查:")
                print(f"    原始代码: '{stock_code}'")
                
                # 尝试不同的代码格式
                possible_codes = [
                    stock_code,
                    stock_code.upper(),
                    stock_code.lower(),
                    stock_code.replace('sh.', '').replace('sz.', ''),
                    'sh.' + stock_code if not stock_code.startswith(('sh.', 'sz.')) else stock_code,
                    'sz.' + stock_code if not stock_code.startswith(('sh.', 'sz.')) else stock_code
                ]
                
                for code in possible_codes:
                    if code in self.rps_data:
                        print(f"    找到匹配: '{code}'")
                        break
                else:
                    print(f"    没有找到匹配的RPS数据")
                    print(f"    RPS数据中的前5个股票代码: {list(self.rps_data.keys())[:5]}")
    
    def _get_rps_value(self, rps_series, target_date):
        """获取RPS值（复制原始逻辑）"""
        try:
            valid_dates = rps_series.index[rps_series.index <= target_date]
            if len(valid_dates) > 0:
                return float(rps_series[valid_dates.max()])
        except Exception as e:
            print(f"      获取RPS值出错: {e}")
        return -99
    
    def test_rps_integration(self):
        """测试RPS集成"""
        print(f"\n🧪 测试RPS集成")
        print("=" * 50)
        
        for stock_code, market_data in self.stock_data.items():
            print(f"\n测试股票: {stock_code}")
            
            # 模拟训练数据生成过程
            data_with_features = market_data.copy()
            
            # 添加RPS特征
            if stock_code in self.rps_data and 'rps' in self.rps_data[stock_code]:
                rps_series = self.rps_data[stock_code]['rps']
                print(f"  使用RPS数据")
                
                # 应用RPS映射
                rps_values = data_with_features.index.map(
                    lambda x: self._get_rps_value(rps_series, x)
                )
                data_with_features['rps'] = rps_values
            else:
                print(f"  使用默认RPS值")
                data_with_features['rps'] = 50.0
            
            # 统计RPS值
            rps_stats = data_with_features['rps'].describe()
            print(f"  RPS统计:")
            print(f"    数量: {rps_stats['count']}")
            print(f"    平均值: {rps_stats['mean']:.4f}")
            print(f"    标准差: {rps_stats['std']:.4f}")
            print(f"    最小值: {rps_stats['min']:.4f}")
            print(f"    最大值: {rps_stats['max']:.4f}")
            
            # 检查唯一值
            unique_values = data_with_features['rps'].unique()
            print(f"    唯一值数量: {len(unique_values)}")
            
            if len(unique_values) == 1:
                print(f"    ⚠️  警告: 所有RPS值都相同 ({unique_values[0]})")
            
            # 显示前几个值
            print(f"    前5个RPS值: {data_with_features['rps'].head().tolist()}")
            
            break  # 只测试第一只股票

def main():
    """主函数"""
    debugger = RPSMatchingDebugger()
    
    # 加载样本数据
    if not debugger.load_sample_market_data(sample_size=3):
        print("❌ 无法加载市场数据")
        return
    
    # 加载RPS数据
    if not debugger.load_rps_data():
        print("❌ 无法加载RPS数据")
        return
    
    # 调试匹配过程
    debugger.debug_rps_matching()
    
    # 测试集成
    debugger.test_rps_integration()
    
    print("\n🎯 问题诊断结论:")
    print("1. 检查股票代码格式是否匹配")
    print("2. 检查日期范围是否重叠")
    print("3. 检查RPS数据结构是否正确")
    print("4. 检查时间对齐逻辑是否正确")

if __name__ == "__main__":
    main()