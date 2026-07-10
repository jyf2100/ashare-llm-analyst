#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPS结果验证和查看工具
用于验证和分析计算出的RPS周期数据

功能:
1. 验证RPS数据的完整性和正确性
2. 展示各周期RPS数据的统计信息
3. 查看特定股票的RPS趋势
4. 生成RPS数据概览报告

作者: AI Assistant
日期: 2025-08-12
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class RPSResultsValidator:
    """
    RPS结果验证器
    """
    
    def __init__(self, results_dir="rps_results"):
        """
        初始化验证器
        
        参数:
        results_dir: str, 结果目录路径
        """
        self.results_dir = results_dir
        self.rps_data = None
        
    def load_rps_data(self, pickle_file=None):
        """
        加载RPS数据
        
        参数:
        pickle_file: str, pickle文件路径（可选）
        
        返回:
        bool: 是否成功加载
        """
        if pickle_file is None:
            # 查找最新的pickle文件
            pickle_files = [f for f in os.listdir(self.results_dir) if f.endswith('.pkl')]
            if not pickle_files:
                print("❌ 没有找到pickle文件")
                return False
            pickle_file = os.path.join(self.results_dir, sorted(pickle_files)[-1])
        
        try:
            with open(pickle_file, 'rb') as f:
                self.rps_data = pickle.load(f)
            print(f"✅ 成功加载RPS数据: {pickle_file}")
            return True
        except Exception as e:
            print(f"❌ 加载RPS数据失败: {e}")
            return False
    
    def validate_data_integrity(self):
        """
        验证数据完整性
        
        返回:
        dict: 验证结果
        """
        if not self.rps_data:
            print("❌ 没有RPS数据")
            return None
        
        print("🔍 === 数据完整性验证 ===")
        
        validation_results = {}
        
        for period_name, period_data in self.rps_data.items():
            print(f"\n📊 验证 {period_name}...")
            
            # 基本统计
            stock_count = len(period_data)
            total_records = sum(len(stock_rps) for stock_rps in period_data.values())
            
            # RPS值范围检查
            all_rps_values = []
            invalid_rps_count = 0
            
            for stock_code, stock_rps in period_data.items():
                for record in stock_rps:
                    rps_value = record['rps']
                    all_rps_values.append(rps_value)
                    
                    # 检查RPS值是否在有效范围内
                    if not (0 <= rps_value <= 100):
                        invalid_rps_count += 1
            
            # 日期连续性检查
            date_gaps = 0
            for stock_code, stock_rps in list(period_data.items())[:10]:  # 检查前10只股票
                dates = [pd.to_datetime(record['date']) for record in stock_rps]
                if len(dates) > 1:
                    date_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                    # 检查是否有异常的日期间隔（超过10天）
                    date_gaps += sum(1 for diff in date_diffs if diff > 10)
            
            validation_results[period_name] = {
                'stock_count': stock_count,
                'total_records': total_records,
                'avg_records_per_stock': total_records / stock_count if stock_count > 0 else 0,
                'rps_min': min(all_rps_values) if all_rps_values else 0,
                'rps_max': max(all_rps_values) if all_rps_values else 0,
                'rps_mean': np.mean(all_rps_values) if all_rps_values else 0,
                'invalid_rps_count': invalid_rps_count,
                'date_gaps': date_gaps
            }
            
            print(f"  股票数量: {stock_count}")
            print(f"  总记录数: {total_records}")
            print(f"  平均记录/股票: {total_records / stock_count:.1f}")
            print(f"  RPS范围: {min(all_rps_values):.2f} - {max(all_rps_values):.2f}")
            print(f"  RPS平均值: {np.mean(all_rps_values):.2f}")
            print(f"  无效RPS值: {invalid_rps_count}")
            print(f"  日期间隔异常: {date_gaps}")
            
            if invalid_rps_count == 0 and date_gaps < 5:
                print(f"  ✅ {period_name} 数据验证通过")
            else:
                print(f"  ⚠️  {period_name} 数据存在问题")
        
        return validation_results
    
    def show_stock_rps_trend(self, stock_code, periods=None):
        """
        显示特定股票的RPS趋势
        
        参数:
        stock_code: str, 股票代码
        periods: list, 要显示的周期（可选）
        """
        if not self.rps_data:
            print("❌ 没有RPS数据")
            return
        
        if periods is None:
            periods = list(self.rps_data.keys())
        
        print(f"\n📈 === {stock_code} RPS趋势 ===")
        
        # 检查股票是否存在
        available_periods = []
        for period_name in periods:
            if period_name in self.rps_data and stock_code in self.rps_data[period_name]:
                available_periods.append(period_name)
        
        if not available_periods:
            print(f"❌ 股票 {stock_code} 在指定周期中没有数据")
            return
        
        # 显示最近的RPS值
        for period_name in available_periods:
            stock_rps = self.rps_data[period_name][stock_code]
            recent_rps = stock_rps[-10:]  # 最近10个交易日
            
            print(f"\n{period_name} 最近10日RPS:")
            for record in recent_rps:
                print(f"  {record['date']}: {record['rps']:.2f}")
    
    def generate_rps_summary(self):
        """
        生成RPS数据概览
        
        返回:
        dict: 概览数据
        """
        if not self.rps_data:
            print("❌ 没有RPS数据")
            return None
        
        print("\n📊 === RPS数据概览 ===")
        
        summary = {}
        
        for period_name, period_data in self.rps_data.items():
            # 获取最新的RPS值
            latest_rps_values = []
            latest_dates = []
            
            for stock_code, stock_rps in period_data.items():
                if stock_rps:
                    latest_record = stock_rps[-1]
                    latest_rps_values.append(latest_record['rps'])
                    latest_dates.append(latest_record['date'])
            
            if latest_rps_values:
                # 计算统计信息
                rps_stats = {
                    'count': len(latest_rps_values),
                    'mean': np.mean(latest_rps_values),
                    'median': np.median(latest_rps_values),
                    'std': np.std(latest_rps_values),
                    'min': np.min(latest_rps_values),
                    'max': np.max(latest_rps_values),
                    'rps_80_plus': sum(1 for rps in latest_rps_values if rps >= 80),
                    'rps_90_plus': sum(1 for rps in latest_rps_values if rps >= 90),
                    'rps_95_plus': sum(1 for rps in latest_rps_values if rps >= 95),
                    'latest_date': max(latest_dates) if latest_dates else None
                }
                
                summary[period_name] = rps_stats
                
                print(f"\n{period_name}:")
                print(f"  股票数量: {rps_stats['count']}")
                print(f"  最新日期: {rps_stats['latest_date']}")
                print(f"  RPS均值: {rps_stats['mean']:.2f}")
                print(f"  RPS中位数: {rps_stats['median']:.2f}")
                print(f"  RPS标准差: {rps_stats['std']:.2f}")
                print(f"  RPS范围: {rps_stats['min']:.2f} - {rps_stats['max']:.2f}")
                print(f"  RPS≥80: {rps_stats['rps_80_plus']} ({rps_stats['rps_80_plus']/rps_stats['count']*100:.1f}%)")
                print(f"  RPS≥90: {rps_stats['rps_90_plus']} ({rps_stats['rps_90_plus']/rps_stats['count']*100:.1f}%)")
                print(f"  RPS≥95: {rps_stats['rps_95_plus']} ({rps_stats['rps_95_plus']/rps_stats['count']*100:.1f}%)")
        
        return summary
    
    def find_high_rps_stocks(self, min_rps=80, periods=None):
        """
        查找高RPS股票
        
        参数:
        min_rps: float, 最小RPS阈值
        periods: list, 要检查的周期（可选）
        
        返回:
        dict: 高RPS股票列表
        """
        if not self.rps_data:
            print("❌ 没有RPS数据")
            return None
        
        if periods is None:
            periods = list(self.rps_data.keys())
        
        print(f"\n🔍 === 查找RPS≥{min_rps}的股票 ===")
        
        high_rps_stocks = {}
        
        for period_name in periods:
            if period_name not in self.rps_data:
                continue
            
            period_data = self.rps_data[period_name]
            high_rps_list = []
            
            for stock_code, stock_rps in period_data.items():
                if stock_rps:
                    latest_rps = stock_rps[-1]['rps']
                    latest_date = stock_rps[-1]['date']
                    
                    if latest_rps >= min_rps:
                        high_rps_list.append({
                            'stock_code': stock_code,
                            'rps': latest_rps,
                            'date': latest_date
                        })
            
            # 按RPS值排序
            high_rps_list.sort(key=lambda x: x['rps'], reverse=True)
            high_rps_stocks[period_name] = high_rps_list
            
            print(f"\n{period_name}: {len(high_rps_list)} 只股票")
            
            # 显示前10只
            for i, stock_info in enumerate(high_rps_list[:10]):
                print(f"  {i+1:2d}. {stock_info['stock_code']}: {stock_info['rps']:.2f} ({stock_info['date']})")
            
            if len(high_rps_list) > 10:
                print(f"  ... 还有 {len(high_rps_list) - 10} 只股票")
        
        return high_rps_stocks
    
    def compare_periods(self, stock_codes=None, num_stocks=5):
        """
        比较不同周期的RPS值
        
        参数:
        stock_codes: list, 要比较的股票代码（可选）
        num_stocks: int, 随机选择的股票数量
        """
        if not self.rps_data:
            print("❌ 没有RPS数据")
            return
        
        print("\n🔄 === 不同周期RPS比较 ===")
        
        # 如果没有指定股票，随机选择一些股票
        if stock_codes is None:
            # 找到所有周期都有数据的股票
            common_stocks = None
            for period_name, period_data in self.rps_data.items():
                period_stocks = set(period_data.keys())
                if common_stocks is None:
                    common_stocks = period_stocks
                else:
                    common_stocks = common_stocks.intersection(period_stocks)
            
            if not common_stocks:
                print("❌ 没有找到所有周期都有数据的股票")
                return
            
            stock_codes = list(common_stocks)[:num_stocks]
        
        # 比较每只股票在不同周期的最新RPS值
        periods = list(self.rps_data.keys())
        
        print(f"\n比较股票: {stock_codes}")
        print(f"比较周期: {periods}")
        
        comparison_data = []
        
        for stock_code in stock_codes:
            stock_comparison = {'stock_code': stock_code}
            
            for period_name in periods:
                if (period_name in self.rps_data and 
                    stock_code in self.rps_data[period_name] and 
                    self.rps_data[period_name][stock_code]):
                    
                    latest_rps = self.rps_data[period_name][stock_code][-1]['rps']
                    stock_comparison[period_name] = latest_rps
                else:
                    stock_comparison[period_name] = None
            
            comparison_data.append(stock_comparison)
        
        # 显示比较结果
        print(f"\n{'股票代码':<12}", end="")
        for period in periods:
            print(f"{period:<8}", end="")
        print()
        
        print("-" * (12 + len(periods) * 8))
        
        for stock_data in comparison_data:
            print(f"{stock_data['stock_code']:<12}", end="")
            for period in periods:
                rps_value = stock_data.get(period)
                if rps_value is not None:
                    print(f"{rps_value:<8.1f}", end="")
                else:
                    print(f"{'N/A':<8}", end="")
            print()

def main():
    """
    主函数
    """
    print("=== RPS结果验证和查看工具 ===")
    
    # 创建验证器
    validator = RPSResultsValidator()
    
    # 加载数据
    if not validator.load_rps_data():
        return
    
    # 验证数据完整性
    validation_results = validator.validate_data_integrity()
    
    # 生成概览
    summary = validator.generate_rps_summary()
    
    # 查找高RPS股票
    high_rps_stocks = validator.find_high_rps_stocks(min_rps=90)
    
    # 比较不同周期
    validator.compare_periods(num_stocks=5)
    
    # 显示特定股票趋势（示例）
    if validator.rps_data:
        # 选择一只有数据的股票进行展示
        sample_period = list(validator.rps_data.keys())[0]
        sample_stocks = list(validator.rps_data[sample_period].keys())[:3]
        
        for stock_code in sample_stocks:
            validator.show_stock_rps_trend(stock_code)
            break  # 只显示一只股票的详细趋势

if __name__ == "__main__":
    main()