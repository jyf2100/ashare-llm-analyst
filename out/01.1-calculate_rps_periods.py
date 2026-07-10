#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算RPS周期数据：5/10/20/60/120/250天
基于market_data目录下的股票数据计算多个RPS周期

功能:
1. 计算5、10、20、60、120、250天RPS周期数据
2. 使用向量化计算提升性能
3. 自动保存计算结果到rps_results目录
4. 提供详细的计算统计信息

作者: AI Assistant
日期: 2025-08-12
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import shutil
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class RPSPeriodsCalculator:
    """
    RPS周期数据计算器
    支持计算5、10、20、60、120、250天RPS周期
    """
    
    def __init__(self, data_dir="market_data"):
        """
        初始化RPS周期计算器
        
        参数:
        data_dir: str, 数据目录路径
        """
        self.data_dir = data_dir
        self.stock_data = {}
        self.price_matrix = None
        self.date_index = None
        self.stock_codes = None
        
        # 创建结果保存目录
        self.output_dir = "rps_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 创建结果目录: {self.output_dir}")
    
    def load_stock_data(self, stock_file):
        """
        加载单个股票数据
        
        参数:
        stock_file: str, 股票文件名
        
        返回:
        DataFrame or None: 股票数据
        """
        file_path = os.path.join(self.data_dir, stock_file)
        
        try:
            df = pd.read_csv(file_path)
            
            # 检查必要的列
            required_columns = ['date', 'close']
            if not all(col in df.columns for col in required_columns):
                return None
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 检查数据质量
            if len(df) < 250:  # 至少需要250天数据
                return None
            
            # 去除价格为0或负数的数据
            df = df[df['close'] > 0]
            
            if len(df) < 250:
                return None
                
            return df
            
        except Exception as e:
            return None
    
    def load_all_stock_data(self):
        """
        加载所有股票数据
        
        返回:
        bool: 是否成功加载数据
        """
        print("📊 开始加载股票数据...")
        
        if not os.path.exists(self.data_dir):
            print(f"❌ 数据目录不存在: {self.data_dir}")
            return False
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"❌ 在 {self.data_dir} 中没有找到CSV文件")
            return False
        
        print(f"📁 发现 {len(csv_files)} 个CSV文件")
        
        success_count = 0
        
        for csv_file in tqdm(csv_files, desc="加载数据"):
            stock_code = csv_file.replace('.csv', '')
            df = self.load_stock_data(csv_file)
            
            if df is not None:
                self.stock_data[stock_code] = df
                success_count += 1
        
        print(f"✅ 成功加载 {success_count} 只股票数据")
        
        if success_count == 0:
            print("❌ 没有成功加载任何股票数据")
            return False
        
        return True
    
    def create_aligned_price_matrix(self, max_period=250):
        """
        创建对齐的价格矩阵
        
        参数:
        max_period: int, 最大周期（用于确定最小数据长度）
        
        返回:
        tuple: (price_matrix, date_index, stock_codes)
        """
        print("🔧 创建价格矩阵...")
        
        if not self.stock_data:
            print("❌ 没有股票数据")
            return None, None, None
        
        # 获取所有日期的并集
        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df['date'])
        
        # 排序日期
        all_dates = sorted(list(all_dates))
        
        # 过滤股票：确保有足够的数据
        valid_stocks = {}
        for stock_code, df in self.stock_data.items():
            if len(df) >= max_period:
                valid_stocks[stock_code] = df
        
        if not valid_stocks:
            print(f"❌ 没有股票具有足够的数据（至少{max_period}天）")
            return None, None, None
        
        print(f"📊 有效股票数量: {len(valid_stocks)}")
        print(f"📅 日期范围: {all_dates[0]} 到 {all_dates[-1]}")
        print(f"📈 总交易日数: {len(all_dates)}")
        
        # 创建价格矩阵
        stock_codes = list(valid_stocks.keys())
        price_matrix = np.full((len(all_dates), len(stock_codes)), np.nan)
        
        for stock_idx, stock_code in enumerate(tqdm(stock_codes, desc="构建矩阵")):
            df = valid_stocks[stock_code]
            
            # 为每个日期填充价格
            for _, row in df.iterrows():
                date_idx = all_dates.index(row['date'])
                price_matrix[date_idx, stock_idx] = row['close']
        
        print(f"✅ 价格矩阵创建完成: {price_matrix.shape}")
        
        return price_matrix, all_dates, stock_codes
    
    def calculate_returns(self, price_matrix, period_days):
        """
        计算指定周期的收益率矩阵
        
        参数:
        price_matrix: np.array, 价格矩阵
        period_days: int, 计算周期
        
        返回:
        np.array: 收益率矩阵
        """
        num_dates, num_stocks = price_matrix.shape
        returns_matrix = np.full((num_dates, num_stocks), np.nan)
        
        # 从period_days开始计算收益率
        for i in range(period_days, num_dates):
            current_prices = price_matrix[i, :]
            past_prices = price_matrix[i - period_days, :]
            
            # 计算收益率
            valid_mask = ~(np.isnan(current_prices) | np.isnan(past_prices) | (past_prices == 0))
            returns_matrix[i, valid_mask] = (current_prices[valid_mask] - past_prices[valid_mask]) / past_prices[valid_mask]
        
        return returns_matrix
    
    def calculate_rps_for_period(self, returns_matrix, period_days):
        """
        计算指定周期的RPS值
        
        参数:
        returns_matrix: np.array, 收益率矩阵
        period_days: int, RPS周期
        
        返回:
        dict: 每只股票的RPS时间序列
        """
        print(f"🧮 计算RPS{period_days}...")
        
        num_dates, num_stocks = returns_matrix.shape
        rps_matrix = np.full((num_dates, num_stocks), np.nan)
        
        # 从period_days开始计算RPS（之前的数据不足）
        for date_idx in tqdm(range(period_days, num_dates), desc=f"计算RPS{period_days}"):
            current_returns = returns_matrix[date_idx, :]
            
            # 移除NaN值
            valid_mask = ~np.isnan(current_returns)
            if valid_mask.sum() < 10:  # 至少需要10只股票
                continue
            
            valid_returns = current_returns[valid_mask]
            
            # 计算RPS
            for stock_idx in range(num_stocks):
                if not valid_mask[stock_idx]:
                    continue
                
                stock_return = current_returns[stock_idx]
                
                # 计算超越的股票数量
                better_count = (valid_returns < stock_return).sum()
                total_count = len(valid_returns)
                
                # RPS = 超越股票数 / 总股票数 * 100
                rps_value = (better_count / total_count) * 100
                rps_matrix[date_idx, stock_idx] = rps_value
        
        # 转换为字典格式
        rps_data = {}
        for stock_idx, stock_code in enumerate(self.stock_codes):
            stock_rps = []
            for date_idx in range(num_dates):
                if not np.isnan(rps_matrix[date_idx, stock_idx]):
                    stock_rps.append({
                        'date': self.date_index[date_idx].strftime('%Y-%m-%d'),
                        'rps': round(rps_matrix[date_idx, stock_idx], 2)
                    })
            
            if stock_rps:  # 只保存有数据的股票
                rps_data[stock_code] = stock_rps
        
        return rps_data
    
    def calculate_multi_period_rps(self, periods=[5, 10, 20, 60, 120, 250]):
        """
        计算多周期RPS
        
        参数:
        periods: list, RPS计算周期列表
        
        返回:
        dict: 各周期的RPS数据
        """
        print(f"🚀 === 开始多周期RPS计算 ===")
        print(f"计算周期: {periods}")
        start_time = time.time()
        
        # 1. 加载数据
        if not self.stock_data:
            self.load_all_stock_data()
        
        if not self.stock_data:
            print("❌ 没有可用的股票数据")
            return None
        
        # 2. 创建价格矩阵
        max_period = max(periods)
        if self.price_matrix is None:
            price_matrix, date_index, stock_codes = self.create_aligned_price_matrix(max_period)
        else:
            price_matrix, date_index, stock_codes = self.price_matrix, self.date_index, self.stock_codes
        
        if price_matrix is None:
            print("❌ 价格矩阵创建失败")
            return None
        
        # 保存矩阵信息
        self.price_matrix = price_matrix
        self.date_index = date_index
        self.stock_codes = stock_codes
        
        # 3. 计算各周期RPS
        all_rps_data = {}
        
        for period in periods:
            print(f"\n📊 处理RPS{period}周期...")
            
            # 计算收益率
            returns_matrix = self.calculate_returns(price_matrix, period)
            
            # 计算RPS
            rps_data = self.calculate_rps_for_period(returns_matrix, period)
            
            if rps_data:
                all_rps_data[f'RPS{period}'] = rps_data
                print(f"✅ RPS{period} 计算完成，包含 {len(rps_data)} 只股票")
            else:
                print(f"❌ RPS{period} 计算失败")
        
        # 4. 保存结果
        if all_rps_data:
            self.save_results(all_rps_data)
            
            # 生成统计报告
            timestamp = datetime.now().strftime('%Y%m%d')
            self.generate_summary_report(all_rps_data, self.output_dir, timestamp)
        
        total_time = time.time() - start_time
        print(f"\n🎉 === 多周期RPS计算完成 ===")
        print(f"⏱️  总耗时: {total_time:.2f} 秒")
        print(f"📊 计算周期: {list(all_rps_data.keys())}")
        
        return all_rps_data
    
    def backup_existing_results(self):
        """
        备份现有的RPS结果文件
        """
        if not os.path.exists(self.output_dir):
            return
        
        # 检查是否有文件需要备份
        existing_files = [f for f in os.listdir(self.output_dir) 
                         if f.endswith('.csv') or f.endswith('.pkl') or f.endswith('.txt')]
        
        if not existing_files:
            print("📁 没有需要备份的文件")
            return
        
        # 创建备份目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join('back', f'rps_results_backup_{timestamp}')
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            print(f"📁 创建备份目录: {backup_dir}")
            
            # 移动现有文件到备份目录
            moved_count = 0
            for file_name in existing_files:
                src_path = os.path.join(self.output_dir, file_name)
                dst_path = os.path.join(backup_dir, file_name)
                shutil.move(src_path, dst_path)
                moved_count += 1
            
            print(f"✅ 成功备份 {moved_count} 个文件到: {backup_dir}")
            
        except Exception as e:
            print(f"❌ 备份文件时发生错误: {e}")
    
    def save_results(self, all_rps_data):
        """
        保存RPS计算结果
        
        参数:
        all_rps_data: dict, 所有周期的RPS数据
        """
        # 备份现有结果文件
        self.backup_existing_results()
        
        #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        date_str = datetime.now().strftime('%Y%m%d')
        
        # 保存为pickle文件
        pickle_file = os.path.join(self.output_dir, f'rps_periods_{date_str}.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_rps_data, f)
        print(f"💾 RPS数据保存到: {pickle_file}")
        
        # 保存为CSV文件（每个周期单独保存）
        for period_name, rps_data in all_rps_data.items():
            # 从period_name中提取周期数字，格式：RPS5 -> 5
            period_num = period_name.replace('RPS', '')
            csv_file = os.path.join(self.output_dir, f'RPS{period_num}_{date_str}.csv')
            
            # 转换为DataFrame格式
            all_records = []
            for stock_code, stock_rps in rps_data.items():
                for record in stock_rps:
                    all_records.append({
                        'stock_code': stock_code,
                        'date': record['date'],
                        'rps': record['rps']
                    })
            
            if all_records:
                df = pd.DataFrame(all_records)
                df.to_csv(csv_file, index=False)
                print(f"📄 {period_name} CSV保存到: {csv_file}")
    
    def generate_summary_report(self, all_rps_data, output_dir, timestamp):
        """
        生成RPS计算统计报告
        
        参数:
        all_rps_data: dict, 所有周期的RPS数据
        output_dir: str, 输出目录
        timestamp: str, 时间戳
        """
        report_file = os.path.join(output_dir, f'rps_periods_report_{timestamp}.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== RPS周期数据计算报告 ===\n")
            f.write(f"计算时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据目录: {self.data_dir}\n\n")
            
            f.write("计算周期统计:\n")
            for period_name, rps_data in all_rps_data.items():
                f.write(f"\n{period_name}:\n")
                f.write(f"  股票数量: {len(rps_data)}\n")
                
                # 计算每只股票的数据点数量
                data_points = [len(stock_rps) for stock_rps in rps_data.values()]
                if data_points:
                    f.write(f"  平均数据点: {np.mean(data_points):.1f}\n")
                    f.write(f"  最大数据点: {max(data_points)}\n")
                    f.write(f"  最小数据点: {min(data_points)}\n")
                
                # 获取最新的RPS值进行统计
                latest_rps_values = []
                for stock_rps in rps_data.values():
                    if stock_rps:
                        latest_rps_values.append(stock_rps[-1]['rps'])
                
                if latest_rps_values:
                    f.write(f"最新RPS统计:\n")
                    f.write(f"  平均值: {np.mean(latest_rps_values):.2f}\n")
                    f.write(f"  中位数: {np.median(latest_rps_values):.2f}\n")
                    f.write(f"  最大值: {np.max(latest_rps_values):.2f}\n")
                    f.write(f"  最小值: {np.min(latest_rps_values):.2f}\n")
                    f.write(f"  RPS>80的股票: {sum(1 for rps in latest_rps_values if rps > 80)}\n")
                    f.write(f"  RPS>90的股票: {sum(1 for rps in latest_rps_values if rps > 90)}\n")
        
        print(f"📊 统计报告保存到: {report_file}")

def main():
    """
    主函数
    """
    print("=== RPS周期数据计算器 ===")
    print("支持计算RPS5、RPS10、RPS20、RPS60、RPS120、RPS250")
    
    # 检查数据目录
    data_dir = "/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/market_data"
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 创建计算器
    calculator = RPSPeriodsCalculator(data_dir=data_dir)
    
    # 设置计算周期
    periods = [5, 10, 20, 60, 120, 250]
    
    print(f"\n配置信息:")
    print(f"数据目录: {data_dir}")
    print(f"计算周期: {periods}")
    
    # 开始计算
    try:
        result = calculator.calculate_multi_period_rps(periods=periods)
        
        if result:
            print("\n🎉 计算成功完成！")
            print("\n📊 结果概览:")
            for period_name, rps_data in result.items():
                print(f"  {period_name}: {len(rps_data)} 只股票")
                
                # 显示部分样本数据
                sample_stocks = list(rps_data.keys())[:3]
                for stock_code in sample_stocks:
                    recent_rps = rps_data[stock_code][-3:]  # 最近3天
                    print(f"    {stock_code} 最近RPS: {[item['rps'] for item in recent_rps]}")
        else:
            print("❌ 计算失败")
            
    except KeyboardInterrupt:
        print("\n⚠️  用户中断计算")
    except Exception as e:
        print(f"❌ 计算过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()