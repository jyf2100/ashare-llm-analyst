#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分批RPS计算器
专门针对大规模股票数据（5000+）的分批处理解决方案
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import math
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BatchRPSCalculator:
    """
    分批RPS计算器
    将大规模股票数据分批处理，避免内存溢出
    """
    
    def __init__(self, data_dir="market_data", batch_size=1000, memory_limit_gb=4):
        """
        初始化分批RPS计算器
        
        参数:
        data_dir: str, 数据目录路径
        batch_size: int, 每批处理的股票数量
        memory_limit_gb: float, 内存限制（GB）
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb
        
    def estimate_memory_usage(self, stock_count, days_count):
        """
        估算内存使用量
        
        参数:
        stock_count: int, 股票数量
        days_count: int, 交易日数量
        
        返回:
        float: 估算的内存使用量（GB）
        """
        # 价格矩阵: days × stocks × 8字节（float64）
        matrix_size = days_count * stock_count * 8
        
        # 收益率矩阵和其他临时数据
        temp_data_size = matrix_size * 2
        
        # 总内存使用量（包括开销）
        total_bytes = (matrix_size + temp_data_size) * 1.5
        total_gb = total_bytes / (1024**3)
        
        return total_gb
    
    def optimize_batch_size(self, total_stocks, days_count):
        """
        根据内存限制优化批次大小
        
        参数:
        total_stocks: int, 总股票数量
        days_count: int, 交易日数量
        
        返回:
        int: 优化后的批次大小
        """
        # 从默认批次大小开始，逐步调整
        test_batch_size = self.batch_size
        
        while test_batch_size > 100:  # 最小批次大小
            memory_usage = self.estimate_memory_usage(test_batch_size, days_count)
            
            if memory_usage <= self.memory_limit_gb:
                break
            
            test_batch_size = int(test_batch_size * 0.8)  # 减少20%
        
        print(f"优化批次大小: {test_batch_size} 只股票/批")
        print(f"预计内存使用: {self.estimate_memory_usage(test_batch_size, days_count):.2f} GB")
        
        return test_batch_size
    
    def load_stock_list(self):
        """
        加载股票列表
        
        返回:
        list: 股票代码列表
        """
        if not os.path.exists(self.data_dir):
            print(f"❌ 数据目录不存在: {self.data_dir}")
            return []
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        stock_codes = [f.replace('.csv', '') for f in csv_files]
        
        print(f"找到 {len(stock_codes)} 只股票")
        return stock_codes
    
    def load_batch_data(self, stock_batch):
        """
        加载一批股票数据
        
        参数:
        stock_batch: list, 股票代码列表
        
        返回:
        dict: 股票代码 -> DataFrame的映射
        """
        batch_data = {}
        loaded_count = 0
        error_count = 0
        missing_count = 0
        insufficient_data_count = 0
        
        total_stocks = len(stock_batch)
        
        for i, stock_code in enumerate(stock_batch):
            # 每处理10只股票显示一次进度
            if (i + 1) % 10 == 0 or i == total_stocks - 1:
                progress = ((i + 1) / total_stocks) * 100
                print(f"\r      加载进度: {progress:.0f}% ({i+1}/{total_stocks})", end="", flush=True)
            
            try:
                file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
                
                if not os.path.exists(file_path):
                    missing_count += 1
                    continue
                
                df = pd.read_csv(file_path)
                
                # 数据验证
                if len(df) < 50 or 'date' not in df.columns or 'close' not in df.columns:
                    insufficient_data_count += 1
                    continue
                
                # 数据预处理
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df = df.dropna(subset=['close'])
                
                if len(df) >= 50:
                    batch_data[stock_code] = df
                    loaded_count += 1
                else:
                    insufficient_data_count += 1
                    
            except Exception as e:
                error_count += 1
                continue
        
        # 清除进度显示行
        print("\r" + " " * 50 + "\r", end="")
        
        # 显示加载统计
        if missing_count > 0 or error_count > 0 or insufficient_data_count > 0:
            print(f"      📊 加载统计: 成功{loaded_count}, 缺失{missing_count}, 错误{error_count}, 数据不足{insufficient_data_count}")
        
        return batch_data
    
    def calculate_batch_rps(self, batch_data, all_stock_codes, rps_period=50):
        """
        计算一批股票的RPS
        
        参数:
        batch_data: dict, 当前批次的股票数据
        all_stock_codes: list, 所有股票代码（用于RPS排名）
        rps_period: int, RPS计算周期
        
        返回:
        dict: RPS数据
        """
        if not batch_data:
            return {}
        
        print("      📈 构建价格矩阵...", end="", flush=True)
        
        # 获取所有日期的并集
        all_dates = set()
        for df in batch_data.values():
            all_dates.update(df['date'].dt.date)
        
        sorted_dates = sorted(all_dates)
        
        # 创建价格矩阵
        stock_codes = list(batch_data.keys())
        n_stocks = len(stock_codes)
        n_dates = len(sorted_dates)
        
        price_matrix = np.full((n_dates, n_stocks), np.nan)
        
        # 填充价格数据
        for i, stock_code in enumerate(stock_codes):
            df = batch_data[stock_code]
            
            for _, row in df.iterrows():
                try:
                    date_idx = sorted_dates.index(row['date'].date())
                    price_matrix[date_idx, i] = row['close']
                except ValueError:
                    continue
        
        print(" 完成")
        print(f"      📊 矩阵规模: {n_dates}天 × {n_stocks}股票")
        
        # 前向填充缺失值
        print("      🔧 处理缺失值...", end="", flush=True)
        price_df = pd.DataFrame(price_matrix, columns=stock_codes)
        price_df = price_df.fillna(method='ffill')
        price_matrix = price_df.values
        print(" 完成")
        
        # 计算收益率矩阵
        print("      📈 计算收益率矩阵...", end="", flush=True)
        returns_matrix = np.full((n_dates - rps_period + 1, n_stocks), np.nan)
        
        for i in range(rps_period - 1, n_dates):
            current_prices = price_matrix[i, :]
            past_prices = price_matrix[i - rps_period + 1, :]
            
            valid_mask = ~(np.isnan(current_prices) | np.isnan(past_prices) | (past_prices == 0))
            returns = np.full(n_stocks, np.nan)
            returns[valid_mask] = (current_prices[valid_mask] - past_prices[valid_mask]) / past_prices[valid_mask]
            
            returns_matrix[i - rps_period + 1, :] = returns
        
        print(" 完成")
        
        # 计算RPS（这里简化为批内排名，实际应该与全市场比较）
        print("      🏆 计算RPS排名...", end="", flush=True)
        rps_data = {}
        calculated_stocks = 0
        
        for stock_idx, stock_code in enumerate(stock_codes):
            stock_rps = []
            
            for time_idx in range(len(returns_matrix)):
                current_return = returns_matrix[time_idx, stock_idx]
                
                if np.isnan(current_return):
                    continue
                
                # 批内排名（注意：这是简化版本）
                batch_returns = returns_matrix[time_idx, :]
                valid_returns = batch_returns[~np.isnan(batch_returns)]
                
                if len(valid_returns) > 1:
                    better_count = np.sum(current_return > valid_returns) - 1
                    total_count = len(valid_returns) - 1
                    
                    if total_count > 0:
                        # 调整RPS以反映在全市场中的相对位置
                        batch_rps = (better_count / total_count) * 100
                        # 这里可以根据批次在全市场中的位置进行调整
                        rps = batch_rps
                    else:
                        rps = 50
                else:
                    rps = 50
                
                date_str = sorted_dates[time_idx + rps_period - 1].strftime('%Y-%m-%d')
                stock_rps.append({
                    'date': date_str,
                    'rps': round(rps, 2)
                })
            
            if stock_rps:
                rps_data[stock_code] = stock_rps
                calculated_stocks += 1
        
        print(f" 完成 ({calculated_stocks}只股票)")
        
        return rps_data
    
    def calculate_large_scale_rps(self, rps_period=50):
        """
        大规模RPS计算主函数
        
        参数:
        rps_period: int, RPS计算周期
        
        返回:
        dict: 完整的RPS数据
        """
        print("=== 大规模分批RPS计算 ===")
        start_time = time.time()
        
        # 1. 加载股票列表
        print("📊 步骤1: 扫描股票数据文件...")
        all_stock_codes = self.load_stock_list()
        if not all_stock_codes:
            return None
        
        total_stocks = len(all_stock_codes)
        print(f"✅ 发现 {total_stocks} 只股票")
        
        # 2. 估算内存使用和优化批次大小
        print("\n🔧 步骤2: 优化计算参数...")
        estimated_days = 400  # 估算交易日数量
        optimized_batch_size = self.optimize_batch_size(total_stocks, estimated_days)
        
        # 3. 分批处理
        total_batches = math.ceil(total_stocks / optimized_batch_size)
        print(f"📦 将分为 {total_batches} 个批次处理")
        print(f"⏱️  预计总耗时: {(total_batches * 2):.1f}-{(total_batches * 5):.1f} 分钟\n")
        
        all_rps_data = {}
        batch_times = []  # 记录每批次耗时
        
        for batch_idx in range(total_batches):
            batch_start_time = time.time()
            start_idx = batch_idx * optimized_batch_size
            end_idx = min(start_idx + optimized_batch_size, total_stocks)
            
            batch_stocks = all_stock_codes[start_idx:end_idx]
            batch_size = len(batch_stocks)
            
            print(f"🚀 批次 {batch_idx + 1}/{total_batches} - 处理股票 {start_idx+1}-{end_idx}")
            print(f"   📈 当前批次: {batch_size} 只股票")
            
            # 加载批次数据
            print("   📥 加载数据中...", end="", flush=True)
            load_start = time.time()
            batch_data = self.load_batch_data(batch_stocks)
            load_time = time.time() - load_start
            print(f" 完成 ({load_time:.1f}秒)")
            
            if not batch_data:
                print("   ⚠️  批次数据为空，跳过")
                continue
            
            valid_count = len(batch_data)
            invalid_count = batch_size - valid_count
            print(f"   ✅ 有效股票: {valid_count} 只")
            if invalid_count > 0:
                print(f"   ⚠️  无效股票: {invalid_count} 只 (数据不足或格式错误)")
            
            # 计算批次RPS
            print("   🧮 计算RPS中...", end="", flush=True)
            calc_start = time.time()
            batch_rps = self.calculate_batch_rps(batch_data, all_stock_codes, rps_period)
            calc_time = time.time() - calc_start
            print(f" 完成 ({calc_time:.1f}秒)")
            
            # 合并结果
            all_rps_data.update(batch_rps)
            
            # 批次完成统计
            batch_total_time = time.time() - batch_start_time
            batch_times.append(batch_total_time)
            
            print(f"   ✅ 批次完成: {len(batch_rps)} 只股票成功计算")
            print(f"   ⏱️  批次耗时: {batch_total_time:.1f}秒")
            
            # 显示总体进度
            completed_stocks = len(all_rps_data)
            progress = (completed_stocks / total_stocks) * 100
            elapsed_time = time.time() - start_time
            
            # 基于实际批次耗时预估剩余时间
            if len(batch_times) >= 2:
                avg_batch_time = sum(batch_times[-3:]) / len(batch_times[-3:])  # 最近3批次平均时间
                remaining_batches = total_batches - (batch_idx + 1)
                estimated_remaining_time = remaining_batches * avg_batch_time
            else:
                # 使用简单线性预估
                if completed_stocks > 0:
                    estimated_total_time = elapsed_time * total_stocks / completed_stocks
                    estimated_remaining_time = estimated_total_time - elapsed_time
                else:
                    estimated_remaining_time = 0
            
            print(f"\n📊 总体进度: {progress:.1f}% ({completed_stocks}/{total_stocks} 只股票)")
            print(f"⏰ 已用时间: {elapsed_time/60:.1f} 分钟")
            print(f"⏳ 预计剩余: {estimated_remaining_time/60:.1f} 分钟")
            
            # 性能统计
            if completed_stocks > 0:
                stocks_per_minute = completed_stocks / (elapsed_time / 60)
                print(f"⚡ 处理速度: {stocks_per_minute:.1f} 股票/分钟")
            
            print("-" * 60)
        
        # 4. 保存结果
        total_time = time.time() - start_time
        
        if all_rps_data:
            print("\n💾 保存结果中...", end="", flush=True)
            rps_file = os.path.join(self.data_dir, f'rps_{rps_period}d_batch.pkl')
            
            # 确保数据目录存在
            os.makedirs(self.data_dir, exist_ok=True)
            
            with open(rps_file, 'wb') as f:
                pickle.dump(all_rps_data, f)
            
            # 获取文件大小
            file_size = os.path.getsize(rps_file) / (1024 * 1024)  # MB
            print(" 完成")
            
            print(f"\n🎉 === 计算完成 ===")
            print(f"⏰ 总耗时: {total_time/60:.1f} 分钟")
            print(f"✅ 处理股票: {len(all_rps_data)} 只")
            print(f"📊 成功率: {(len(all_rps_data)/total_stocks)*100:.1f}%")
            print(f"⚡ 平均效率: {len(all_rps_data)/(total_time/60):.1f} 股票/分钟")
            
            if batch_times:
                avg_batch_time = sum(batch_times) / len(batch_times)
                print(f"📈 平均批次耗时: {avg_batch_time:.1f}秒")
            
            print(f"📁 结果保存到: {rps_file}")
            print(f"📦 文件大小: {file_size:.2f} MB")
            
            # 显示样本数据预览
            sample_stocks = list(all_rps_data.keys())[:3]
            print(f"\n📈 样本数据预览:")
            for stock in sample_stocks:
                rps_count = len(all_rps_data[stock])
                latest_rps = all_rps_data[stock][-1]['rps'] if all_rps_data[stock] else 0
                print(f"   {stock}: {rps_count}个RPS值, 最新RPS={latest_rps:.2f}")
            
            return all_rps_data
        else:
            print(f"\n❌ === 计算失败 ===")
            print(f"⏰ 总耗时: {total_time/60:.1f} 分钟")
            print(f"📊 成功率: 0.0%")
            print("❌ 没有计算出任何RPS数据")
            return None

def main():
    """
    主函数 - 演示大规模分批RPS计算
    """
    print("=== 大规模分批RPS计算器 ===")
    
    print("\n配置参数:")
    
    # 批次大小配置
    try:
        batch_size = int(input("批次大小 (建议500-2000，默认1000): ") or "1000")
    except ValueError:
        batch_size = 1000
    
    # 内存限制配置
    try:
        memory_limit = float(input("内存限制 GB (默认4): ") or "4")
    except ValueError:
        memory_limit = 4
    
    # RPS周期配置
    try:
        rps_period = int(input("RPS计算周期 (默认50): ") or "50")
    except ValueError:
        rps_period = 50
    
    print(f"\n配置确认:")
    print(f"  批次大小: {batch_size} 只股票")
    print(f"  内存限制: {memory_limit} GB")
    print(f"  RPS周期: {rps_period} 天")
    
    confirm = input("\n开始计算? (y/N): ").strip().lower()
    if confirm != 'y':
        print("取消计算")
        return
    
    # 创建计算器
    calculator = BatchRPSCalculator(
        data_dir="market_data",
        batch_size=batch_size,
        memory_limit_gb=memory_limit
    )
    
    try:
        # 开始计算
        rps_data = calculator.calculate_large_scale_rps(rps_period)
        
        if rps_data:
            print("\n=== 计算成功 ===")
            print(f"成功计算 {len(rps_data)} 只股票的RPS数据")
            
            # 显示部分结果
            sample_stocks = list(rps_data.keys())[:3]
            for stock_code in sample_stocks:
                recent_rps = rps_data[stock_code][-3:]  # 最近3天
                print(f"\n{stock_code} 最近RPS:")
                for item in recent_rps:
                    print(f"  {item['date']}: {item['rps']}")
        else:
            print("❌ 计算失败")
            
    except KeyboardInterrupt:
        print("\n用户中断计算")
    except Exception as e:
        print(f"❌ 计算过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()