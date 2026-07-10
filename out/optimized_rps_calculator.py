#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的RPS计算器
使用向量化计算大幅提升性能，适用于大规模股票数据
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class OptimizedRPSCalculator:
    """
    优化的RPS计算器
    使用向量化计算替代循环，大幅提升计算性能
    """
    
    def __init__(self, data_dir="market_data"):
        """
        初始化RPS计算器
        
        参数:
        data_dir: str, 数据目录路径
        """
        self.data_dir = data_dir
        
    def load_all_stock_data(self):
        """
        加载所有股票数据
        
        返回:
        dict: 股票代码 -> DataFrame的映射
        """
        print("=== 加载股票数据 ===")
        
        if not os.path.exists(self.data_dir):
            print(f"❌ 数据目录不存在: {self.data_dir}")
            return {}
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print("❌ 没有找到CSV数据文件")
            return {}
        
        print(f"找到 {len(csv_files)} 个股票数据文件")
        
        all_data = {}
        valid_count = 0
        
        for file in tqdm(csv_files, desc="读取数据"):
            try:
                stock_code = file.replace('.csv', '')
                file_path = os.path.join(self.data_dir, file)
                
                df = pd.read_csv(file_path)
                
                # 数据验证
                if len(df) < 50:  # 至少需要50天数据
                    continue
                    
                if 'date' not in df.columns or 'close' not in df.columns:
                    continue
                
                # 确保日期格式正确
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                # 确保收盘价为数值型
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df = df.dropna(subset=['close'])
                
                if len(df) >= 50:
                    all_data[stock_code] = df
                    valid_count += 1
                    
            except Exception as e:
                continue
        
        print(f"✓ 成功加载 {valid_count} 只股票数据")
        return all_data
    
    def create_aligned_price_matrix(self, all_data, rps_period=50):
        """
        创建对齐的价格矩阵，用于向量化计算
        
        参数:
        all_data: dict, 所有股票数据
        rps_period: int, RPS计算周期
        
        返回:
        tuple: (price_matrix, date_index, stock_codes)
        """
        print("=== 创建价格矩阵 ===")
        
        # 获取所有日期的并集
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df['date'].dt.date)
        
        # 排序日期
        sorted_dates = sorted(all_dates)
        print(f"日期范围: {sorted_dates[0]} 到 {sorted_dates[-1]}")
        
        # 创建价格矩阵
        stock_codes = list(all_data.keys())
        n_stocks = len(stock_codes)
        n_dates = len(sorted_dates)
        
        print(f"创建 {n_dates} x {n_stocks} 价格矩阵")
        
        # 初始化价格矩阵（NaN表示缺失数据）
        price_matrix = np.full((n_dates, n_stocks), np.nan)
        
        # 填充价格数据
        for i, stock_code in enumerate(tqdm(stock_codes, desc="构建矩阵")):
            df = all_data[stock_code]
            
            for _, row in df.iterrows():
                date_idx = sorted_dates.index(row['date'].date())
                price_matrix[date_idx, i] = row['close']
        
        # 前向填充缺失值（使用最近的有效价格）
        price_df = pd.DataFrame(price_matrix, columns=stock_codes)
        price_df = price_df.fillna(method='ffill')
        price_matrix = price_df.values
        
        print(f"✓ 价格矩阵创建完成: {price_matrix.shape}")
        return price_matrix, sorted_dates, stock_codes
    
    def calculate_vectorized_rps(self, price_matrix, date_index, stock_codes, rps_period=50):
        """
        使用向量化计算RPS
        
        参数:
        price_matrix: np.array, 价格矩阵 (日期 x 股票)
        date_index: list, 日期索引
        stock_codes: list, 股票代码列表
        rps_period: int, RPS计算周期
        
        返回:
        dict: 股票代码 -> RPS数据的映射
        """
        print(f"🚀 === 开始向量化RPS计算（周期: {rps_period}天）===")
        
        n_dates, n_stocks = price_matrix.shape
        rps_data = {}
        
        print(f"📊 计算规模: {n_dates}个交易日 × {n_stocks}只股票")
        print(f"📈 预计生成: {n_dates - rps_period + 1}个时间点的RPS数据")
        
        # 计算所有股票在所有时间点的收益率矩阵
        print("\n📈 步骤1: 计算收益率矩阵...")
        returns_matrix = np.full((n_dates - rps_period + 1, n_stocks), np.nan)
        
        calc_start_time = time.time()
        
        for i in tqdm(range(rps_period - 1, n_dates), desc="💹 计算收益率"):
            current_prices = price_matrix[i, :]
            past_prices = price_matrix[i - rps_period + 1, :]
            
            # 向量化计算收益率
            valid_mask = ~(np.isnan(current_prices) | np.isnan(past_prices) | (past_prices == 0))
            returns = np.full(n_stocks, np.nan)
            returns[valid_mask] = (current_prices[valid_mask] - past_prices[valid_mask]) / past_prices[valid_mask]
            
            returns_matrix[i - rps_period + 1, :] = returns
        
        returns_calc_time = time.time() - calc_start_time
        print(f"✅ 收益率矩阵计算完成，耗时: {returns_calc_time:.1f}秒")
        
        # 统计有效数据
        valid_returns_count = np.sum(~np.isnan(returns_matrix))
        total_returns_count = returns_matrix.size
        valid_ratio = (valid_returns_count / total_returns_count) * 100
        print(f"📊 有效收益率数据: {valid_returns_count}/{total_returns_count} ({valid_ratio:.1f}%)")
        
        print("\n🏆 步骤2: 计算RPS排名...")
        rps_start_time = time.time()
        
        # 为每只股票计算RPS
        processed_stocks = 0
        for stock_idx in tqdm(range(n_stocks), desc="🧮 计算RPS"):
            stock_code = stock_codes[stock_idx]
            stock_rps = []
            
            for time_idx in range(len(returns_matrix)):
                current_return = returns_matrix[time_idx, stock_idx]
                
                if np.isnan(current_return):
                    continue
                
                # 获取同期所有股票的收益率
                all_returns = returns_matrix[time_idx, :]
                valid_returns = all_returns[~np.isnan(all_returns)]
                
                if len(valid_returns) > 1:
                    # 向量化计算RPS：当前股票收益率超过多少比例的其他股票
                    better_count = np.sum(current_return > valid_returns) - 1  # 减去自己
                    total_count = len(valid_returns) - 1  # 减去自己
                    
                    if total_count > 0:
                        rps = (better_count / total_count) * 100
                    else:
                        rps = 50
                else:
                    rps = 50
                
                date_str = date_index[time_idx + rps_period - 1].strftime('%Y-%m-%d')
                stock_rps.append({
                    'date': date_str,
                    'rps': round(rps, 2)
                })
            
            if stock_rps:
                rps_data[stock_code] = stock_rps
                processed_stocks += 1
            
            # 每处理100只股票显示一次进度
            if (stock_idx + 1) % 100 == 0:
                elapsed = time.time() - rps_start_time
                progress = ((stock_idx + 1) / n_stocks) * 100
                speed = (stock_idx + 1) / elapsed
                remaining_time = (n_stocks - stock_idx - 1) / speed if speed > 0 else 0
                print(f"   📊 进度: {progress:.1f}% ({stock_idx+1}/{n_stocks}), 速度: {speed:.1f}股票/秒, 预计剩余: {remaining_time:.0f}秒")
        
        rps_calc_time = time.time() - rps_start_time
        total_calc_time = time.time() - calc_start_time
        
        print(f"\n✅ RPS计算完成!")
        print(f"📊 成功处理: {processed_stocks}/{n_stocks} 只股票 ({(processed_stocks/n_stocks)*100:.1f}%)")
        print(f"⏱️  RPS计算耗时: {rps_calc_time:.1f}秒")
        print(f"⏰ 总计算耗时: {total_calc_time:.1f}秒")
        print(f"⚡ 平均速度: {processed_stocks/total_calc_time:.1f} 股票/秒")
        
        return rps_data
    
    def calculate_rps_optimized(self, rps_period=50):
        """
        优化的RPS计算主函数
        
        参数:
        rps_period: int, RPS计算周期
        
        返回:
        dict: RPS数据
        """
        print(f"🚀 === 开始优化RPS计算 (周期: {rps_period}天) ===")
        start_time = time.time()
        
        # 1. 加载数据
        print("\n📊 步骤1: 加载股票数据...")
        all_data = self.load_all_stock_data()
        if not all_data:
            print("❌ 数据加载失败")
            return None
        
        load_time = time.time()
        print(f"✅ 数据加载完成，耗时: {load_time - start_time:.2f} 秒")
        print(f"📈 成功加载 {len(all_data)} 只股票数据")
        
        # 2. 创建价格矩阵
        print("\n🔧 步骤2: 创建价格矩阵...")
        price_matrix, date_index, stock_codes = self.create_aligned_price_matrix(all_data, rps_period)
        
        matrix_time = time.time()
        print(f"✅ 价格矩阵创建完成，耗时: {matrix_time - load_time:.2f} 秒")
        print(f"📊 矩阵规模: {price_matrix.shape[0]}天 × {price_matrix.shape[1]}股票")
        
        # 3. 向量化计算RPS
        print("\n🧮 步骤3: 向量化RPS计算...")
        rps_data = self.calculate_vectorized_rps(price_matrix, date_index, stock_codes, rps_period)
        
        calc_time = time.time()
        print(f"✅ RPS计算完成，耗时: {calc_time - matrix_time:.2f} 秒")
        
        # 4. 保存结果
        print("\n💾 步骤4: 保存结果...")
        rps_file = os.path.join(self.data_dir, f'rps_{rps_period}d_optimized.pkl')
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        with open(rps_file, 'wb') as f:
            pickle.dump(rps_data, f)
        
        # 获取文件大小
        file_size = os.path.getsize(rps_file) / (1024 * 1024)  # MB
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 === 计算完成 ===")
        print(f"⏰ 总耗时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
        print(f"📊 处理股票: {len(rps_data)} 只")
        print(f"📈 成功率: {(len(rps_data)/len(stock_codes))*100:.1f}%")
        print(f"⚡ 处理效率: {len(rps_data)/total_time:.2f} 股票/秒")
        print(f"📁 结果保存到: {rps_file}")
        print(f"📦 文件大小: {file_size:.2f} MB")
        
        # 显示性能分解
        print(f"\n📊 === 性能分解 ===")
        print(f"📥 数据加载: {load_time - start_time:.1f}秒 ({((load_time - start_time)/total_time)*100:.1f}%)")
        print(f"🔧 矩阵构建: {matrix_time - load_time:.1f}秒 ({((matrix_time - load_time)/total_time)*100:.1f}%)")
        print(f"🧮 RPS计算: {calc_time - matrix_time:.1f}秒 ({((calc_time - matrix_time)/total_time)*100:.1f}%)")
        print(f"💾 结果保存: {time.time() - calc_time:.1f}秒 ({((time.time() - calc_time)/total_time)*100:.1f}%)")
        
        # 显示样本数据
        if rps_data:
            sample_stocks = list(rps_data.keys())[:3]
            print(f"\n📈 样本数据预览:")
            for stock in sample_stocks:
                rps_count = len(rps_data[stock])
                latest_rps = rps_data[stock][-1]['rps'] if rps_data[stock] else 0
                print(f"   {stock}: {rps_count}个RPS值, 最新RPS={latest_rps:.2f}")
        
        return rps_data

def main():
    """
    主函数 - 演示优化的RPS计算
    """
    print("=== 优化RPS计算器演示 ===")
    
    calculator = OptimizedRPSCalculator(data_dir="market_data")
    
    print("\n选择计算模式:")
    print("1. 20日RPS计算")
    print("2. 50日RPS计算")
    print("3. 120日RPS计算")
    print("4. 自定义周期")
    
    try:
        choice = input("\n请选择模式 (1-4): ").strip()
        
        if choice == '1':
            rps_period = 20
        elif choice == '2':
            rps_period = 50
        elif choice == '3':
            rps_period = 120
        elif choice == '4':
            rps_period = int(input("请输入RPS计算周期（天数）: "))
        else:
            print("无效选择，使用默认20日RPS")
            rps_period = 20
        
        print(f"\n开始计算 {rps_period} 日RPS...")
        rps_data = calculator.calculate_rps_optimized(rps_period)
        
        if rps_data:
            print("\n=== 计算完成 ===")
            print(f"成功计算 {len(rps_data)} 只股票的RPS数据")
            
            # 显示部分结果
            sample_stocks = list(rps_data.keys())[:3]
            for stock_code in sample_stocks:
                recent_rps = rps_data[stock_code][-5:]  # 最近5天
                print(f"\n{stock_code} 最近RPS:")
                for item in recent_rps:
                    print(f"  {item['date']}: {item['rps']}")
        else:
            print("❌ RPS计算失败")
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()