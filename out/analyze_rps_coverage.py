#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析RPS数据覆盖率分布
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RPSCoverageAnalyzer:
    def __init__(self, market_data_dir="market_data", rps_data_dir="rps_results"):
        self.market_data_dir = market_data_dir
        self.rps_data_dir = rps_data_dir
        self.rps_periods = [20, 60, 120]
        
    def load_market_data(self, max_stocks=100):
        """加载股票市场数据"""
        logger.info("开始加载市场数据...")
        market_data = {}
        
        if not os.path.exists(self.market_data_dir):
            logger.error(f"市场数据目录不存在: {self.market_data_dir}")
            return market_data
            
        stock_files = [f for f in os.listdir(self.market_data_dir) if f.endswith('.csv')]
        logger.info(f"找到 {len(stock_files)} 个股票数据文件")
        
        # 限制处理的股票数量
        if max_stocks and len(stock_files) > max_stocks:
            stock_files = stock_files[:max_stocks]
            logger.info(f"限制处理股票数量为: {max_stocks}")
            
        for i, filename in enumerate(stock_files):
            try:
                stock_code = filename.replace('.csv', '')
                file_path = os.path.join(self.market_data_dir, filename)
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                market_data[stock_code] = df
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已加载 {i + 1}/{len(stock_files)} 个股票")
                    
            except Exception as e:
                logger.warning(f"加载股票 {filename} 失败: {e}")
                
        logger.info(f"成功加载 {len(market_data)} 个股票的市场数据")
        return market_data
        
    def load_rps_data(self):
        """加载RPS数据"""
        logger.info("开始加载RPS数据...")
        rps_data = {}
        
        if not os.path.exists(self.rps_data_dir):
            logger.error(f"RPS数据目录不存在: {self.rps_data_dir}")
            return rps_data
            
        # 查找最新的多周期RPS数据文件
        import glob
        multi_period_files = glob.glob(os.path.join(self.rps_data_dir, "multi_period_rps_*.pkl"))
        
        if multi_period_files:
            # 使用最新的文件
            latest_file = max(multi_period_files)
            logger.info(f"找到多周期RPS数据文件: {latest_file}")
            
            try:
                import pickle
                with open(latest_file, 'rb') as f:
                    multi_rps_data = pickle.load(f)
                    
                # 转换为期望的格式
                for period in self.rps_periods:
                    rps_key = f'rps{period}'
                    rps_key_upper = f'RPS{period}'
                    if rps_key_upper in multi_rps_data:
                        # 将字典转换为DataFrame
                        rps_dict = multi_rps_data[rps_key_upper]
                        if isinstance(rps_dict, dict):
                            # 转换为DataFrame格式
                            df = pd.DataFrame.from_dict(rps_dict, orient='index')
                            df.index = pd.to_datetime(df.index)
                            rps_data[rps_key] = df
                            logger.info(f"加载RPS{period}数据: {df.shape}")
                        else:
                            rps_data[rps_key] = rps_dict
                            logger.info(f"加载RPS{period}数据: {type(rps_dict)}")
                    else:
                        logger.warning(f"多周期数据中未找到RPS{period}")
                        
            except Exception as e:
                logger.error(f"加载多周期RPS数据失败: {e}")
        else:
            logger.warning("未找到多周期RPS数据文件")
                
        return rps_data
        
    def _get_rps_value(self, rps_series, target_date):
        """获取RPS值"""
        try:
            valid_dates = rps_series.index[rps_series.index <= target_date]
            if len(valid_dates) > 0:
                rps_value = float(rps_series[valid_dates.max()])
                # 确保RPS值在合理范围内
                if 0 <= rps_value <= 100:
                    return rps_value
        except Exception as e:
            pass
        return np.nan
        
    def analyze_coverage(self):
        """分析RPS数据覆盖率"""
        logger.info("开始分析RPS数据覆盖率...")
        
        # 加载数据
        market_data = self.load_market_data()
        rps_data = self.load_rps_data()
        
        if not market_data or not rps_data:
            logger.error("数据加载失败")
            return
            
        coverage_stats = []
        
        for stock_code, stock_df in market_data.items():
            try:
                # 获取股票的日期范围
                dates = stock_df['date'].tolist()
                
                stock_coverage = {'stock_code': stock_code}
                
                # 分析每个RPS周期的覆盖率
                for period in self.rps_periods:
                    rps_key = f'rps{period}'
                    
                    if rps_key in rps_data and stock_code in rps_data[rps_key].columns:
                        rps_series = rps_data[rps_key][stock_code]
                        
                        # 计算覆盖率
                        rps_values = []
                        for date in dates:
                            rps_value = self._get_rps_value(rps_series, date)
                            rps_values.append(rps_value)
                            
                        rps_values = pd.Series(rps_values)
                        total_samples = len(rps_values)
                        valid_samples = len(rps_values.dropna())
                        coverage_rate = valid_samples / total_samples if total_samples > 0 else 0
                        
                        stock_coverage[rps_key] = coverage_rate
                    else:
                        stock_coverage[rps_key] = 0.0
                        
                coverage_stats.append(stock_coverage)
                
            except Exception as e:
                logger.warning(f"分析股票 {stock_code} 失败: {e}")
                
        # 转换为DataFrame并分析
        coverage_df = pd.DataFrame(coverage_stats)
        
        print("\n=== RPS数据覆盖率分析 ===")
        print(f"总股票数: {len(coverage_df)}")
        
        for period in self.rps_periods:
            rps_key = f'rps{period}'
            if rps_key in coverage_df.columns:
                coverage_col = coverage_df[rps_key]
                print(f"\n{rps_key.upper()}覆盖率统计:")
                print(f"  平均覆盖率: {coverage_col.mean():.2%}")
                print(f"  中位数覆盖率: {coverage_col.median():.2%}")
                print(f"  最小覆盖率: {coverage_col.min():.2%}")
                print(f"  最大覆盖率: {coverage_col.max():.2%}")
                
                # 统计不同覆盖率阈值下的股票数量
                thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]
                for threshold in thresholds:
                    count = (coverage_col >= threshold).sum()
                    percentage = count / len(coverage_col) * 100
                    print(f"  覆盖率>={threshold:.0%}的股票: {count}只 ({percentage:.1f}%)")
                    
        # 保存详细结果
        output_file = "rps_coverage_analysis.csv"
        coverage_df.to_csv(output_file, index=False)
        print(f"\n详细覆盖率数据已保存到: {output_file}")
        
        return coverage_df

def main():
    analyzer = RPSCoverageAnalyzer()
    analyzer.analyze_coverage()

if __name__ == "__main__":
    main()