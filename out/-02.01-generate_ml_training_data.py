#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习训练数据生成器
从market_data和rps_results目录读取数据，生成训练特征和标签
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
import shutil
from datetime import datetime
from typing import Dict, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLTrainingDataGenerator:
    """机器学习训练数据生成器"""
    
    def __init__(self, market_data_dir="market_data", rps_data_dir="rps_results"):
        self.market_data_dir = market_data_dir
        self.rps_data_dir = rps_data_dir
        self.stock_data = {}
        self.rps_data = {}  # 存储多周期RPS数据: {stock_code: {period: series}}
        self.rps_periods = [5, 10, 20]  # 支持的RPS周期
    
    def backup_existing_training_data(self, output_dir="training_data"):
        """备份现有的训练数据文件"""
        if not os.path.exists(output_dir):
            return
        
        # 查找需要备份的文件类型
        backup_extensions = ['.npy', '.csv', '.pkl', '.txt']
        files_to_backup = []
        
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path) and any(file.endswith(ext) for ext in backup_extensions):
                files_to_backup.append(file)
        
        if not files_to_backup:
            logger.info(f"训练数据目录 {output_dir} 中没有需要备份的文件")
            return
        
        # 创建备份目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join('back', f'training_data_backup_{timestamp}')
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            logger.info(f"创建备份目录: {backup_dir}")
            
            # 移动文件到备份目录
            for file in files_to_backup:
                src_path = os.path.join(output_dir, file)
                dst_path = os.path.join(backup_dir, file)
                shutil.move(src_path, dst_path)
                logger.info(f"备份文件: {file} -> {backup_dir}")
            
            logger.info(f"成功备份 {len(files_to_backup)} 个训练数据文件到 {backup_dir}")
            
        except Exception as e:
            logger.error(f"备份训练数据文件时出错: {e}")
            raise
    
    def load_market_data(self, max_stocks=50):
        """加载股票市场数据"""
        logger.info("开始加载市场数据...")
        
        if not os.path.exists(self.market_data_dir):
            logger.error(f"目录不存在: {self.market_data_dir}")
            return False
        
        csv_files = [f for f in os.listdir(self.market_data_dir) if f.endswith('.csv')]
        if max_stocks is not None:
            csv_files = csv_files[:max_stocks]
        
        for csv_file in csv_files:
            try:
                file_path = os.path.join(self.market_data_dir, csv_file)
                stock_code = csv_file.replace('.csv', '')
                
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
                
                # 基本数据清洗
                df = df.dropna(subset=['close'])
                df = df[df['close'] > 0]
                
                if len(df) >= 100:  # 至少100天数据
                    self.stock_data[stock_code] = df
                    
            except Exception as e:
                logger.warning(f"加载 {csv_file} 失败: {e}")
        
        logger.info(f"成功加载 {len(self.stock_data)} 只股票数据")
        return len(self.stock_data) > 0
    
    def load_rps_data(self):
        """加载多周期RPS数据（CSV格式）"""
        if not os.path.exists(self.rps_data_dir):
            logger.warning(f"RPS目录不存在: {self.rps_data_dir}")
            return False
        
        loaded_periods = []
        
        # 获取最新日期的RPS文件
        csv_files = [f for f in os.listdir(self.rps_data_dir) if f.endswith('.csv')]
        if not csv_files:
            logger.warning(f"在{self.rps_data_dir}中没有找到CSV文件")
            return False
        
        # 按日期排序，获取最新的文件
        csv_files.sort(reverse=True)
        latest_date = None
        for csv_file in csv_files:
            if 'RPS' in csv_file:
                # 提取日期，格式：RPS5_20250812.csv
                parts = csv_file.split('_')
                if len(parts) >= 2:
                    date_part = parts[1].replace('.csv', '')
                    if latest_date is None:
                        latest_date = date_part
                    break
        
        if latest_date is None:
            logger.warning("无法确定RPS文件的日期")
            return False
        
        logger.info(f"使用日期为 {latest_date} 的RPS数据")
        
        # 加载每个周期的RPS数据
        for period in self.rps_periods:
            rps_file = f"RPS{period}_{latest_date}.csv"
            file_path = os.path.join(self.rps_data_dir, rps_file)
            
            if not os.path.exists(file_path):
                logger.warning(f"RPS{period}文件不存在: {file_path}")
                continue
                
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 检查必要的列
                if not all(col in df.columns for col in ['stock_code', 'date', 'rps']):
                    logger.warning(f"RPS{period}文件格式不正确，缺少必要列")
                    continue
                
                # 转换日期格式
                df['date'] = pd.to_datetime(df['date'])
                
                # 按股票代码分组处理
                for stock_code, group in df.groupby('stock_code'):
                    if stock_code not in self.rps_data:
                        self.rps_data[stock_code] = {}
                    
                    # 创建时间序列
                    rps_series = group.set_index('date')['rps'].sort_index()
                    self.rps_data[stock_code][f'rps{period}'] = rps_series
                
                loaded_periods.append(period)
                unique_stocks = df['stock_code'].nunique()
                logger.info(f"加载RPS{period}文件: {rps_file}，股票数量: {unique_stocks}")
                
            except Exception as e:
                logger.warning(f"加载RPS{period}文件失败: {e}")
        
        logger.info(f"成功加载RPS周期: {loaded_periods}")
        return len(loaded_periods) > 0
    
    def calculate_features(self, df):
        """计算技术指标特征"""
        data = df.copy()
        
        # 移动平均线
        data['ma5'] = data['close'].rolling(5).mean()
        data['ma20'] = data['close'].rolling(20).mean()
        data['ma60'] = data['close'].rolling(60).mean()
        
        # 价格相对位置
        data['price_position'] = (data['close'] - data['close'].rolling(20).min()) / \
                               (data['close'].rolling(20).max() - data['close'].rolling(20).min())
        
        # 收益率
        data['return_1d'] = data['close'].pct_change()
        data['return_5d'] = data['close'].pct_change(5)
        data['return_20d'] = data['close'].pct_change(20)
        
        # 波动率
        data['volatility'] = data['return_1d'].rolling(20).std()
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        return data
    
    def generate_training_data(self, output_dir="training_data"):
        """生成训练数据"""
        logger.info("开始生成训练数据...")
        
        # 备份现有的训练数据文件
        self.backup_existing_training_data(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_features = []
        all_labels = []
        all_info = []
        
        for stock_code, data in self.stock_data.items():
            try:
                # 计算技术指标
                data_with_features = self.calculate_features(data)
                
                # 检查RPS数据质量
                rps_quality_passed = True
                rps_coverage_info = {}
                
                # 添加多周期RPS特征并检查数据质量
                if stock_code in self.rps_data:
                    rps_features_added = []
                    for period in self.rps_periods:
                        rps_key = f'rps{period}'
                        if rps_key in self.rps_data[stock_code]:
                            rps_series = self.rps_data[stock_code][rps_key]
                            rps_values = data_with_features.index.map(
                                lambda x: self._get_rps_value(rps_series, x)
                            )
                            
                            # 计算有效样本覆盖率
                            total_samples = len(rps_values)
                            valid_samples = len(rps_values.dropna())
                            coverage_rate = valid_samples / total_samples if total_samples > 0 else 0
                            rps_coverage_info[rps_key] = coverage_rate
                            
                            # 检查覆盖率是否满足要求（>80%）
                            if coverage_rate <= 0.80:
                                rps_quality_passed = False
                                logger.warning(f"股票 {stock_code} {rps_key}数据质量不足: 覆盖率{coverage_rate:.2%} <= 80%")
                            
                            # 确保所有RPS值都是有效的
                            data_with_features[rps_key] = rps_values.fillna(50.0)
                            # 替换任何异常值
                            data_with_features[rps_key] = data_with_features[rps_key].apply(
                                lambda x: 50.0 if x < 0 or x > 100 else x
                            )
                            rps_features_added.append(rps_key)
                        else:
                            # 如果某个周期的RPS数据不存在，质量检查失败
                            rps_quality_passed = False
                            logger.warning(f"股票 {stock_code}: 缺少{rps_key}数据")
                            data_with_features[rps_key] = 50.0  # 默认值
                    
                    if rps_quality_passed:
                        logger.info(f"股票 {stock_code}: RPS数据质量检查通过，覆盖率 {rps_coverage_info}")
                    else:
                        logger.warning(f"股票 {stock_code}: RPS数据质量检查失败，跳过该股票")
                        continue  # 跳过质量不合格的股票
                        
                else:
                    # 没有RPS数据的股票直接跳过
                    logger.warning(f"股票 {stock_code}: 未找到RPS数据，跳过该股票")
                    continue
                
                # 使用历史数据（排除最近30天）
                train_data = data_with_features.iloc[:-30]
                
                if len(train_data) < 100:
                    continue
                
                # 计算未来5日收益率作为标签
                future_returns = train_data['close'].shift(-5) / train_data['close'] - 1
                labels = (future_returns > 0.05).astype(int)  # 5%收益率阈值
                
                # 选择特征列（包含多周期RPS）
                rps_cols = [f'rps{period}' for period in self.rps_periods]
                feature_cols = ['close', 'ma5', 'ma20', 'ma60', 'price_position', 
                              'return_1d', 'return_5d', 'return_20d', 'volatility', 'rsi'] + rps_cols
                
                features_data = train_data[feature_cols].dropna()
                labels_data = labels.dropna()
                
                # 对齐数据
                common_idx = features_data.index.intersection(labels_data.index)
                if len(common_idx) < 50:
                    continue
                
                features_aligned = features_data.loc[common_idx]
                labels_aligned = labels_data.loc[common_idx]
                
                # 添加到总数据集
                for idx in common_idx:
                    all_features.append(features_aligned.loc[idx].values)
                    all_labels.append(labels_aligned.loc[idx])
                    all_info.append({'stock_code': stock_code, 'date': idx.strftime('%Y-%m-%d')})
                
            except Exception as e:
                logger.warning(f"处理股票 {stock_code} 失败: {e}")
                continue
        
        if len(all_features) == 0:
            logger.error("未生成任何训练样本")
            return False
        
        # 转换为numpy数组
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # 保存数据
        timestamp = datetime.now().strftime('%Y%m%d')
        
        np.save(os.path.join(output_dir, f'features_{timestamp}.npy'), X)
        np.save(os.path.join(output_dir, f'labels_{timestamp}.npy'), y)
        
        # 保存为CSV
        df_result = pd.DataFrame(X, columns=feature_cols)
        df_result['label'] = y
        df_result['stock_code'] = [info['stock_code'] for info in all_info]
        df_result['date'] = [info['date'] for info in all_info]
        
        df_result.to_csv(os.path.join(output_dir, f'training_data_{timestamp}.csv'), index=False)
        
        logger.info(f"训练数据生成完成:")
        logger.info(f"  样本数: {len(X)}")
        logger.info(f"  特征数: {X.shape[1]}")
        logger.info(f"  正样本比例: {y.mean():.2%}")
        logger.info(f"  保存到: {output_dir}")
        
        return True
    
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
        return np.nan  # 返回NaN用于质量检查

def main():
    """主函数"""
    print("机器学习训练数据生成器")
    print("=" * 40)
    
    generator = MLTrainingDataGenerator()
    
    # 加载数据 - 加载所有股票数据
    if not generator.load_market_data(max_stocks=None):
        print("❌ 市场数据加载失败")
        return
    
    generator.load_rps_data()
    
    # 生成训练数据
    if generator.generate_training_data():
        print("✅ 训练数据生成成功")
        print("\n使用方法:")
        print("import numpy as np")
        print("X = np.load('training_data/features_YYYYMMDD.npy')")
        print("y = np.load('training_data/labels_YYYYMMDD.npy')")
    else:
        print("❌ 训练数据生成失败")

if __name__ == "__main__":
    main()