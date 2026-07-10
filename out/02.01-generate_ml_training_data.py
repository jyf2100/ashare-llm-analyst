#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版机器学习训练数据生成器
专为中线投资策略设计，包含多目标标签、增强特征工程和智能数据质量控制
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
import shutil
import yaml
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MLTrainingConfig:
    """机器学习训练配置类"""
    # 预测周期配置
    PREDICTION_HORIZONS: List[int] = None
    RETURN_THRESHOLDS: List[float] = None
    
    # 特征窗口配置
    FEATURE_WINDOWS: Dict = None
    
    # RPS配置
    RPS_PERIODS: List[int] = None
    
    # 数据质量配置
    MIN_TRAINING_SAMPLES: int = 200
    MIN_COVERAGE_RATE: float = 0.70
    VALIDATION_SPLIT_RATIO: float = 0.2
    
    # 样本平衡配置
    USE_SAMPLE_WEIGHTS: bool = True
    BALANCE_METHOD: str = 'balanced'
    
    def __post_init__(self):
        if self.PREDICTION_HORIZONS is None:
            self.PREDICTION_HORIZONS = [5, 10, 20]  # 多周期预测
        if self.RETURN_THRESHOLDS is None:
            self.RETURN_THRESHOLDS = [0.03, 0.05, 0.08]  # 不同收益率阈值
        if self.FEATURE_WINDOWS is None:
            self.FEATURE_WINDOWS = {
                'short': [5, 10],
                'medium': [20, 30], 
                'long': [60, 120]
            }
        if self.RPS_PERIODS is None:
            self.RPS_PERIODS = [5, 10, 20, 60]  # 扩展RPS周期

class OptimizedMLTrainingDataGenerator:
    """优化版机器学习训练数据生成器"""
    
    def __init__(self, market_data_dir="market_data", rps_data_dir="rps_results", config=None):
        self.market_data_dir = market_data_dir
        self.rps_data_dir = rps_data_dir
        self.config = config or MLTrainingConfig()
        self.stock_data = {}
        self.rps_data = {}  # 存储多周期RPS数据
        self.market_index_data = None  # 市场指数数据
        self.feature_importance = {}  # 特征重要性
        
    def backup_existing_training_data(self, output_dir="training_data"):
        """备份现有的训练数据文件"""
        if not os.path.exists(output_dir):
            return
        
        backup_extensions = ['.npy', '.csv', '.pkl', '.txt', '.yaml']
        files_to_backup = []
        
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path) and any(file.endswith(ext) for ext in backup_extensions):
                files_to_backup.append(file)
        
        if not files_to_backup:
            logger.info(f"训练数据目录 {output_dir} 中没有需要备份的文件")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join('back', f'training_data_backup_{timestamp}')
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            logger.info(f"创建备份目录: {backup_dir}")
            
            for file in files_to_backup:
                src_path = os.path.join(output_dir, file)
                dst_path = os.path.join(backup_dir, file)
                shutil.move(src_path, dst_path)
                logger.info(f"备份文件: {file} -> {backup_dir}")
            
            logger.info(f"成功备份 {len(files_to_backup)} 个训练数据文件到 {backup_dir}")
            
        except Exception as e:
            logger.error(f"备份训练数据文件时出错: {e}")
            raise
    
    def load_market_data(self, max_stocks=None):
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
                
                # 增强数据清洗
                df = self._enhanced_data_cleaning(df)
                
                if len(df) >= self.config.MIN_TRAINING_SAMPLES:
                    self.stock_data[stock_code] = df
                    
            except Exception as e:
                logger.warning(f"加载 {csv_file} 失败: {e}")
        
        logger.info(f"成功加载 {len(self.stock_data)} 只股票数据")
        return len(self.stock_data) > 0
    
    def _enhanced_data_cleaning(self, df):
        """增强数据清洗"""
        # 基本清洗
        df = df.dropna(subset=['close'])
        df = df[df['close'] > 0]
        
        # 异常值检测和处理
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # 使用IQR方法检测异常值
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 记录异常值数量
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    logger.debug(f"检测到 {len(outliers)} 个 {col} 异常值")
                
                # 限制异常值
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        # 价格一致性检查
        df = df[df['high'] >= df['low']]
        df = df[df['high'] >= df['close']]
        df = df[df['low'] <= df['close']]
        
        return df
    
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
        for period in self.config.RPS_PERIODS:
            rps_file = f"RPS{period}_{latest_date}.csv"
            file_path = os.path.join(self.rps_data_dir, rps_file)
            
            if not os.path.exists(file_path):
                logger.warning(f"RPS{period}文件不存在: {file_path}")
                continue
                
            try:
                df = pd.read_csv(file_path)
                
                if not all(col in df.columns for col in ['stock_code', 'date', 'rps']):
                    logger.warning(f"RPS{period}文件格式不正确，缺少必要列")
                    continue
                
                df['date'] = pd.to_datetime(df['date'])
                
                for stock_code, group in df.groupby('stock_code'):
                    if stock_code not in self.rps_data:
                        self.rps_data[stock_code] = {}
                    
                    rps_series = group.set_index('date')['rps'].sort_index()
                    self.rps_data[stock_code][f'rps{period}'] = rps_series
                
                loaded_periods.append(period)
                unique_stocks = df['stock_code'].nunique()
                logger.info(f"加载RPS{period}文件: {rps_file}，股票数量: {unique_stocks}")
                
            except Exception as e:
                logger.warning(f"加载RPS{period}文件失败: {e}")
        
        logger.info(f"成功加载RPS周期: {loaded_periods}")
        return len(loaded_periods) > 0
    
    def calculate_enhanced_features(self, df):
        """计算增强技术指标特征"""
        data = df.copy()
        
        # 基础移动平均线
        for window in [5, 10, 20, 30, 60, 120]:
            if len(data) >= window:
                data[f'ma{window}'] = data['close'].rolling(window).mean()
        
        # 价格相对位置
        data['price_position_20'] = (data['close'] - data['close'].rolling(20).min()) / \
                                   (data['close'].rolling(20).max() - data['close'].rolling(20).min())
        
        # 多周期收益率
        for period in [1, 3, 5, 10, 20, 60]:
            data[f'return_{period}d'] = data['close'].pct_change(period)
        
        # 波动率
        for window in [10, 20, 60]:
            if len(data) >= window:
                data[f'volatility_{window}d'] = data['return_1d'].rolling(window).std()
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # 布林带
        data['bb_middle'] = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # 成交量指标
        data['volume_ma_20'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma_20']
        
        # 趋势强度
        data['trend_strength'] = (data['ma20'] - data['ma60']) / data['ma60']
        
        # 动量指标
        data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
        
        return data
    
    def calculate_data_quality_score(self, stock_data, rps_data):
        """计算数据质量评分"""
        # 数据完整性
        completeness = len(stock_data.dropna()) / len(stock_data)
        
        # RPS覆盖率
        rps_coverage = 0
        if rps_data:
            total_rps_points = 0
            valid_rps_points = 0
            for period_key, rps_series in rps_data.items():
                total_rps_points += len(stock_data)
                valid_rps_points += len(rps_series.dropna())
            rps_coverage = valid_rps_points / total_rps_points if total_rps_points > 0 else 0
        
        # 综合评分
        quality_score = (completeness * 0.6 + rps_coverage * 0.4)
        
        return {
            'completeness': completeness,
            'rps_coverage': rps_coverage,
            'quality_score': quality_score
        }
    
    def generate_multi_target_labels(self, data):
        """生成多目标标签"""
        labels = {}
        
        for horizon in self.config.PREDICTION_HORIZONS:
            # 未来收益率
            future_returns = data['close'].shift(-horizon) / data['close'] - 1
            
            # 多阈值二分类标签
            for threshold in self.config.RETURN_THRESHOLDS:
                label_name = f'return_{horizon}d_gt_{int(threshold*100)}pct'
                labels[label_name] = (future_returns > threshold).astype(int)
            
            # 多级分类标签
            bins = [-np.inf, -0.1, -0.05, 0.05, 0.1, np.inf]
            label_names = ['大跌', '小跌', '震荡', '小涨', '大涨']
            multi_class_labels = pd.cut(future_returns, bins=bins, labels=range(5))
            labels[f'multi_class_{horizon}d'] = multi_class_labels.astype(float)
            
            # 连续标签（回归）
            labels[f'return_{horizon}d_continuous'] = future_returns
        
        return labels
    
    def detect_data_drift(self, current_data, historical_data, columns):
        """检测数据漂移"""
        drift_scores = {}
        
        for column in columns:
            if column in current_data.columns and column in historical_data.columns:
                current_values = current_data[column].dropna()
                historical_values = historical_data[column].dropna()
                
                if len(current_values) > 0 and len(historical_values) > 0:
                    # KS检验
                    ks_stat, p_value = stats.ks_2samp(historical_values, current_values)
                    drift_scores[column] = {
                        'ks_stat': ks_stat,
                        'p_value': p_value,
                        'drift_detected': p_value < 0.05
                    }
        
        return drift_scores
    
    def generate_training_data(self, output_dir="training_data"):
        """生成优化的训练数据"""
        logger.info("开始生成优化训练数据...")
        
        # 备份现有数据
        self.backup_existing_training_data(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        all_features = []
        all_labels = {}
        all_info = []
        quality_reports = []
        
        # 初始化标签字典
        for horizon in self.config.PREDICTION_HORIZONS:
            for threshold in self.config.RETURN_THRESHOLDS:
                label_name = f'return_{horizon}d_gt_{int(threshold*100)}pct'
                all_labels[label_name] = []
            all_labels[f'multi_class_{horizon}d'] = []
            all_labels[f'return_{horizon}d_continuous'] = []
        
        for stock_code, data in self.stock_data.items():
            try:
                # 计算增强特征
                data_with_features = self.calculate_enhanced_features(data)
                
                # 数据质量检查
                rps_data_for_stock = self.rps_data.get(stock_code, {})
                quality_info = self.calculate_data_quality_score(data_with_features, rps_data_for_stock)
                
                if quality_info['quality_score'] < self.config.MIN_COVERAGE_RATE:
                    logger.warning(f"股票 {stock_code}: 数据质量评分 {quality_info['quality_score']:.2%} 低于阈值，跳过")
                    continue
                
                # 添加RPS特征
                if stock_code in self.rps_data:
                    for period in self.config.RPS_PERIODS:
                        rps_key = f'rps{period}'
                        if rps_key in self.rps_data[stock_code]:
                            rps_series = self.rps_data[stock_code][rps_key]
                            rps_values = data_with_features.index.map(
                                lambda x: self._get_rps_value(rps_series, x)
                            )
                            data_with_features[rps_key] = rps_values.fillna(50.0)
                        else:
                            data_with_features[rps_key] = 50.0
                else:
                    for period in self.config.RPS_PERIODS:
                        data_with_features[f'rps{period}'] = 50.0
                
                # 使用历史数据（排除最近30天）
                train_data = data_with_features.iloc[:-30]
                
                if len(train_data) < self.config.MIN_TRAINING_SAMPLES:
                    continue
                
                # 生成多目标标签
                labels_dict = self.generate_multi_target_labels(train_data)
                
                # 选择特征列
                feature_cols = self._select_feature_columns(data_with_features)
                features_data = train_data[feature_cols].dropna()
                
                # 对齐所有标签数据
                common_idx = features_data.index
                for label_name, label_data in labels_dict.items():
                    label_data_clean = label_data.dropna()
                    common_idx = common_idx.intersection(label_data_clean.index)
                
                if len(common_idx) < 50:
                    continue
                
                # 添加到总数据集
                features_aligned = features_data.loc[common_idx]
                
                for idx in common_idx:
                    all_features.append(features_aligned.loc[idx].values)
                    
                    # 添加所有标签
                    for label_name, label_data in labels_dict.items():
                        if label_name not in all_labels:
                            all_labels[label_name] = []
                        all_labels[label_name].append(label_data.loc[idx])
                    
                    all_info.append({
                        'stock_code': stock_code, 
                        'date': idx.strftime('%Y-%m-%d'),
                        'quality_score': quality_info['quality_score']
                    })
                
                quality_reports.append({
                    'stock_code': stock_code,
                    **quality_info,
                    'samples_count': len(common_idx)
                })
                
            except Exception as e:
                logger.warning(f"处理股票 {stock_code} 失败: {e}")
                continue
        
        if len(all_features) == 0:
            logger.error("未生成任何训练样本")
            return False
        
        # 转换为numpy数组
        X = np.array(all_features)
        
        # 保存数据
        timestamp = datetime.now().strftime('%Y%m%d')
        feature_cols = self._select_feature_columns(data_with_features)
        
        # 保存特征数据
        np.save(os.path.join(output_dir, f'features_{timestamp}.npy'), X)
        
        # 保存所有标签
        for label_name, label_values in all_labels.items():
            if len(label_values) > 0:
                y = np.array(label_values)
                np.save(os.path.join(output_dir, f'labels_{label_name}_{timestamp}.npy'), y)
        
        # 保存为CSV
        df_result = pd.DataFrame(X, columns=feature_cols)
        
        # 添加主要标签到CSV
        main_label = f'return_{self.config.PREDICTION_HORIZONS[0]}d_gt_{int(self.config.RETURN_THRESHOLDS[1]*100)}pct'
        if main_label in all_labels:
            df_result['label'] = all_labels[main_label]
        
        df_result['stock_code'] = [info['stock_code'] for info in all_info]
        df_result['date'] = [info['date'] for info in all_info]
        df_result['quality_score'] = [info['quality_score'] for info in all_info]
        
        df_result.to_csv(os.path.join(output_dir, f'training_data_{timestamp}.csv'), index=False)
        
        # 保存质量报告
        quality_df = pd.DataFrame(quality_reports)
        quality_df.to_csv(os.path.join(output_dir, f'quality_report_{timestamp}.csv'), index=False)
        
        # 保存配置
        config_dict = {
            'prediction_horizons': self.config.PREDICTION_HORIZONS,
            'return_thresholds': self.config.RETURN_THRESHOLDS,
            'rps_periods': self.config.RPS_PERIODS,
            'feature_columns': feature_cols,
            'label_columns': list(all_labels.keys()),
            'generation_time': datetime.now().isoformat(),
            'total_samples': len(X),
            'total_stocks': len(set([info['stock_code'] for info in all_info])),
            'feature_count': X.shape[1]
        }
        
        with open(os.path.join(output_dir, f'config_{timestamp}.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        # 输出统计信息
        logger.info(f"优化训练数据生成完成:")
        logger.info(f"  样本数: {len(X)}")
        logger.info(f"  特征数: {X.shape[1]}")
        logger.info(f"  股票数: {len(set([info['stock_code'] for info in all_info]))}")
        logger.info(f"  标签类型: {len(all_labels)}")
        
        if main_label in all_labels:
            main_label_values = np.array(all_labels[main_label])
            logger.info(f"  主标签正样本比例: {main_label_values.mean():.2%}")
        
        logger.info(f"  平均数据质量评分: {np.mean([info['quality_score'] for info in all_info]):.2%}")
        logger.info(f"  保存到: {output_dir}")
        
        return True
    
    def _select_feature_columns(self, data):
        """选择特征列"""
        # 基础价格特征
        base_features = ['close']
        
        # 移动平均线特征
        ma_features = [col for col in data.columns if col.startswith('ma') and col[2:].isdigit()]
        
        # 技术指标特征
        technical_features = [
            'price_position_20', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volume_ratio', 'trend_strength', 'momentum_20'
        ]
        
        # 收益率特征
        return_features = [col for col in data.columns if col.startswith('return_') and col.endswith('d')]
        
        # 波动率特征
        volatility_features = [col for col in data.columns if col.startswith('volatility_')]
        
        # RPS特征
        rps_features = [f'rps{period}' for period in self.config.RPS_PERIODS]
        
        # 合并所有特征
        all_features = base_features + ma_features + technical_features + return_features + volatility_features + rps_features
        
        # 过滤存在的特征
        available_features = [col for col in all_features if col in data.columns]
        
        return available_features
    
    def _get_rps_value(self, rps_series, target_date):
        """获取RPS值"""
        try:
            valid_dates = rps_series.index[rps_series.index <= target_date]
            if len(valid_dates) > 0:
                rps_value = float(rps_series[valid_dates.max()])
                if 0 <= rps_value <= 100:
                    return rps_value
        except Exception:
            pass
        return np.nan

def main():
    """主函数"""
    print("优化版机器学习训练数据生成器")
    print("=" * 50)
    
    # 创建配置
    config = MLTrainingConfig()
    
    # 创建生成器
    generator = OptimizedMLTrainingDataGenerator(config=config)
    
    # 加载数据
    if not generator.load_market_data(max_stocks=None):
        print("❌ 市场数据加载失败")
        return
    
    generator.load_rps_data()
    
    # 生成训练数据
    if generator.generate_training_data():
        print("✅ 优化训练数据生成成功")
        print("\n主要改进:")
        print("- 多目标标签设计（二分类、多分类、回归）")
        print("- 增强特征工程（MACD、布林带、成交量等）")
        print("- 智能数据质量控制")
        print("- 配置文件化管理")
        print("- 详细质量报告")
        print("\n使用方法:")
        print("import numpy as np")
        print("X = np.load('training_data/features_YYYYMMDD.npy')")
        print("y = np.load('training_data/labels_[LABEL_NAME]_YYYYMMDD.npy')")
    else:
        print("❌ 优化训练数据生成失败")

if __name__ == "__main__":
    main()