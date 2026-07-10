#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的机器学习模型训练器
解决特征尺度问题和预测范围压缩问题
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import yaml
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedMLModelTrainer:
    """改进的机器学习模型训练器"""
    
    def __init__(self, training_data_dir="training_data", model_save_dir="models_improved"):
        self.training_data_dir = training_data_dir
        self.model_save_dir = model_save_dir
        self.feature_columns = []
        self.config = {}
        
        # 创建模型保存目录
        os.makedirs(model_save_dir, exist_ok=True)
    
    def load_config(self):
        """加载配置文件"""
        config_files = [f for f in os.listdir(self.training_data_dir) if f.startswith('config_') and f.endswith('.yaml')]
        if config_files:
            config_file = sorted(config_files)[-1]
            config_path = os.path.join(self.training_data_dir, config_file)
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.feature_columns = self.config.get('feature_columns', [])
            logger.info(f"加载配置文件: {config_file}")
            logger.info(f"特征数量: {len(self.feature_columns)}")
            return True
        return False
    
    def load_training_data(self, target_label=None):
        """加载训练数据"""
        if not os.path.exists(self.training_data_dir):
            logger.error(f"训练数据目录不存在: {self.training_data_dir}")
            return None, None, None
        
        # 加载配置
        if not self.load_config():
            logger.warning("未找到配置文件，使用默认设置")
        
        # 查找最新的训练数据文件
        feature_files = [f for f in os.listdir(self.training_data_dir) if f.startswith('features_') and f.endswith('.npy')]
        
        if not feature_files:
            logger.error("未找到特征数据文件")
            return None, None, None
        
        # 使用最新的特征文件
        latest_feature_file = sorted(feature_files)[-1]
        timestamp = latest_feature_file.split('_')[-1].replace('.npy', '')
        
        logger.info(f"加载特征数据: {latest_feature_file}")
        try:
            X = np.load(os.path.join(self.training_data_dir, latest_feature_file))
            logger.info(f"特征数据形状: {X.shape}")
        except Exception as e:
            logger.error(f"加载特征数据失败: {e}")
            return None, None, None
        
        # 如果未指定目标标签，使用默认标签
        if target_label is None:
            target_label = 'return_5d_continuous'
        
        label_file = f'labels_{target_label}_{timestamp}.npy'
        label_path = os.path.join(self.training_data_dir, label_file)
        
        if not os.path.exists(label_path):
            logger.error(f"标签文件不存在: {label_file}")
            available_labels = [f for f in os.listdir(self.training_data_dir) if f.startswith('labels_') and f.endswith('.npy')]
            logger.info(f"可用的标签文件: {available_labels}")
            return None, None, None
        
        try:
            y = np.load(label_path)
            logger.info(f"加载标签: {label_file}")
            logger.info(f"标签统计: 最小值={y.min():.4f}, 最大值={y.max():.4f}, 均值={y.mean():.4f}")
            
            return X, y, target_label
            
        except Exception as e:
            logger.error(f"加载标签数据失败: {e}")
            return None, None, None
    
    def improved_preprocess_data(self, X, y):
        """改进的数据预处理"""
        logger.info("开始改进的数据预处理...")
        
        # 检查特征统计
        logger.info(f"原始特征统计:")
        logger.info(f"  形状: {X.shape}")
        logger.info(f"  最小值: {X.min():.4f}")
        logger.info(f"  最大值: {X.max():.4f}")
        logger.info(f"  均值范围: {X.mean(axis=0).min():.4f} - {X.mean(axis=0).max():.4f}")
        logger.info(f"  标准差范围: {X.std(axis=0).min():.4f} - {X.std(axis=0).max():.4f}")
        
        # 检查并处理异常值
        if np.isnan(X).any():
            logger.warning("发现缺失值，使用均值填充")
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        
        if np.isinf(X).any():
            logger.warning("发现无穷值，进行处理")
            X = np.nan_to_num(X, posinf=np.nanmax(X[np.isfinite(X)]), neginf=np.nanmin(X[np.isfinite(X)]))
        
        # 使用RobustScaler代替StandardScaler，对异常值更鲁棒
        logger.info("使用RobustScaler进行特征标准化...")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 检查标准化后的特征统计
        logger.info(f"标准化后特征统计:")
        logger.info(f"  最小值: {X_scaled.min():.4f}")
        logger.info(f"  最大值: {X_scaled.max():.4f}")
        logger.info(f"  均值范围: {X_scaled.mean(axis=0).min():.4f} - {X_scaled.mean(axis=0).max():.4f}")
        logger.info(f"  标准差范围: {X_scaled.std(axis=0).min():.4f} - {X_scaled.std(axis=0).max():.4f}")
        
        return X_scaled, y, scaler
    
    def train_improved_model(self, X, y, target_label, test_size=0.2, random_state=42):
        """训练改进的模型"""
        logger.info(f"开始训练改进模型，目标标签: {target_label}")
        
        # 判断是否为回归任务
        is_regression = target_label.endswith('_continuous')
        
        # 分割训练集和测试集
        if is_regression:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
        
        # 创建模型
        if is_regression:
            # 回归模型 - 调整参数以获得更好的预测范围
            model = RandomForestRegressor(
                n_estimators=100,  # 增加树的数量
                max_depth=15,      # 增加深度以捕获更复杂的模式
                min_samples_split=5,  # 减少最小分割样本数
                min_samples_leaf=2,   # 减少叶子节点最小样本数
                max_features='sqrt',  # 使用sqrt特征选择
                random_state=random_state,
                n_jobs=-1
            )
        else:
            # 分类模型
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            )
        
        # 训练模型
        logger.info("开始训练模型...")
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估模型
        results = {
            'model_type': 'random_forest_improved',
            'target_label': target_label,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        if is_regression:
            # 回归评估指标
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"回归模型性能:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            
            # 检查预测范围
            logger.info(f"预测结果分析:")
            logger.info(f"  真实值范围: {y_test.min():.4f} - {y_test.max():.4f}")
            logger.info(f"  预测值范围: {y_pred.min():.4f} - {y_pred.max():.4f}")
            logger.info(f"  真实值均值: {y_test.mean():.4f}")
            logger.info(f"  预测值均值: {y_pred.mean():.4f}")
            logger.info(f"  真实值标准差: {y_test.std():.4f}")
            logger.info(f"  预测值标准差: {y_pred.std():.4f}")
            
            # 计算相关性
            correlation = np.corrcoef(y_test, y_pred)[0, 1]
            logger.info(f"  相关性: {correlation:.4f}")
            
            results.update({
                'rmse': rmse,
                'r2': r2,
                'correlation': correlation,
                'pred_range': (y_pred.min(), y_pred.max()),
                'true_range': (y_test.min(), y_test.max())
            })
        else:
            # 分类评估指标
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"分类模型性能:")
            logger.info(f"  准确率: {accuracy:.4f}")
            logger.info(f"  F1分数: {f1:.4f}")
            
            results.update({
                'accuracy': accuracy,
                'f1': f1
            })
        
        return model, results
    
    def save_improved_model(self, model, scaler, target_label, results):
        """保存改进的模型"""
        timestamp = datetime.now().strftime('%Y%m%d')
        
        # 保存模型
        model_filename = f"improved_random_forest_{target_label}_{timestamp}.pkl"
        model_path = os.path.join(self.model_save_dir, model_filename)
        joblib.dump(model, model_path)
        
        # 保存标准化器
        scaler_filename = f"improved_scaler_{target_label}_{timestamp}.pkl"
        scaler_path = os.path.join(self.model_save_dir, scaler_filename)
        joblib.dump(scaler, scaler_path)
        
        # 保存配置
        config = {
            'model_type': 'random_forest_improved',
            'target_label': target_label,
            'feature_columns': self.feature_columns,
            'scaler_type': 'RobustScaler',
            'model_results': results,
            'timestamp': timestamp
        }
        
        config_filename = f"improved_model_config_{target_label}_{timestamp}.yaml"
        config_path = os.path.join(self.model_save_dir, config_filename)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"模型保存成功:")
        logger.info(f"  模型: {model_path}")
        logger.info(f"  标准化器: {scaler_path}")
        logger.info(f"  配置: {config_path}")
        
        return True

def main():
    """主函数"""
    print("改进的机器学习模型训练器")
    print("=" * 50)
    
    trainer = ImprovedMLModelTrainer()
    
    # 测试的目标标签
    target_labels = [
        'return_5d_continuous',
        'return_10d_continuous', 
        'return_20d_continuous',
        'return_5d_gt_5pct'
    ]
    
    for target_label in target_labels:
        print(f"\n训练模型: {target_label}")
        print("-" * 40)
        
        try:
            # 加载训练数据
            X, y, actual_label = trainer.load_training_data(target_label)
            if X is None or y is None:
                print(f"❌ 跳过 {target_label}: 数据加载失败")
                continue
            
            # 改进的数据预处理
            X_processed, y_processed, scaler = trainer.improved_preprocess_data(X, y)
            
            # 训练改进的模型
            model, results = trainer.train_improved_model(X_processed, y_processed, actual_label)
            
            # 保存模型
            trainer.save_improved_model(model, scaler, actual_label, results)
            
            print(f"✅ {target_label} 训练完成")
            
        except Exception as e:
            print(f"❌ {target_label} 训练失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()