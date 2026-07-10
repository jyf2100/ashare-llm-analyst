#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版机器学习模型训练器 - 随机森林专用版
支持多目标标签和增强评估指标
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import shutil
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
from collections import defaultdict
import warnings
import time
import gc
from scipy import stats
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedMLModelTrainer:
    """优化版机器学习模型训练器 - 随机森林专用版"""
    
    def __init__(self, training_data_dir="training_data", model_save_dir="models", scaler_type="robust", enable_hyperparameter_tuning=False):
        self.training_data_dir = training_data_dir
        self.model_save_dir = model_save_dir
        self.scaler_type = scaler_type  # 标准化器类型：'standard' 或 'robust'
        self.enable_hyperparameter_tuning = enable_hyperparameter_tuning  # 是否启用超参数调优
        self.models = {}  # 存储多个模型
        self.scalers = {}  # 存储多个标准化器
        self.feature_columns = []
        self.label_columns = []
        self.config = {}
        
        # 数据缓存
        self._feature_cache = None
        self._feature_cache_timestamp = None
        self._processed_feature_cache = None
        self._scaler_cache = None
        
        # 创建模型保存目录
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 只支持随机森林模型
        self.model_types = {
            'random_forest': RandomForestClassifier
        }
        
        # 随机森林优化参数配置
        self.model_params = {
            'random_forest_classifier': {
                'n_estimators': 200,  # 增加树的数量以提高性能
                'max_depth': 20,      # 增加深度以捕获更复杂的模式
                'min_samples_split': 2,   # 更严格的分割条件
                'min_samples_leaf': 1,    # 更严格的叶子节点条件
                'max_features': 'sqrt',   # 使用sqrt特征选择
                'random_state': 42,
                'n_jobs': -1,
                'bootstrap': True,    # 启用bootstrap采样
                'oob_score': True,    # 计算袋外分数
                'class_weight': 'balanced'  # 处理类别不平衡
            },
            'random_forest_regressor': {
                'n_estimators': 200,  # 增加树的数量以提高性能
                'max_depth': 20,      # 增加深度以捕获更复杂的模式
                'min_samples_split': 2,   # 更严格的分割条件
                'min_samples_leaf': 1,    # 更严格的叶子节点条件
                'max_features': 'sqrt',   # 使用sqrt特征选择
                'random_state': 42,
                'n_jobs': -1,
                'bootstrap': True,    # 启用bootstrap采样
                'oob_score': True     # 计算袋外分数（回归任务不需要class_weight）
            }
        }
        
        # 超参数搜索网格（用于GridSearchCV）
        self.param_grids = {
            'random_forest_classifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'random_forest_regressor': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        }

    
    def backup_existing_models(self):
        """备份现有的模型文件和配置文件到back目录（只备份非当天的文件）"""
        # 获取当天日期字符串
        today_str = datetime.now().strftime('%Y%m%d')
        
        # 定义需要处理的目录和对应的文件类型
        directories_to_process = {
            self.model_save_dir: ['.pkl', '.joblib', '.model', '.yaml', '.yml'],
            self.training_data_dir: ['.yaml', '.yml']
        }
        
        all_files_to_backup = []
        
        # 遍历每个目录
        for directory, backup_extensions in directories_to_process.items():
            if not os.path.exists(directory):
                continue
                
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and any(file.endswith(ext) for ext in backup_extensions):
                    # 检查文件是否为当天生成的
                    file_stat = os.stat(file_path)
                    file_date = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y%m%d')
                    
                    # 只备份非当天的文件
                    if file_date != today_str:
                        all_files_to_backup.append((directory, file))
                    else:
                        logger.info(f"保留当天文件: {os.path.join(directory, file)}")
        
        if not all_files_to_backup:
            logger.info("没有需要备份的非当天文件")
            return
        
        # 创建back目录下的备份目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_base_dir = os.path.join('back', f'backup_{timestamp}')
        
        try:
            os.makedirs(backup_base_dir, exist_ok=True)
            logger.info(f"创建备份目录: {backup_base_dir}")
            
            # 移动文件到备份目录，保持原有的目录结构
            for directory, file in all_files_to_backup:
                # 确定目标备份子目录
                if directory == self.model_save_dir:
                    backup_subdir = os.path.join(backup_base_dir, 'models')
                elif directory == self.training_data_dir:
                    backup_subdir = os.path.join(backup_base_dir, 'training_data')
                else:
                    backup_subdir = backup_base_dir
                
                # 创建子目录
                os.makedirs(backup_subdir, exist_ok=True)
                
                # 移动文件
                src_path = os.path.join(directory, file)
                dst_path = os.path.join(backup_subdir, file)
                shutil.move(src_path, dst_path)
                logger.info(f"备份文件: {src_path} -> {dst_path}")
            
            logger.info(f"成功备份 {len(all_files_to_backup)} 个非当天文件到 {backup_base_dir}")
            logger.info(f"保留了当天生成的文件")
            
        except Exception as e:
            logger.error(f"备份文件时出错: {e}")
            raise
    
    def load_config(self):
        """加载配置文件"""
        config_files = [f for f in os.listdir(self.training_data_dir) if f.startswith('config_') and f.endswith('.yaml')]
        if config_files:
            config_file = sorted(config_files)[-1]
            config_path = os.path.join(self.training_data_dir, config_file)
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.feature_columns = self.config.get('feature_columns', [])
            self.label_columns = self.config.get('label_columns', [])
            logger.info(f"加载配置文件: {config_file}")
            logger.info(f"特征数量: {len(self.feature_columns)}")
            logger.info(f"标签数量: {len(self.label_columns)}")
            return True
        return False
    
    def load_training_data(self, target_label=None):
        """加载训练数据（带缓存优化）"""
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
        
        # 检查特征数据缓存
        if self._feature_cache is None or self._feature_cache_timestamp != timestamp:
            logger.info(f"加载特征数据: {latest_feature_file}")
            try:
                self._feature_cache = np.load(os.path.join(self.training_data_dir, latest_feature_file))
                self._feature_cache_timestamp = timestamp
                logger.info(f"特征数据已缓存: 样本数={len(self._feature_cache)}, 特征数={self._feature_cache.shape[1]}")
            except Exception as e:
                logger.error(f"加载特征数据失败: {e}")
                return None, None, None
        
        # 如果未指定目标标签，使用默认标签
        if target_label is None:
            target_label = 'return_5d_gt_5pct'  # 默认使用5日5%收益率标签
        
        label_file = f'labels_{target_label}_{timestamp}.npy'
        label_path = os.path.join(self.training_data_dir, label_file)
        
        if not os.path.exists(label_path):
            logger.error(f"标签文件不存在: {label_file}")
            # 列出可用的标签文件
            available_labels = [f for f in os.listdir(self.training_data_dir) if f.startswith('labels_') and f.endswith('.npy')]
            logger.info(f"可用的标签文件: {available_labels}")
            return None, None, None
        
        try:
            y = np.load(label_path)
            logger.info(f"加载标签: {label_file}")
            
            # 简化日志输出
            if target_label.endswith('_continuous'):
                logger.info(f"回归任务 - 标签均值: {y.mean():.4f}")
            else:
                unique_labels = np.unique(y)
                if len(unique_labels) == 2:
                    logger.info(f"二分类任务 - 正样本比例: {y.mean():.2%}")
                else:
                    logger.info(f"多分类任务 - {len(unique_labels)}个类别")
            
            return self._feature_cache.copy(), y, target_label
            
        except Exception as e:
            logger.error(f"加载标签数据失败: {e}")
            return None, None, None
    
    def preprocess_data(self, X, y):
        """数据预处理（带缓存优化）"""
        # 生成数据的哈希值作为缓存键（改进版）
        try:
            # 使用数据的形状、均值和标准差来生成更稳定的哈希
            data_signature = (
                X.shape,
                round(float(np.mean(X)), 6),
                round(float(np.std(X)), 6),
                round(float(np.min(X)), 6),
                round(float(np.max(X)), 6)
            )
            data_hash = hash(data_signature)
        except Exception as e:
            logger.warning(f"生成数据哈希失败: {e}，跳过缓存")
            data_hash = None
        
        # 暂时禁用缓存以确保数据处理正确性
        # TODO: 重新设计缓存机制
        # if (data_hash is not None and 
        #     self._processed_feature_cache is not None and 
        #     self._scaler_cache is not None and 
        #     hasattr(self, '_last_data_hash') and 
        #     self._last_data_hash == data_hash):
        #     logger.info("使用缓存的预处理数据")
        #     self.scaler = self._scaler_cache
        #     return self._processed_feature_cache.copy(), y
        
        logger.info("开始数据预处理...")
        
        # 确保X是numpy数组
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # 检查数据形状
        if X.ndim != 2:
            logger.error(f"特征数据维度错误: {X.ndim}, 期望2维")
            raise ValueError(f"特征数据必须是2维数组，当前维度: {X.ndim}")
        
        # 检查并处理缺失值
        nan_mask = np.isnan(X)
        if nan_mask.any():
            nan_count = nan_mask.sum()
            logger.warning(f"发现 {nan_count} 个缺失值，使用中位数填充")
            # 使用中位数填充，对异常值更鲁棒
            for col in range(X.shape[1]):
                col_mask = nan_mask[:, col]
                if col_mask.any():
                    col_median = np.nanmedian(X[:, col])
                    X[col_mask, col] = col_median
        
        # 检查并处理无穷值
        inf_mask = np.isinf(X)
        if inf_mask.any():
            inf_count = inf_mask.sum()
            logger.warning(f"发现 {inf_count} 个无穷值，进行处理")
            # 分别处理正无穷和负无穷
            for col in range(X.shape[1]):
                col_data = X[:, col]
                finite_data = col_data[np.isfinite(col_data)]
                if len(finite_data) > 0:
                    col_max = np.max(finite_data)
                    col_min = np.min(finite_data)
                    # 用有限值的最大值和最小值替换无穷值
                    X[np.isposinf(col_data), col] = col_max
                    X[np.isneginf(col_data), col] = col_min
                else:
                    # 如果整列都是无穷值，用0填充
                    X[:, col] = 0
        
        # 异常值检测和处理（使用IQR方法）
        logger.info("开始异常值检测和处理...")
        outlier_samples = np.zeros(X.shape[0], dtype=bool)  # 记录包含异常值的样本
        total_outlier_features = 0  # 记录异常特征的总数
        
        for col in range(X.shape[1]):
            col_data = X[:, col]
            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR  # 使用1.5倍IQR作为异常值阈值
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            if outlier_mask.any():
                outlier_samples |= outlier_mask  # 标记包含异常值的样本
                total_outlier_features += outlier_mask.sum()
                # 用边界值替换异常值
                X[col_data < lower_bound, col] = lower_bound
                X[col_data > upper_bound, col] = upper_bound
        
        outlier_sample_count = outlier_samples.sum()
        if outlier_sample_count > 0:
            logger.warning(f"检测到 {outlier_sample_count} 个样本包含异常值（共 {total_outlier_features} 个异常特征值），已进行边界值替换")
            logger.info(f"异常样本比例: {outlier_sample_count/X.shape[0]*100:.2f}%")
        
        # 选择标准化器类型
        if self.scaler_type == 'robust':
            logger.info("使用RobustScaler进行特征标准化（对异常值更鲁棒）...")
            self.scaler = RobustScaler()
        else:
            logger.info("使用StandardScaler进行特征标准化...")
            self.scaler = StandardScaler()
        
        # 记录标准化前的特征统计
        logger.info(f"标准化前特征统计:")
        logger.info(f"  形状: {X.shape}")
        logger.info(f"  最小值: {X.min():.4f}")
        logger.info(f"  最大值: {X.max():.4f}")
        logger.info(f"  均值范围: {X.mean(axis=0).min():.4f} - {X.mean(axis=0).max():.4f}")
        logger.info(f"  标准差范围: {X.std(axis=0).min():.4f} - {X.std(axis=0).max():.4f}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # 记录标准化后的特征统计
        logger.info(f"标准化后特征统计:")
        logger.info(f"  最小值: {X_scaled.min():.4f}")
        logger.info(f"  最大值: {X_scaled.max():.4f}")
        logger.info(f"  均值范围: {X_scaled.mean(axis=0).min():.4f} - {X_scaled.mean(axis=0).max():.4f}")
        logger.info(f"  标准差范围: {X_scaled.std(axis=0).min():.4f} - {X_scaled.std(axis=0).max():.4f}")
        
        # 缓存预处理结果
        self._processed_feature_cache = X_scaled.copy()
        self._scaler_cache = self.scaler
        self._last_data_hash = data_hash
        
        logger.info("数据预处理完成并已缓存")
        return X_scaled, y
    
    def train_single_model(self, X, y, model_type='random_forest', target_label='', test_size=0.2, random_state=42, enable_cv=False, early_stopping=True, is_preprocessed=False):
        """训练单个随机森林模型（优化版）"""
        start_time = time.time()
        
        # 检查模型类型
        if model_type != 'random_forest':
            logger.warning(f"不支持的模型类型: {model_type}，使用随机森林替代")
            model_type = 'random_forest'
        
        # 数据预处理（如果尚未预处理）
        if is_preprocessed:
            X_processed, y_processed = X, y
            logger.info("使用已预处理的数据")
        else:
            X_processed, y_processed = self.preprocess_data(X, y)
        
        # 检查是否为回归任务
        is_regression = target_label.endswith('_continuous')
        
        # 分割训练集和测试集
        if is_regression:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=test_size, random_state=random_state
            )
        else:
            # 分类任务使用分层抽样
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=random_state, stratify=y_processed
                )
            except ValueError:
                # 如果某些类别样本太少，不使用分层抽样
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=random_state
                )
        
        # 创建随机森林模型
        if is_regression:
            model_params = self.model_params['random_forest_regressor'].copy()
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid_key = 'random_forest_regressor'
        else:
            model_params = self.model_params['random_forest_classifier'].copy()
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
            param_grid_key = 'random_forest_classifier'
        
        # 训练模型（使用超参数调优或默认参数）
        if self.enable_hyperparameter_tuning:
            logger.info("开始超参数调优...")
            param_grid = self.param_grids[param_grid_key]
            
            # 使用较少的CV折数以节省时间
            cv_folds = 3 if len(X_train) > 10000 else 2
            
            if is_regression:
                grid_search = GridSearchCV(
                    base_model, param_grid, cv=cv_folds, 
                    scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
                )
            else:
                grid_search = GridSearchCV(
                    base_model, param_grid, cv=cv_folds, 
                    scoring='f1_weighted', n_jobs=-1, verbose=1
                )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logger.info(f"最佳参数: {grid_search.best_params_}")
            logger.info(f"最佳CV分数: {grid_search.best_score_:.4f}")
        else:
            # 使用默认优化参数
            if is_regression:
                model = RandomForestRegressor(**model_params)
            else:
                model = RandomForestClassifier(**model_params)
            model.fit(X_train, y_train)
        
        # 记录训练后的信息
        if hasattr(model, 'oob_score_'):
            logger.debug(f"随机森林袋外分数: {model.oob_score_:.4f}")
            
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算训练时间
        train_time = time.time() - start_time
        
        # 评估模型
        results = {
            'model_type': model_type, 
            'target_label': target_label,
            'train_time': train_time,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # 初始化y_pred_proba变量
        y_pred_proba = None
        
        if is_regression:
            # 回归评估指标
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            from scipy.stats import pearsonr
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # 计算相关性
            correlation, p_value = pearsonr(y_test, y_pred)
            
            # 预测范围分析
            pred_range = y_pred.max() - y_pred.min()
            actual_range = y_test.max() - y_test.min()
            range_ratio = pred_range / actual_range if actual_range > 0 else 0
            
            # 负值比例分析
            negative_pred_ratio = (y_pred < 0).mean()
            negative_actual_ratio = (y_test < 0).mean()
            
            logger.info(f"随机森林: RMSE={rmse:.4f}, R²={r2:.4f}, 相关性={correlation:.4f}, 训练时间={train_time:.2f}s")
            logger.info(f"  预测范围: {pred_range:.4f} (实际: {actual_range:.4f}, 比例: {range_ratio:.2f})")
            logger.info(f"  负值比例: 预测={negative_pred_ratio:.2%}, 实际={negative_actual_ratio:.2%}")
            
            results.update({
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'correlation': correlation,
                'p_value': p_value,
                'pred_range': pred_range,
                'actual_range': actual_range,
                'range_ratio': range_ratio,
                'negative_pred_ratio': negative_pred_ratio,
                'negative_actual_ratio': negative_actual_ratio
            })
        else:
            # 分类评估指标
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # 计算概率预测（用于AUC）
            try:
                y_pred_proba = model.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:  # 二分类
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:  # 多分类
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"计算AUC时出错: {e}")
                auc = None
            
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            logger.info(f"随机森林: 准确率={accuracy:.4f}, F1={f1:.4f}, AUC={auc_str}, 训练时间={train_time:.2f}s")
            
            results.update({
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc
            })
        
        # 特征重要性
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            # 只保存前20个最重要的特征
            top_indices = np.argsort(feature_importance)[-20:][::-1]
            top_importance = feature_importance[top_indices]
            results['feature_importance'] = {
                'indices': top_indices.tolist(),
                'values': top_importance.tolist()
            }
        
        # 交叉验证（可选）
        if enable_cv:
            logger.info("执行5折交叉验证...")
            cv_start = time.time()
            if is_regression:
                cv_scores = cross_val_score(model, X_processed, y_processed, cv=5, scoring='r2', n_jobs=-1)
                cv_metric = 'R²'
            else:
                cv_scores = cross_val_score(model, X_processed, y_processed, cv=5, scoring='accuracy', n_jobs=-1)
                cv_metric = '准确率'
            
            cv_time = time.time() - cv_start
            logger.info(f"交叉验证{cv_metric}: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f}), 时间: {cv_time:.2f}s")
            
            results.update({
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_time': cv_time
            })
        
        # 保存模型和scaler到实例中（用于后续预测）
        self.model = model
        # 确保scaler被正确设置（从preprocess_data方法中获取）
        if hasattr(self, 'scaler') and self.scaler is not None:
            # scaler已在preprocess_data中设置，无需额外操作
            logger.info("模型和标准化器已保存到实例中")
        else:
            logger.warning("标准化器未正确设置")
        
        # 清理内存
        del X_train, X_test, y_train, y_test
        if y_pred_proba is not None:
            del y_pred_proba
        gc.collect()
        
        return model, results
    
    def save_models(self, target_label):
        """保存模型和标准化器"""
        timestamp = datetime.now().strftime('%Y%m%d') #_%H%M%S')
        
        for model_key, model in self.models.items():
            # 保存模型
            model_path = os.path.join(self.model_save_dir, f'{model_key}_{timestamp}.pkl')
            joblib.dump(model, model_path)
            logger.info(f"模型已保存: {model_path}")
            
            # 保存对应的标准化器
            if model_key in self.scalers:
                scaler_path = os.path.join(self.model_save_dir, f'scaler_{model_key}_{timestamp}.pkl')
                joblib.dump(self.scalers[model_key], scaler_path)
                logger.info(f"标准化器已保存: {scaler_path}")
    
    def load_model(self, model_path, scaler_path):
        """加载模型和标准化器"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"模型加载成功: {model_path}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def predict(self, X, model_key=None):
        """预测
        
        Args:
            X: 输入特征数据
            model_key: 指定使用的模型键，如果为None则使用最后训练的模型
        """
        # 选择模型和标准化器
        if model_key is not None:
            if model_key not in self.models:
                logger.error(f"指定的模型不存在: {model_key}")
                return None, None
            model = self.models[model_key]
            scaler = self.scalers.get(model_key)
        else:
            # 使用单一模型模式（向后兼容）
            if hasattr(self, 'model') and self.model is not None:
                model = self.model
                scaler = getattr(self, 'scaler', None)
            elif self.models:
                # 使用最后添加的模型
                model_key = list(self.models.keys())[-1]
                model = self.models[model_key]
                scaler = self.scalers.get(model_key)
                logger.info(f"使用模型: {model_key}")
            else:
                logger.error("没有可用的模型")
                return None, None
        
        if model is None:
            logger.error("模型未训练或加载")
            return None, None
        
        if scaler is None:
            logger.error("标准化器未训练或加载")
            return None, None
        
        # 输入数据验证
        if X is None or len(X) == 0:
            logger.error("输入数据为空")
            return None, None
        
        # 转换为numpy数组
        X = np.array(X)
        
        # 检查维度
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # 处理缺失值和无穷值
        if np.any(np.isnan(X)):
            logger.warning("输入数据包含NaN值，使用中位数填充")
            X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        
        if np.any(np.isinf(X)):
            logger.warning("输入数据包含无穷值，进行处理")
            finite_mask = np.isfinite(X)
            if np.any(finite_mask):
                X = np.where(np.isinf(X), 
                           np.where(X > 0, np.nanmax(X[finite_mask]), np.nanmin(X[finite_mask])), 
                           X)
        
        # 标准化
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            logger.error(f"数据标准化失败: {e}")
            return None, None
        
        # 预测
        try:
            predictions = model.predict(X_scaled)
            
            # 如果是分类任务，尝试获取概率预测
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X_scaled)
                except:
                    logger.warning("无法获取概率预测")
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None, None

    def train_models(self, target_label, model_types=['random_forest'], test_size=0.2, enable_cv=False):
        """训练多个随机森林模型（实际上只训练一个）"""
        # 强制只使用随机森林
        model_types = ['random_forest']
        
        logger.info(f"开始训练随机森林模型，目标标签: {target_label}")
        
        # 加载数据
        X, y, actual_label = self.load_training_data(target_label)
        if X is None:
            return None
        
        # 预处理数据
        X_processed, y_processed = self.preprocess_data(X, y)
        
        # 训练随机森林模型
        model, results = self.train_single_model(
            X_processed, y_processed, 
            model_type='random_forest',
            target_label=actual_label,
            test_size=test_size,
            enable_cv=enable_cv,
            is_preprocessed=True
        )
        
        # 存储模型和标准化器
        model_key = f"random_forest_{actual_label}"
        self.models[model_key] = model
        self.scalers[model_key] = self.scaler
        
        logger.info(f"随机森林模型训练完成: {model_key}")
        
        return {model_key: results}


def batch_train_all_labels(training_data_dir="training_data", model_save_dir="models", 
                          model_types=['random_forest'], enable_cv=False, test_size=0.2, 
                          enable_hyperparameter_tuning=False):
    """批量训练所有标签的随机森林模型"""
    # 强制只使用随机森林
    model_types = ['random_forest']
    
    trainer = OptimizedMLModelTrainer(training_data_dir, model_save_dir, 
                                      enable_hyperparameter_tuning=enable_hyperparameter_tuning)
    
    # 备份现有模型
    trainer.backup_existing_models()
    
    # 获取所有可用的标签
    if not os.path.exists(training_data_dir):
        logger.error(f"训练数据目录不存在: {training_data_dir}")
        return
    
    label_files = [f for f in os.listdir(training_data_dir) if f.startswith('labels_') and f.endswith('.npy')]
    
    if not label_files:
        logger.error("未找到标签文件")
        return
    
    # 提取标签名称
    labels = []
    for file in label_files:
        # 文件格式: labels_{label_name}_{timestamp}.npy
        parts = file.replace('labels_', '').replace('.npy', '').split('_')
        if len(parts) >= 2:
            label_name = '_'.join(parts[:-1])  # 去掉时间戳
            if label_name not in labels:
                labels.append(label_name)
    
    logger.info(f"发现 {len(labels)} 个标签: {labels}")
    
    all_results = {}
    
    for label in labels:
        logger.info(f"\n{'='*50}")
        logger.info(f"训练标签: {label}")
        logger.info(f"{'='*50}")
        
        try:
            results = trainer.train_models(
                target_label=label,
                model_types=model_types,
                test_size=test_size,
                enable_cv=enable_cv
            )
            
            if results:
                all_results[label] = results
                
                # 保存模型
                trainer.save_models(label)
                
                logger.info(f"标签 {label} 的随机森林模型已保存")
            else:
                logger.warning(f"标签 {label} 训练失败")
                
        except Exception as e:
            logger.error(f"训练标签 {label} 时出错: {e}")
            continue
        finally:
            # 内存清理
            gc.collect()
            logger.debug(f"完成标签 {label} 的内存清理")
    
    # 保存训练结果摘要
    summary_file = os.path.join(model_save_dir, f'training_summary_{datetime.now().strftime("%Y%m%d")}.yaml')
    with open(summary_file, 'w', encoding='utf-8') as f:
        yaml.dump(all_results, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"批量训练完成！")
    logger.info(f"成功训练 {len(all_results)} 个标签的随机森林模型")
    logger.info(f"训练结果摘要已保存到: {summary_file}")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    # 默认只使用随机森林进行训练
    # 可以通过修改enable_hyperparameter_tuning=True来启用超参数调优（会显著增加训练时间）
    batch_train_all_labels(
        training_data_dir="training_data",
        model_save_dir="models",
        model_types=['random_forest'],  # 只使用随机森林
        enable_cv=True,  # 启用交叉验证
        test_size=0.2,
        enable_hyperparameter_tuning=False  # 设为True启用超参数调优
    )