#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型兼容性测试和修复

测试现有模型的兼容性，如果无法加载则重新训练简单模型用于演示
"""

import os
import sys
import pickle
import joblib
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def test_model_loading(models_dir='/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/models'):
    """测试模型加载"""
    print("测试模型文件兼容性...")
    print("=" * 50)
    
    # 获取所有pkl文件
    pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and 'random_forest' in f]
    
    working_models = []
    broken_models = []
    
    for pkl_file in pkl_files[:5]:  # 只测试前5个
        file_path = os.path.join(models_dir, pkl_file)
        try:
            model = joblib.load(file_path)
            print(f"✓ {pkl_file} - 加载成功")
            working_models.append(pkl_file)
        except Exception as e:
            print(f"✗ {pkl_file} - 加载失败: {str(e)}")
            broken_models.append(pkl_file)
    
    print(f"\n总结：{len(working_models)} 个模型可用，{len(broken_models)} 个模型损坏")
    return working_models, broken_models

def create_demo_models(output_dir='/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/demo_models'):
    """创建演示用的简单模型"""
    print("\n创建演示模型...")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成模拟训练数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 29
    
    # 特征名称（与实际训练时保持一致）
    feature_columns = [
        'close', 'open', 'high', 'low', 'volume',
        'rsi_14', 'ma_5', 'ma_10', 'ma_20', 'ma_60',
        'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
        'atr_14', 'adx_14', 'cci_20', 'williams_r', 'stoch_k',
        'stoch_d', 'roc_10', 'momentum_10', 'trix'
    ]
    
    # 生成特征数据
    X = np.random.randn(n_samples, n_features)
    
    # 创建不同类型的模型
    models_to_create = [
        {
            'name': 'multi_class_5d',
            'type': 'classification',
            'n_classes': 3,
            'description': '5日多分类走势预测'
        },
        {
            'name': 'return_5d_gt_8pct',
            'type': 'binary_classification', 
            'n_classes': 2,
            'description': '5日收益>8%概率预测'
        },
        {
            'name': 'return_5d_continuous',
            'type': 'regression',
            'n_classes': None,
            'description': '5日连续收益率预测'
        }
    ]
    
    created_models = []
    
    for model_config in models_to_create:
        try:
            model_name = model_config['name']
            model_type = model_config['type']
            
            print(f"创建模型：{model_config['description']}")
            
            # 生成目标变量
            if model_type == 'classification':
                y = np.random.randint(0, model_config['n_classes'], n_samples)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'binary_classification':
                y = np.random.randint(0, 2, n_samples)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # regression
                y = np.random.randn(n_samples) * 0.1  # 模拟收益率
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 创建标准化器
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 训练模型
            model.fit(X_train_scaled, y_train)
            
            # 评估模型
            if model_type in ['classification', 'binary_classification']:
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"  准确率: {accuracy:.4f}")
                
                # 保存模型配置
                config = {
                    'model_type': 'random_forest',
                    'target_label': model_name,
                    'feature_columns': feature_columns,
                    'test_accuracy': float(accuracy),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'created_at': datetime.now().isoformat()
                }
            else:
                y_pred = model.predict(X_test_scaled)
                mse = np.mean((y_test - y_pred) ** 2)
                print(f"  MSE: {mse:.6f}")
                
                # 保存模型配置
                config = {
                    'model_type': 'random_forest',
                    'target_label': model_name,
                    'feature_columns': feature_columns,
                    'test_mse': float(mse),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'created_at': datetime.now().isoformat()
                }
            
            # 保存文件
            date_suffix = datetime.now().strftime('%Y%m%d')
            
            # 保存模型
            model_file = f'{output_dir}/demo_random_forest_{model_name}_{date_suffix}.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # 保存标准化器
            scaler_file = f'{output_dir}/demo_scaler_random_forest_{model_name}_{date_suffix}.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            # 保存配置
            config_file = f'{output_dir}/demo_model_config_{model_name}_{date_suffix}.yaml'
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            created_models.append(model_name)
            print(f"  ✓ 模型已保存")
            
        except Exception as e:
            print(f"  ✗ 创建失败: {str(e)}")
    
    print(f"\n成功创建 {len(created_models)} 个演示模型")
    return created_models

def test_demo_models(demo_dir='/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/demo_models'):
    """测试演示模型"""
    print("\n测试演示模型...")
    print("=" * 50)
    
    date_suffix = datetime.now().strftime('%Y%m%d')
    
    # 测试加载和预测
    test_models = ['multi_class_5d', 'return_5d_gt_8pct', 'return_5d_continuous']
    
    for model_name in test_models:
        try:
            # 加载文件
            model_file = f'{demo_dir}/demo_random_forest_{model_name}_{date_suffix}.pkl'
            scaler_file = f'{demo_dir}/demo_scaler_random_forest_{model_name}_{date_suffix}.pkl'
            config_file = f'{demo_dir}/demo_model_config_{model_name}_{date_suffix}.yaml'
            
            # 加载模型
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # 加载标准化器
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            # 加载配置
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 测试预测
            test_features = np.random.randn(1, 29)
            test_features_scaled = scaler.transform(test_features)
            prediction = model.predict(test_features_scaled)[0]
            
            print(f"✓ {model_name}: 预测值 = {prediction}")
            
            # 如果是分类模型，显示概率
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(test_features_scaled)[0]
                print(f"  概率分布: {probabilities}")
            
        except Exception as e:
            print(f"✗ {model_name}: 测试失败 - {str(e)}")

def main():
    """主函数"""
    print("模型兼容性测试和修复工具")
    print("=" * 60)
    
    # 测试现有模型
    working_models, broken_models = test_model_loading()
    
    if len(working_models) == 0:
        print("\n所有现有模型都无法加载，创建演示模型...")
        created_models = create_demo_models()
        
        if created_models:
            test_demo_models()
            print("\n✓ 演示模型创建成功！")
            print("\n使用说明：")
            print("1. 演示模型保存在 demo_models/ 目录")
            print("2. 可以修改 practical_model_usage.py 使用演示模型")
            print("3. 演示模型使用相同的特征结构，可以直接替换")
        else:
            print("\n✗ 演示模型创建失败")
    else:
        print(f"\n✓ 发现 {len(working_models)} 个可用模型，无需创建演示模型")
        for model in working_models:
            print(f"  - {model}")

if __name__ == "__main__":
    main()