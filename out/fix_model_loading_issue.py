#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复模型加载问题的解决方案
问题：原始模型使用joblib保存，但测试代码使用pickle加载
解决：提供正确的加载方法和转换工具
"""

import os
import pickle
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yaml

def diagnose_model_loading_issue():
    """
    诊断模型加载问题的根本原因
    """
    print("🔍 诊断模型加载问题")
    print("=" * 50)
    
    models_dir = '/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/models'
    
    if not os.path.exists(models_dir):
        print(f"❌ 模型目录不存在: {models_dir}")
        return
    
    # 获取所有pkl文件
    pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    print(f"📁 找到 {len(pkl_files)} 个 .pkl 文件")
    
    # 测试不同的加载方法
    test_files = pkl_files[:3]  # 只测试前3个文件
    
    for pkl_file in test_files:
        file_path = os.path.join(models_dir, pkl_file)
        print(f"\n📄 测试文件: {pkl_file}")
        
        # 方法1: 使用pickle加载
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            print("  ✓ pickle.load() - 成功")
        except Exception as e:
            print(f"  ✗ pickle.load() - 失败: {str(e)}")
        
        # 方法2: 使用joblib加载
        try:
            model = joblib.load(file_path)
            print("  ✓ joblib.load() - 成功")
            print(f"    模型类型: {type(model).__name__}")
            if hasattr(model, 'n_estimators'):
                print(f"    树的数量: {model.n_estimators}")
        except Exception as e:
            print(f"  ✗ joblib.load() - 失败: {str(e)}")

def fix_model_loading_in_existing_scripts():
    """
    修复现有脚本中的模型加载问题
    """
    print("\n🔧 修复现有脚本中的模型加载问题")
    print("=" * 50)
    
    # 需要修复的脚本列表
    scripts_to_fix = [
        '/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/model_compatibility_test.py',
        '/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/practical_model_usage.py'
    ]
    
    for script_path in scripts_to_fix:
        if os.path.exists(script_path):
            print(f"📝 需要修复: {os.path.basename(script_path)}")
            print(f"   将 'pickle.load()' 替换为 'joblib.load()'")
        else:
            print(f"❌ 文件不存在: {script_path}")

def create_correct_model_loader():
    """
    创建正确的模型加载器示例
    """
    print("\n📦 创建正确的模型加载器")
    print("=" * 50)
    
    loader_code = '''
# 正确的模型加载方法
import joblib
import os

def load_model_correctly(model_path):
    """正确加载joblib保存的模型"""
    try:
        model = joblib.load(model_path)
        print(f"✓ 成功加载模型: {os.path.basename(model_path)}")
        return model
    except Exception as e:
        print(f"✗ 加载模型失败: {str(e)}")
        return None

def load_scaler_correctly(scaler_path):
    """正确加载joblib保存的标准化器"""
    try:
        scaler = joblib.load(scaler_path)
        print(f"✓ 成功加载标准化器: {os.path.basename(scaler_path)}")
        return scaler
    except Exception as e:
        print(f"✗ 加载标准化器失败: {str(e)}")
        return None

# 使用示例
models_dir = '/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/models'
model_file = 'random_forest_multi_class_5d_20250818.pkl'
scaler_file = 'scaler_random_forest_multi_class_5d_20250818.pkl'

model = load_model_correctly(os.path.join(models_dir, model_file))
scaler = load_scaler_correctly(os.path.join(models_dir, scaler_file))
'''
    
    print("正确的加载方法:")
    print(loader_code)

def test_correct_loading():
    """
    测试正确的加载方法
    """
    print("\n🧪 测试正确的加载方法")
    print("=" * 50)
    
    models_dir = '/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/models'
    
    # 查找一个模型文件进行测试
    pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and 'random_forest' in f and 'scaler' not in f]
    
    if not pkl_files:
        print("❌ 没有找到可测试的模型文件")
        return
    
    test_file = pkl_files[0]
    test_path = os.path.join(models_dir, test_file)
    
    print(f"📄 测试文件: {test_file}")
    
    try:
        # 使用正确的方法加载
        model = joblib.load(test_path)
        print("✓ 使用 joblib.load() 成功加载模型")
        print(f"  模型类型: {type(model).__name__}")
        
        if hasattr(model, 'n_estimators'):
            print(f"  树的数量: {model.n_estimators}")
        if hasattr(model, 'n_features_in_'):
            print(f"  特征数量: {model.n_features_in_}")
        if hasattr(model, 'classes_'):
            print(f"  类别数量: {len(model.classes_)}")
            
        # 查找对应的标准化器
        scaler_file = test_file.replace('random_forest_', 'scaler_random_forest_')
        scaler_path = os.path.join(models_dir, scaler_file)
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✓ 成功加载对应的标准化器")
            print(f"  标准化器类型: {type(scaler).__name__}")
            if hasattr(scaler, 'n_features_in_'):
                print(f"  期望特征数量: {scaler.n_features_in_}")
        else:
            print(f"❌ 未找到对应的标准化器: {scaler_file}")
            
    except Exception as e:
        print(f"❌ 加载失败: {str(e)}")

def provide_solution_summary():
    """
    提供解决方案总结
    """
    print("\n💡 解决方案总结")
    print("=" * 50)
    
    solutions = [
        "1. 问题根源：原始模型使用 joblib.dump() 保存，但测试代码使用 pickle.load() 加载",
        "2. 解决方法：将所有 pickle.load() 替换为 joblib.load()",
        "3. 修复步骤：",
        "   a) 在脚本开头导入 joblib",
        "   b) 将 'with open(file, 'rb') as f: pickle.load(f)' 替换为 'joblib.load(file)'",
        "   c) 确保使用相同的加载方法加载模型和标准化器",
        "4. 验证方法：运行修复后的脚本，确认模型可以正常加载和使用"
    ]
    
    for solution in solutions:
        print(solution)
    
    print("\n📋 需要修复的文件:")
    files_to_fix = [
        "model_compatibility_test.py",
        "practical_model_usage.py",
        "quick_model_demo.py",
        "其他使用 pickle.load() 加载 .pkl 模型文件的脚本"
    ]
    
    for file_name in files_to_fix:
        print(f"  - {file_name}")

if __name__ == "__main__":
    print("🚀 模型加载问题诊断和修复工具")
    print("=" * 60)
    
    # 1. 诊断问题
    diagnose_model_loading_issue()
    
    # 2. 测试正确的加载方法
    test_correct_loading()
    
    # 3. 提供修复建议
    fix_model_loading_in_existing_scripts()
    
    # 4. 创建正确的加载器示例
    create_correct_model_loader()
    
    # 5. 提供解决方案总结
    provide_solution_summary()
    
    print("\n✅ 诊断完成！请按照上述建议修复相关脚本。")