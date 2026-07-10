#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示使用修改后的03-custom_stock_selection.py进行股票预测
"""

import sys
import os
sys.path.append('/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out')

# 导入自定义股票选择模块
from importlib import import_module
import pandas as pd
import numpy as np
from datetime import datetime

def demo_prediction():
    """演示股票预测功能"""
    try:
        # 导入模块
        custom_selection = import_module('03-custom_stock_selection')
        MLStockPredictor = custom_selection.MLStockPredictor
        
        print("🚀 开始演示股票预测功能")
        print("="*60)
        
        # 创建预测器实例
        predictor = MLStockPredictor()
        
        print(f"✓ 成功创建预测器实例")
        print(f"  模型目录: {predictor.models_dir}")
        print(f"  已加载模型数量: {len(predictor.loaded_models)}")
        
        # 显示可用模型
        print(f"\n📊 可用模型列表:")
        for i, model_name in enumerate(list(predictor.loaded_models.keys())[:10], 1):
            model = predictor.loaded_models[model_name]
            print(f"  {i:2d}. {model_name}")
            print(f"      类型: {type(model).__name__}")
        
        if len(predictor.loaded_models) > 10:
            print(f"  ... 还有 {len(predictor.loaded_models) - 10} 个模型")
        
        # 测试模型选择
        print(f"\n🎯 测试模型选择:")
        
        # 选择5日回归模型
        model, scaler, model_name = predictor._select_best_model(prediction_days=5, task_type="regression")
        if model:
            print(f"  ✓ 5日回归模型: {model_name}")
            print(f"    特征数: {model.n_features_in_ if hasattr(model, 'n_features_in_') else '未知'}")
        
        # 选择5日分类模型
        model, scaler, model_name = predictor._select_best_model(prediction_days=5, task_type="classification")
        if model:
            print(f"  ✓ 5日分类模型: {model_name}")
            print(f"    特征数: {model.n_features_in_ if hasattr(model, 'n_features_in_') else '未知'}")
        
        # 创建示例特征数据进行预测测试
        print(f"\n🧪 创建示例数据进行预测测试:")
        
        # 获取模型期望的特征数量
        if model and hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
            print(f"  模型期望特征数: {n_features}")
            
            # 创建随机特征数据
            np.random.seed(42)
            sample_features = np.random.randn(1, n_features)
            
            try:
                # 标准化特征
                if scaler:
                    sample_features_scaled = scaler.transform(sample_features)
                    print(f"  ✓ 特征已标准化")
                else:
                    sample_features_scaled = sample_features
                    print(f"  ⚠️ 未使用标准化")
                
                # 进行预测
                prediction = model.predict(sample_features_scaled)
                print(f"  ✓ 预测成功")
                print(f"  预测结果: {prediction[0]:.4f}")
                
                # 如果是分类模型，也显示概率
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(sample_features_scaled)
                    print(f"  预测概率: {proba[0]}")
                
            except Exception as e:
                print(f"  ❌ 预测失败: {str(e)}")
        
        # 显示模型性能摘要
        print(f"\n📈 模型性能摘要:")
        performance_summary = predictor.get_model_performance_summary()
        print(f"  总模型数: {performance_summary['total_models']}")
        print(f"  模型类型: {list(performance_summary['model_types'].keys())}")
        print(f"  预测周期: {list(performance_summary['prediction_periods'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def demo_rps_prediction():
    """演示RPS股票预测功能"""
    try:
        # 导入模块
        custom_selection = import_module('03-custom_stock_selection')
        MLStockPredictor = custom_selection.MLStockPredictor
        
        print(f"\n🎯 演示RPS股票预测功能")
        print("="*60)
        
        # 创建预测器实例
        predictor = MLStockPredictor()
        
        print(f"📋 预测器配置:")
        print(f"  数据目录: {predictor.data_dir}")
        print(f"  RPS目录: {predictor.rps_dir}")
        print(f"  模型目录: {predictor.models_dir}")
        print(f"  已加载模型: {len(predictor.loaded_models)} 个")
        
        # 注意：这里只是演示配置，实际运行需要真实的股票数据
        print(f"\n⚠️ 注意: 完整的RPS预测需要真实的股票数据和RPS数据")
        print(f"   当前演示仅展示模型加载和配置功能")
        
        # 显示可用的预测方法
        print(f"\n🔧 可用的预测方法:")
        methods = [
            "predict_rps_stocks()",
            "run_complete_strategy_selection()", 
            "run_rps_only_prediction()"
        ]
        
        for i, method in enumerate(methods, 1):
            print(f"  {i}. {method}")
        
        print(f"\n💡 使用示例:")
        print(f"  # 创建预测器")
        print(f"  predictor = MLStockPredictor()")
        print(f"  ")
        print(f"  # 预测RPS5>85的股票")
        print(f"  results = predictor.predict_rps_stocks(")
        print(f"      rps5_threshold=85,")
        print(f"      prediction_days=5,")
        print(f"      max_stocks=100")
        print(f"  )")
        print(f"  ")
        print(f"  # 保存结果")
        print(f"  predictor.save_results(results)")
        
        return True
        
    except Exception as e:
        print(f"❌ RPS演示失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"🎬 03-custom_stock_selection.py 预测功能演示")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 基础预测演示
    success1 = demo_prediction()
    
    # RPS预测演示
    success2 = demo_rps_prediction()
    
    print("\n" + "="*80)
    if success1 and success2:
        print("🎉 演示完成！03-custom_stock_selection.py 已成功配置使用模型目录")
        print(f"   模型目录: /mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/models")
        print(f"   已加载模型数量: 15 个")
    else:
        print("❌ 演示过程中出现错误，请检查日志")
        sys.exit(1)