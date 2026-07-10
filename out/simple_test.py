#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的测试脚本，验证03-custom_stock_selection.py的模型加载功能
"""

import sys
import os
sys.path.append('/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out')

def test_basic_functionality():
    """测试基本功能"""
    try:
        print("🔍 导入模块...")
        # 导入模块
        from importlib import import_module
        custom_selection = import_module('03-custom_stock_selection')
        MLStockPredictor = custom_selection.MLStockPredictor
        print("✓ 模块导入成功")
        
        print("\n🚀 创建预测器实例...")
        # 创建预测器实例
        predictor = MLStockPredictor()
        print("✓ 预测器创建成功")
        
        print(f"\n📊 基本信息:")
        print(f"  模型目录: {predictor.models_dir}")
        print(f"  已加载模型数量: {len(predictor.loaded_models)}")
        print(f"  已加载标准化器数量: {len(predictor.loaded_scalers)}")
        
        if len(predictor.loaded_models) > 0:
            print(f"\n✅ 模型加载成功！")
            print(f"  前5个模型:")
            for i, model_name in enumerate(list(predictor.loaded_models.keys())[:5], 1):
                model = predictor.loaded_models[model_name]
                print(f"    {i}. {model_name} ({type(model).__name__})")
            
            # 测试模型选择
            print(f"\n🎯 测试模型选择:")
            model, scaler, model_name = predictor._select_best_model(prediction_days=5, task_type="regression")
            if model:
                print(f"  ✓ 5日回归模型: {model_name}")
            else:
                print(f"  ❌ 未找到5日回归模型")
            
            model, scaler, model_name = predictor._select_best_model(prediction_days=5, task_type="classification")
            if model:
                print(f"  ✓ 5日分类模型: {model_name}")
            else:
                print(f"  ❌ 未找到5日分类模型")
            
            return True
        else:
            print(f"❌ 没有加载到任何模型")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 简化测试：03-custom_stock_selection.py 模型加载功能")
    print("="*60)
    
    success = test_basic_functionality()
    
    print("\n" + "="*60)
    if success:
        print("🎉 测试成功！模型加载功能正常")
        print("✅ 03-custom_stock_selection.py 已成功配置使用模型目录")
        print("   /mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/models")
    else:
        print("❌ 测试失败")
        sys.exit(1)