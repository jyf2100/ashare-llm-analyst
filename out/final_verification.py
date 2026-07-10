#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终验证：展示03-custom_stock_selection.py使用模型目录进行预测的完整流程
"""

import sys
import os
sys.path.append('/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out')
import numpy as np
from datetime import datetime

def final_verification():
    """最终验证功能"""
    try:
        print("🎯 最终验证：03-custom_stock_selection.py 模型目录配置")
        print("="*70)
        
        # 导入模块
        from importlib import import_module
        custom_selection = import_module('03-custom_stock_selection')
        MLStockPredictor = custom_selection.MLStockPredictor
        
        print("✅ 1. 模块导入成功")
        
        # 创建预测器实例
        predictor = MLStockPredictor()
        
        print("✅ 2. 预测器实例创建成功")
        print(f"   模型目录: {predictor.models_dir}")
        print(f"   已加载模型: {len(predictor.loaded_models)} 个")
        print(f"   已加载标准化器: {len(predictor.loaded_scalers)} 个")
        
        # 验证模型目录配置
        expected_path = "/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/models"
        if predictor.models_dir == expected_path:
            print("✅ 3. 模型目录配置正确")
        else:
            print(f"❌ 3. 模型目录配置错误")
            print(f"   期望: {expected_path}")
            print(f"   实际: {predictor.models_dir}")
            return False
        
        # 验证模型加载
        if len(predictor.loaded_models) >= 15:
            print("✅ 4. 模型加载数量正确 (≥15个)")
        else:
            print(f"❌ 4. 模型加载数量不足: {len(predictor.loaded_models)}")
            return False
        
        # 验证模型选择功能
        print("\n🔍 5. 验证模型选择功能:")
        
        # 测试5日回归模型
        model_5d_reg, scaler_5d_reg, name_5d_reg = predictor._select_best_model(
            prediction_days=5, task_type="regression"
        )
        if model_5d_reg:
            print(f"   ✅ 5日回归模型: {name_5d_reg}")
            print(f"      类型: {type(model_5d_reg).__name__}")
            print(f"      特征数: {getattr(model_5d_reg, 'n_features_in_', '未知')}")
        else:
            print(f"   ❌ 5日回归模型选择失败")
            return False
        
        # 测试5日分类模型
        model_5d_cls, scaler_5d_cls, name_5d_cls = predictor._select_best_model(
            prediction_days=5, task_type="classification"
        )
        if model_5d_cls:
            print(f"   ✅ 5日分类模型: {name_5d_cls}")
            print(f"      类型: {type(model_5d_cls).__name__}")
            print(f"      特征数: {getattr(model_5d_cls, 'n_features_in_', '未知')}")
        else:
            print(f"   ❌ 5日分类模型选择失败")
            return False
        
        # 验证预测功能
        print("\n🧪 6. 验证预测功能:")
        
        # 创建示例特征数据
        if hasattr(model_5d_reg, 'n_features_in_'):
            n_features = model_5d_reg.n_features_in_
            print(f"   创建 {n_features} 维特征数据...")
            
            # 生成随机特征
            np.random.seed(42)
            sample_features = np.random.randn(1, n_features)
            
            # 回归预测测试
            try:
                if scaler_5d_reg:
                    features_scaled = scaler_5d_reg.transform(sample_features)
                else:
                    features_scaled = sample_features
                
                prediction_reg = model_5d_reg.predict(features_scaled)
                print(f"   ✅ 回归预测成功: {prediction_reg[0]:.4f}")
            except Exception as e:
                print(f"   ❌ 回归预测失败: {str(e)}")
                return False
            
            # 分类预测测试
            try:
                if scaler_5d_cls:
                    features_scaled = scaler_5d_cls.transform(sample_features)
                else:
                    features_scaled = sample_features
                
                prediction_cls = model_5d_cls.predict(features_scaled)
                proba_cls = model_5d_cls.predict_proba(features_scaled)
                print(f"   ✅ 分类预测成功: {prediction_cls[0]} (概率: {proba_cls[0].max():.4f})")
            except Exception as e:
                print(f"   ❌ 分类预测失败: {str(e)}")
                return False
        
        # 显示可用的预测方法
        print("\n📋 7. 可用的预测方法:")
        methods = [
            "predict_rps_stocks(rps5_threshold=85, prediction_days=5)",
            "run_complete_strategy_selection(rps5_threshold=85, prediction_days=5)",
            "run_rps_only_prediction(rps5_threshold=85, prediction_days=5)"
        ]
        
        for i, method in enumerate(methods, 1):
            print(f"   {i}. {method}")
        
        # 显示使用示例
        print("\n💡 8. 使用示例代码:")
        print("```python")
        print("# 导入模块")
        print("from importlib import import_module")
        print("custom_selection = import_module('03-custom_stock_selection')")
        print("MLStockPredictor = custom_selection.MLStockPredictor")
        print("")
        print("# 创建预测器（自动使用正确的模型目录）")
        print("predictor = MLStockPredictor()")
        print("")
        print("# 进行RPS股票预测")
        print("results = predictor.predict_rps_stocks(")
        print("    rps5_threshold=85,")
        print("    prediction_days=5,")
        print("    max_stocks=100")
        print(")")
        print("")
        print("# 保存结果")
        print("predictor.save_results(results, 'my_predictions.json')")
        print("```")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"🔬 最终验证 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = final_verification()
    
    print("\n" + "="*70)
    if success:
        print("🎉 验证成功！")
        print("✅ 03-custom_stock_selection.py 已成功配置使用模型目录")
        print("✅ 所有15个训练好的模型已正确加载")
        print("✅ 模型选择和预测功能正常工作")
        print("")
        print("🚀 现在可以使用 03-custom_stock_selection.py 进行股票预测了！")
    else:
        print("❌ 验证失败，请检查错误信息")
        sys.exit(1)