#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型组合预测器

该脚本展示如何组合使用不同类型的模型进行股票预测：
1. 多分类模型：预测股票未来走势类别
2. 二分类模型：预测是否达到特定收益阈值
3. 回归模型：预测具体的收益率数值

通过组合多个模型的预测结果，可以获得更全面和可靠的预测信息。
"""

import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out')

class ModelEnsemblePredictor:
    """模型组合预测器"""
    
    def __init__(self, models_dir='/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/models'):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.configs = {}
        self.feature_columns = None
        
    def load_models(self, date_suffix='20250818'):
        """加载所有可用的模型"""
        print(f"正在加载模型（日期后缀：{date_suffix}）...")
        
        # 定义模型类型和对应的预测任务
        model_types = {
            # 多分类模型
            'multi_class_5d': '5日多分类走势',
            'multi_class_10d': '10日多分类走势',
            'multi_class_20d': '20日多分类走势',
            
            # 二分类模型（不同收益阈值）
            'return_5d_gt_3pct': '5日收益>3%',
            'return_5d_gt_5pct': '5日收益>5%',
            'return_5d_gt_8pct': '5日收益>8%',
            'return_10d_gt_5pct': '10日收益>5%',
            'return_10d_gt_8pct': '10日收益>8%',
            'return_20d_gt_5pct': '20日收益>5%',
            'return_20d_gt_8pct': '20日收益>8%',
            
            # 回归模型
            'return_5d_continuous': '5日连续收益率',
            'return_10d_continuous': '10日连续收益率',
            'return_20d_continuous': '20日连续收益率'
        }
        
        loaded_count = 0
        for model_key, description in model_types.items():
            try:
                # 加载模型文件
                model_file = f'{self.models_dir}/random_forest_{model_key}_{date_suffix}.pkl'
                config_file = f'{self.models_dir}/model_config_{model_key}_{date_suffix}.yaml'
                scaler_file = f'{self.models_dir}/scaler_random_forest_{model_key}_{date_suffix}.pkl'
                
                if os.path.exists(model_file) and os.path.exists(config_file):
                    # 加载模型
                    with open(model_file, 'rb') as f:
                        self.models[model_key] = pickle.load(f)
                    
                    # 加载配置
                    with open(config_file, 'r', encoding='utf-8') as f:
                        self.configs[model_key] = yaml.safe_load(f)
                    
                    # 加载标准化器（如果存在）
                    if os.path.exists(scaler_file):
                        with open(scaler_file, 'rb') as f:
                            self.scalers[model_key] = pickle.load(f)
                    
                    # 获取特征列（所有模型应该使用相同的特征）
                    if self.feature_columns is None:
                        self.feature_columns = self.configs[model_key]['feature_columns']
                    
                    print(f"✓ 已加载：{description} ({model_key})")
                    loaded_count += 1
                    
            except Exception as e:
                print(f"✗ 加载失败：{description} ({model_key}) - {str(e)}")
        
        print(f"\n总共加载了 {loaded_count} 个模型")
        print(f"特征维度：{len(self.feature_columns)} 个特征")
        
        return loaded_count > 0
    
    def prepare_features(self, stock_data):
        """准备特征数据"""
        # 这里需要根据实际的特征提取逻辑来实现
        # 确保特征顺序和训练时一致
        if isinstance(stock_data, dict):
            # 如果输入是字典格式的特征
            features = []
            for col in self.feature_columns:
                if col in stock_data:
                    features.append(stock_data[col])
                else:
                    features.append(0.0)  # 缺失特征用0填充
            return np.array(features).reshape(1, -1)
        
        elif isinstance(stock_data, (list, np.ndarray)):
            # 如果输入是数组格式
            features = np.array(stock_data).reshape(1, -1)
            if features.shape[1] != len(self.feature_columns):
                raise ValueError(f"特征维度不匹配：期望{len(self.feature_columns)}，实际{features.shape[1]}")
            return features
        
        else:
            raise ValueError("不支持的数据格式")
    
    def predict_single_model(self, model_key, features):
        """使用单个模型进行预测"""
        if model_key not in self.models:
            return None
        
        try:
            # 标准化特征（如果有标准化器）
            if model_key in self.scalers:
                features_scaled = self.scalers[model_key].transform(features)
            else:
                features_scaled = features
            
            # 进行预测
            model = self.models[model_key]
            prediction = model.predict(features_scaled)[0]
            
            # 获取预测概率（如果是分类模型）
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                return {
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'confidence': np.max(probabilities)
                }
            else:
                # 回归模型
                return {
                    'prediction': prediction,
                    'probabilities': None,
                    'confidence': None
                }
                
        except Exception as e:
            print(f"模型 {model_key} 预测失败：{str(e)}")
            return None
    
    def predict_ensemble(self, stock_data, prediction_horizon='5d'):
        """组合预测：根据预测时间周期组合相关模型"""
        features = self.prepare_features(stock_data)
        results = {}
        
        print(f"\n=== {prediction_horizon} 预测结果 ===")
        
        # 根据预测周期选择相关模型
        if prediction_horizon == '5d':
            model_keys = [
                'multi_class_5d',      # 多分类走势
                'return_5d_gt_3pct',   # 收益>3%概率
                'return_5d_gt_5pct',   # 收益>5%概率
                'return_5d_gt_8pct',   # 收益>8%概率
                'return_5d_continuous' # 具体收益率
            ]
        elif prediction_horizon == '10d':
            model_keys = [
                'multi_class_10d',
                'return_10d_gt_5pct',
                'return_10d_gt_8pct',
                'return_10d_continuous'
            ]
        elif prediction_horizon == '20d':
            model_keys = [
                'multi_class_20d',
                'return_20d_gt_5pct',
                'return_20d_gt_8pct',
                'return_20d_continuous'
            ]
        else:
            raise ValueError(f"不支持的预测周期：{prediction_horizon}")
        
        # 执行预测
        for model_key in model_keys:
            if model_key in self.models:
                result = self.predict_single_model(model_key, features)
                if result is not None:
                    results[model_key] = result
        
        return results
    
    def interpret_predictions(self, predictions, prediction_horizon='5d'):
        """解释预测结果"""
        print(f"\n=== {prediction_horizon} 预测解释 ===")
        
        # 多分类走势解释
        multi_class_key = f'multi_class_{prediction_horizon}'
        if multi_class_key in predictions:
            pred = predictions[multi_class_key]
            class_names = ['下跌', '震荡', '上涨']  # 根据实际类别调整
            if pred['probabilities'] is not None:
                print(f"走势预测：{class_names[int(pred['prediction'])]} (置信度: {pred['confidence']:.3f})")
                for i, prob in enumerate(pred['probabilities']):
                    print(f"  {class_names[i]}: {prob:.3f}")
        
        # 二分类收益概率解释
        thresholds = ['3pct', '5pct', '8pct']
        for threshold in thresholds:
            key = f'return_{prediction_horizon}_gt_{threshold}'
            if key in predictions:
                pred = predictions[key]
                threshold_val = threshold.replace('pct', '%')
                if pred['probabilities'] is not None:
                    prob_positive = pred['probabilities'][1] if len(pred['probabilities']) > 1 else pred['probabilities'][0]
                    print(f"收益>{threshold_val}概率: {prob_positive:.3f}")
        
        # 回归预测解释
        continuous_key = f'return_{prediction_horizon}_continuous'
        if continuous_key in predictions:
            pred = predictions[continuous_key]
            print(f"预期收益率: {pred['prediction']:.3f} ({pred['prediction']*100:.2f}%)")
        
        # 综合建议
        self.generate_recommendation(predictions, prediction_horizon)
    
    def generate_recommendation(self, predictions, prediction_horizon='5d'):
        """生成投资建议"""
        print(f"\n=== {prediction_horizon} 投资建议 ===")
        
        # 收集关键指标
        expected_return = None
        high_return_prob = 0.0
        trend_confidence = 0.0
        
        # 获取预期收益率
        continuous_key = f'return_{prediction_horizon}_continuous'
        if continuous_key in predictions:
            expected_return = predictions[continuous_key]['prediction']
        
        # 获取高收益概率
        for threshold in ['8pct', '5pct', '3pct']:
            key = f'return_{prediction_horizon}_gt_{threshold}'
            if key in predictions and predictions[key]['probabilities'] is not None:
                prob = predictions[key]['probabilities'][1] if len(predictions[key]['probabilities']) > 1 else 0
                if prob > high_return_prob:
                    high_return_prob = prob
                break
        
        # 获取趋势置信度
        multi_class_key = f'multi_class_{prediction_horizon}'
        if multi_class_key in predictions:
            trend_confidence = predictions[multi_class_key]['confidence']
        
        # 生成建议
        if expected_return is not None and expected_return > 0.05:  # 预期收益>5%
            if high_return_prob > 0.7 and trend_confidence > 0.6:
                recommendation = "强烈买入"
                reason = f"预期收益率{expected_return*100:.2f}%，高收益概率{high_return_prob:.2f}，趋势置信度{trend_confidence:.2f}"
            elif high_return_prob > 0.5:
                recommendation = "买入"
                reason = f"预期收益率{expected_return*100:.2f}%，高收益概率{high_return_prob:.2f}"
            else:
                recommendation = "谨慎买入"
                reason = f"预期收益率{expected_return*100:.2f}%，但高收益概率较低({high_return_prob:.2f})"
        elif expected_return is not None and expected_return > 0:
            recommendation = "持有"
            reason = f"预期收益率{expected_return*100:.2f}%，收益为正但不高"
        else:
            recommendation = "卖出/观望"
            reason = f"预期收益率{expected_return*100:.2f}% 为负" if expected_return is not None else "预期收益不明确"
        
        print(f"建议：{recommendation}")
        print(f"理由：{reason}")
    
    def predict_stock(self, stock_code, stock_data=None):
        """预测单只股票的完整分析"""
        print(f"\n{'='*50}")
        print(f"股票代码：{stock_code}")
        print(f"预测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        
        if stock_data is None:
            # 这里应该从数据源获取股票数据
            # 为了演示，使用随机数据
            print("警告：使用随机数据进行演示")
            stock_data = np.random.randn(len(self.feature_columns))
        
        # 对不同时间周期进行预测
        for horizon in ['5d', '10d', '20d']:
            try:
                predictions = self.predict_ensemble(stock_data, horizon)
                if predictions:
                    self.interpret_predictions(predictions, horizon)
                else:
                    print(f"\n{horizon} 预测：无可用模型")
            except Exception as e:
                print(f"\n{horizon} 预测失败：{str(e)}")

def main():
    """主函数：演示模型组合预测"""
    print("模型组合预测器演示")
    print("=" * 50)
    
    # 初始化预测器
    predictor = ModelEnsemblePredictor()
    
    # 加载模型
    if not predictor.load_models():
        print("错误：无法加载任何模型")
        return
    
    # 演示预测
    print("\n开始演示预测...")
    
    # 示例1：使用随机数据预测
    print("\n示例1：随机数据预测")
    predictor.predict_stock("000001.SZ")
    
    # 示例2：使用特定特征值预测
    print("\n\n示例2：特定特征值预测")
    # 创建一个示例特征向量（需要29个特征）
    sample_features = {
        'close': 10.5,
        'volume': 1000000,
        'rsi_14': 65.0,
        'ma_5': 10.2,
        'ma_20': 9.8,
        # ... 其他特征
    }
    
    # 由于我们知道需要29个特征，这里用简化的方式
    sample_array = np.random.randn(29)  # 实际使用时应该是真实的特征值
    predictor.predict_stock("600000.SH", sample_array)
    
    print("\n演示完成！")
    print("\n使用说明：")
    print("1. 替换 stock_data 为真实的股票特征数据")
    print("2. 根据实际需求调整预测周期")
    print("3. 根据模型配置调整类别名称和阈值")
    print("4. 可以添加更多的组合策略和风险控制")

if __name__ == "__main__":
    main()