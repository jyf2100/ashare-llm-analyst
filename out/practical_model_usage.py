#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实际模型使用示例

展示如何在实际股票分析中使用训练好的模型进行预测
包括特征提取、模型加载、预测和结果解释
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

class StockPredictor:
    """股票预测器 - 实际使用版本"""
    
    def __init__(self, models_dir='/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/demo_models'):
        self.models_dir = models_dir
        self.best_models = {}  # 存储每个任务的最佳模型
        self.scalers = {}
        self.configs = {}
        
    def load_best_models(self, date_suffix='20250818'):
        """加载性能最佳的模型"""
        print(f"正在加载最佳模型（日期：{date_suffix}）...")
        
        # 定义任务类型和对应的最佳模型选择策略
        task_configs = {
            'short_term_trend': {  # 短期趋势预测
                'model': 'multi_class_5d',
                'description': '5日走势预测',
                'type': 'classification'
            },
            'medium_term_trend': {  # 中期趋势预测
                'model': 'multi_class_10d', 
                'description': '10日走势预测',
                'type': 'classification'
            },
            'high_return_probability': {  # 高收益概率
                'model': 'return_5d_gt_8pct',
                'description': '5日收益>8%概率',
                'type': 'binary_classification'
            },
            'expected_return': {  # 预期收益
                'model': 'return_5d_continuous',
                'description': '5日预期收益率',
                'type': 'regression'
            }
        }
        
        loaded_tasks = []
        for task_name, config in task_configs.items():
            model_key = config['model']
            try:
                # 文件路径
                model_file = f'{self.models_dir}/demo_random_forest_{model_key}_{date_suffix}.pkl'
                config_file = f'{self.models_dir}/demo_model_config_{model_key}_{date_suffix}.yaml'
                scaler_file = f'{self.models_dir}/demo_scaler_random_forest_{model_key}_{date_suffix}.pkl'
                
                if os.path.exists(model_file) and os.path.exists(config_file):
                    # 加载模型
                    model = joblib.load(model_file)
                    
                    # 加载配置
                    with open(config_file, 'r', encoding='utf-8') as f:
                        model_config = yaml.safe_load(f)
                    
                    # 加载标准化器
                    scaler = None
                    if os.path.exists(scaler_file):
                        scaler = joblib.load(scaler_file)
                    
                    # 存储模型信息
                    self.best_models[task_name] = {
                        'model': model,
                        'scaler': scaler,
                        'config': model_config,
                        'type': config['type'],
                        'description': config['description']
                    }
                    
                    print(f"✓ {config['description']} - 准确率: {model_config.get('test_accuracy', 'N/A')}")
                    loaded_tasks.append(task_name)
                    
            except Exception as e:
                print(f"✗ 加载失败：{config['description']} - {str(e)}")
        
        print(f"\n成功加载 {len(loaded_tasks)} 个预测任务")
        return len(loaded_tasks) > 0
    
    def extract_features_from_data(self, stock_data):
        """从股票数据中提取特征
        
        Args:
            stock_data: DataFrame 包含股票历史数据
            
        Returns:
            numpy.array: 特征向量
        """
        # 这里需要实现与训练时相同的特征提取逻辑
        # 为了演示，我们使用一个简化版本
        
        if isinstance(stock_data, pd.DataFrame) and len(stock_data) > 0:
            # 基础价格特征
            latest = stock_data.iloc[-1]
            features = []
            
            # 价格相关特征
            features.extend([
                latest.get('close', 0),
                latest.get('open', 0),
                latest.get('high', 0),
                latest.get('low', 0),
                latest.get('volume', 0)
            ])
            
            # 技术指标（需要根据实际特征列表调整）
            # 这里添加24个额外特征以匹配训练时的29个特征
            for i in range(24):
                features.append(np.random.randn())  # 实际使用时应该是真实计算的技术指标
            
            return np.array(features[:29]).reshape(1, -1)  # 确保是29个特征
        
        else:
            # 如果没有数据，返回零向量
            return np.zeros((1, 29))
    
    def predict_stock_analysis(self, stock_code, stock_data):
        """对股票进行完整分析预测"""
        print(f"\n{'='*60}")
        print(f"股票分析报告 - {stock_code}")
        print(f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # 提取特征
        features = self.extract_features_from_data(stock_data)
        
        # 执行各项预测任务
        results = {}
        for task_name, model_info in self.best_models.items():
            try:
                result = self._predict_single_task(task_name, features)
                results[task_name] = result
            except Exception as e:
                print(f"预测任务 {task_name} 失败：{str(e)}")
                results[task_name] = None
        
        # 生成分析报告
        self._generate_analysis_report(results)
        
        return results
    
    def _predict_single_task(self, task_name, features):
        """执行单个预测任务"""
        model_info = self.best_models[task_name]
        model = model_info['model']
        scaler = model_info['scaler']
        task_type = model_info['type']
        
        # 特征标准化
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # 执行预测
        prediction = model.predict(features_scaled)[0]
        
        result = {
            'prediction': prediction,
            'task_type': task_type,
            'description': model_info['description']
        }
        
        # 如果是分类任务，获取概率
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            result['probabilities'] = probabilities
            result['confidence'] = np.max(probabilities)
        
        return result
    
    def _generate_analysis_report(self, results):
        """生成分析报告"""
        print("\n📊 预测结果分析")
        print("-" * 40)
        
        # 短期趋势分析
        if 'short_term_trend' in results and results['short_term_trend']:
            trend_result = results['short_term_trend']
            trend_names = ['下跌', '震荡', '上涨']
            trend_pred = int(trend_result['prediction'])
            confidence = trend_result.get('confidence', 0)
            
            print(f"📈 短期走势（5日）：{trend_names[trend_pred]} (置信度: {confidence:.2f})")
            if 'probabilities' in trend_result:
                for i, prob in enumerate(trend_result['probabilities']):
                    print(f"   {trend_names[i]}: {prob:.3f}")
        
        # 中期趋势分析
        if 'medium_term_trend' in results and results['medium_term_trend']:
            trend_result = results['medium_term_trend']
            trend_names = ['下跌', '震荡', '上涨']
            trend_pred = int(trend_result['prediction'])
            confidence = trend_result.get('confidence', 0)
            
            print(f"📈 中期走势（10日）：{trend_names[trend_pred]} (置信度: {confidence:.2f})")
        
        # 高收益概率分析
        if 'high_return_probability' in results and results['high_return_probability']:
            prob_result = results['high_return_probability']
            if 'probabilities' in prob_result:
                high_return_prob = prob_result['probabilities'][1] if len(prob_result['probabilities']) > 1 else 0
                print(f"💰 高收益概率（>8%）：{high_return_prob:.3f}")
        
        # 预期收益分析
        if 'expected_return' in results and results['expected_return']:
            return_result = results['expected_return']
            expected_return = return_result['prediction']
            print(f"💵 预期收益率：{expected_return:.4f} ({expected_return*100:.2f}%)")
        
        # 综合投资建议
        self._generate_investment_advice(results)
    
    def _generate_investment_advice(self, results):
        """生成投资建议"""
        print("\n💡 投资建议")
        print("-" * 40)
        
        # 收集关键指标
        short_trend = None
        medium_trend = None
        high_return_prob = 0.0
        expected_return = 0.0
        
        if 'short_term_trend' in results and results['short_term_trend']:
            short_trend = int(results['short_term_trend']['prediction'])
        
        if 'medium_term_trend' in results and results['medium_term_trend']:
            medium_trend = int(results['medium_term_trend']['prediction'])
        
        if 'high_return_probability' in results and results['high_return_probability']:
            prob_result = results['high_return_probability']
            if 'probabilities' in prob_result:
                high_return_prob = prob_result['probabilities'][1] if len(prob_result['probabilities']) > 1 else 0
        
        if 'expected_return' in results and results['expected_return']:
            expected_return = results['expected_return']['prediction']
        
        # 生成建议
        advice_score = 0
        reasons = []
        
        # 趋势评分
        if short_trend == 2:  # 上涨
            advice_score += 2
            reasons.append("短期趋势向上")
        elif short_trend == 1:  # 震荡
            advice_score += 0
            reasons.append("短期趋势震荡")
        else:  # 下跌
            advice_score -= 2
            reasons.append("短期趋势向下")
        
        if medium_trend == 2:  # 上涨
            advice_score += 1
            reasons.append("中期趋势向上")
        elif medium_trend == 0:  # 下跌
            advice_score -= 1
            reasons.append("中期趋势向下")
        
        # 收益评分
        if expected_return > 0.05:
            advice_score += 2
            reasons.append(f"预期收益率较高({expected_return*100:.1f}%)")
        elif expected_return > 0:
            advice_score += 1
            reasons.append(f"预期收益率为正({expected_return*100:.1f}%)")
        else:
            advice_score -= 2
            reasons.append(f"预期收益率为负({expected_return*100:.1f}%)")
        
        # 高收益概率评分
        if high_return_prob > 0.7:
            advice_score += 2
            reasons.append(f"高收益概率很高({high_return_prob:.2f})")
        elif high_return_prob > 0.5:
            advice_score += 1
            reasons.append(f"高收益概率中等({high_return_prob:.2f})")
        
        # 生成最终建议
        if advice_score >= 4:
            recommendation = "🟢 强烈买入"
        elif advice_score >= 2:
            recommendation = "🟡 买入"
        elif advice_score >= 0:
            recommendation = "🟠 持有/观望"
        else:
            recommendation = "🔴 卖出/回避"
        
        print(f"建议等级：{recommendation}")
        print(f"评分：{advice_score}/6")
        print("主要依据：")
        for reason in reasons:
            print(f"  • {reason}")
        
        # 风险提示
        print("\n⚠️  风险提示：")
        print("  • 模型预测仅供参考，不构成投资建议")
        print("  • 股市有风险，投资需谨慎")
        print("  • 建议结合基本面分析和市场环境")

def demo_usage():
    """演示如何使用股票预测器"""
    print("股票预测器使用演示")
    print("=" * 50)
    
    # 初始化预测器
    predictor = StockPredictor()
    
    # 加载模型
    if not predictor.load_best_models():
        print("错误：无法加载模型")
        return
    
    # 模拟股票数据
    print("\n正在模拟股票数据...")
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    stock_data = pd.DataFrame({
        'date': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # 执行预测分析
    results = predictor.predict_stock_analysis("000001.SZ", stock_data)
    
    print("\n" + "="*60)
    print("演示完成！")
    print("\n使用说明：")
    print("1. 将模拟数据替换为真实股票数据")
    print("2. 完善特征提取逻辑")
    print("3. 根据需要调整预测任务和模型选择")
    print("4. 可以批量分析多只股票")

if __name__ == "__main__":
    demo_usage()