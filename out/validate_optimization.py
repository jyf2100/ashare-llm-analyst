#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略优化效果验证工具
对比优化前后的策略表现
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import yaml
import os

class OptimizationValidator:
    """优化效果验证器"""
    
    def __init__(self, 
                 original_results_file: str = "analysis_results_v2.json",
                 optimized_config_file: str = "optimized_config.yaml"):
        self.original_results_file = original_results_file
        self.optimized_config_file = optimized_config_file
        self.original_results = self._load_original_results()
        self.optimized_config = self._load_optimized_config()
        
    def _load_original_results(self) -> Dict[str, Any]:
        """加载原始分析结果"""
        try:
            with open(self.original_results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 找不到原始结果文件 {self.original_results_file}")
            return {}
    
    def _load_optimized_config(self) -> Dict[str, Any]:
        """加载优化配置"""
        try:
            with open(self.optimized_config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"警告: 找不到优化配置文件 {self.optimized_config_file}")
            return {}
    
    def compare_parameters(self) -> Dict[str, Dict[str, Any]]:
        """对比参数变化"""
        print("📊 参数对比分析")
        print("="*50)
        
        # 原始参数（从代码中提取的默认值）
        original_params = {
            'basic': {
                'ma_trend_threshold': 1.0,
                'volume_ratio_threshold': 1.2,
                'price_position_threshold': 0.6,
                'rps_threshold': 70
            },
            'advanced': {
                'volume_surge_threshold': 2.0,
                'ma250_days_threshold': 15,
                'volatility_threshold': 0.05,
                'rps_threshold': 85
            },
            'momentum': {
                'momentum_5d_threshold': 0.05,
                'momentum_20d_threshold': 0.15,
                'volume_confirmation_threshold': 1.5,
                'price_position_threshold': 0.8,
                'rps_threshold': 80
            },
            'ml_enhanced': {
                'probability_threshold': 0.6,
                'use_ensemble': True,
                'feature_importance_threshold': 0.01
            }
        }
        
        optimized_strategies = self.optimized_config.get('strategies', {})
        comparison = {}
        
        for strategy_name in original_params.keys():
            if strategy_name in optimized_strategies:
                original = original_params[strategy_name]
                optimized = optimized_strategies[strategy_name].get('parameters', {})
                
                changes = {}
                for param, orig_value in original.items():
                    opt_value = optimized.get(param, orig_value)
                    if orig_value != opt_value:
                        if isinstance(orig_value, (int, float)):
                            change_pct = ((opt_value - orig_value) / orig_value) * 100
                            changes[param] = {
                                'original': orig_value,
                                'optimized': opt_value,
                                'change_pct': change_pct,
                                'direction': '↑' if opt_value > orig_value else '↓'
                            }
                        else:
                            changes[param] = {
                                'original': orig_value,
                                'optimized': opt_value,
                                'change_pct': 0,
                                'direction': '→'
                            }
                
                # 添加新参数
                for param, opt_value in optimized.items():
                    if param not in original:
                        changes[param] = {
                            'original': None,
                            'optimized': opt_value,
                            'change_pct': 0,
                            'direction': '✨'  # 新增
                        }
                
                comparison[strategy_name] = {
                    'changes': changes,
                    'total_changes': len(changes),
                    'improvement_score': optimized_strategies[strategy_name].get('optimization', {}).get('improvement_score', 0)
                }
                
                # 打印对比结果
                print(f"\n🔧 {strategy_name.upper()} 策略:")
                if changes:
                    for param, change in changes.items():
                        orig = change['original']
                        opt = change['optimized']
                        direction = change['direction']
                        
                        if orig is None:
                            print(f"  {direction} {param}: 新增 → {opt}")
                        elif isinstance(orig, (int, float)) and change['change_pct'] != 0:
                            print(f"  {direction} {param}: {orig} → {opt} ({change['change_pct']:+.1f}%)")
                        elif orig != opt:
                            print(f"  {direction} {param}: {orig} → {opt}")
                else:
                    print("  ✅ 无参数变化")
        
        return comparison
    
    def simulate_optimization_impact(self) -> Dict[str, Dict[str, float]]:
        """模拟优化影响"""
        print("\n🎯 优化影响预测")
        print("="*50)
        
        original_strategies = self.original_results.get('strategies', {})
        optimized_strategies = self.optimized_config.get('strategies', {})
        
        impact_analysis = {}
        
        for strategy_name, original_data in original_strategies.items():
            if strategy_name in optimized_strategies:
                original_rate = original_data.get('average_selection_rate', 0)
                original_score = original_data.get('average_score', 0)
                
                optimization_info = optimized_strategies[strategy_name].get('optimization', {})
                expected_rate = optimization_info.get('expected_selection_rate', original_rate)
                improvement_score = optimization_info.get('improvement_score', 0)
                
                # 预测新的评分（基于改进评分）
                predicted_score = original_score * (1 + improvement_score * 0.2)  # 假设最多提升20%
                
                impact = {
                    'original_selection_rate': original_rate,
                    'predicted_selection_rate': expected_rate,
                    'selection_rate_change': expected_rate - original_rate,
                    'selection_rate_change_pct': ((expected_rate - original_rate) / max(original_rate, 0.01)) * 100,
                    'original_score': original_score,
                    'predicted_score': predicted_score,
                    'score_improvement': predicted_score - original_score,
                    'improvement_score': improvement_score
                }
                
                impact_analysis[strategy_name] = impact
                
                # 打印预测结果
                print(f"\n📈 {strategy_name.upper()} 策略预测:")
                print(f"  选中率: {original_rate:.2f}% → {expected_rate:.2f}% ({impact['selection_rate_change_pct']:+.1f}%)")
                print(f"  评分: {original_score:.3f} → {predicted_score:.3f} ({impact['score_improvement']:+.3f})")
                print(f"  改进评分: {improvement_score:.1f}/1.0")
                
                # 评估影响
                if improvement_score > 0.8:
                    print(f"  🚀 预期显著改进")
                elif improvement_score > 0.5:
                    print(f"  📊 预期适度改进")
                elif improvement_score > 0.2:
                    print(f"  🔧 预期轻微改进")
                else:
                    print(f"  ✅ 当前表现良好")
        
        return impact_analysis
    
    def generate_test_plan(self) -> List[Dict[str, Any]]:
        """生成测试计划"""
        print("\n📋 测试计划生成")
        print("="*50)
        
        test_plan = []
        optimized_strategies = self.optimized_config.get('strategies', {})
        
        # 按改进评分排序，优先测试改进最大的策略
        sorted_strategies = sorted(
            optimized_strategies.items(),
            key=lambda x: x[1].get('optimization', {}).get('improvement_score', 0),
            reverse=True
        )
        
        for i, (strategy_name, strategy_config) in enumerate(sorted_strategies, 1):
            improvement_score = strategy_config.get('optimization', {}).get('improvement_score', 0)
            
            if improvement_score > 0:
                test_case = {
                    'priority': i,
                    'strategy': strategy_name,
                    'improvement_score': improvement_score,
                    'test_type': self._determine_test_type(improvement_score),
                    'test_duration': self._determine_test_duration(improvement_score),
                    'success_criteria': self._define_success_criteria(strategy_name, strategy_config),
                    'rollback_criteria': self._define_rollback_criteria(strategy_name)
                }
                test_plan.append(test_case)
                
                print(f"\n🧪 测试 {i}: {strategy_name.upper()} 策略")
                print(f"  优先级: {i} (改进评分: {improvement_score:.1f})")
                print(f"  测试类型: {test_case['test_type']}")
                print(f"  测试周期: {test_case['test_duration']}")
                print(f"  成功标准: {test_case['success_criteria']}")
        
        return test_plan
    
    def _determine_test_type(self, improvement_score: float) -> str:
        """确定测试类型"""
        if improvement_score >= 0.8:
            return "全面回测 + 实盘小额测试"
        elif improvement_score >= 0.5:
            return "历史数据回测"
        else:
            return "参数敏感性测试"
    
    def _determine_test_duration(self, improvement_score: float) -> str:
        """确定测试周期"""
        if improvement_score >= 0.8:
            return "2-4周"
        elif improvement_score >= 0.5:
            return "1-2周"
        else:
            return "3-5天"
    
    def _define_success_criteria(self, strategy_name: str, strategy_config: Dict) -> str:
        """定义成功标准"""
        optimization_info = strategy_config.get('optimization', {})
        expected_rate = optimization_info.get('expected_selection_rate', 0)
        
        if expected_rate > 0:
            return f"选中率达到 {expected_rate:.1f}% 且选股质量不下降"
        else:
            return "选中率提升且无明显质量下降"
    
    def _define_rollback_criteria(self, strategy_name: str) -> str:
        """定义回滚标准"""
        return "选中率异常波动 >50% 或连续3天无选股结果"
    
    def create_monitoring_dashboard_config(self) -> Dict[str, Any]:
        """创建监控仪表板配置"""
        dashboard_config = {
            'monitoring_metrics': {
                'selection_rate': {
                    'description': '每日选中率',
                    'alert_threshold': {'min': 0.1, 'max': 10.0},
                    'trend_window': 7
                },
                'average_score': {
                    'description': '平均评分',
                    'alert_threshold': {'min': 0.1, 'max': 1.0},
                    'trend_window': 7
                },
                'high_quality_stocks': {
                    'description': '高质量股票数量',
                    'alert_threshold': {'min': 1},
                    'trend_window': 14
                },
                'strategy_stability': {
                    'description': '策略稳定性(选中率标准差)',
                    'alert_threshold': {'max': 5.0},
                    'trend_window': 14
                }
            },
            'comparison_baseline': {
                'original_results_file': self.original_results_file,
                'comparison_period': '30天'
            },
            'alert_rules': {
                'selection_rate_drop': '选中率连续3天低于预期50%',
                'quality_degradation': '高质量股票数量连续5天为0',
                'stability_issue': '选中率标准差超过阈值',
                'score_decline': '平均评分连续下降超过20%'
            }
        }
        
        return dashboard_config
    
    def generate_validation_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("# 策略优化验证报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"基于分析: {self.original_results_file}")
        report.append(f"优化配置: {self.optimized_config_file}")
        
        # 参数对比
        report.append("\n## 参数对比分析")
        comparison = self.compare_parameters()
        
        for strategy_name, data in comparison.items():
            report.append(f"\n### {strategy_name.upper()} 策略")
            report.append(f"- 参数变化数量: {data['total_changes']}")
            report.append(f"- 改进评分: {data['improvement_score']:.1f}/1.0")
            
            if data['changes']:
                report.append("\n**参数变化:**")
                for param, change in data['changes'].items():
                    orig = change['original']
                    opt = change['optimized']
                    direction = change['direction']
                    
                    if orig is None:
                        report.append(f"- {param}: 新增 → {opt}")
                    elif orig != opt:
                        if isinstance(orig, (int, float)) and change['change_pct'] != 0:
                            report.append(f"- {param}: {orig} → {opt} ({change['change_pct']:+.1f}%)")
                        else:
                            report.append(f"- {param}: {orig} → {opt}")
        
        # 影响预测
        report.append("\n## 优化影响预测")
        impact_analysis = self.simulate_optimization_impact()
        
        for strategy_name, impact in impact_analysis.items():
            report.append(f"\n### {strategy_name.upper()} 策略")
            report.append(f"- 选中率变化: {impact['original_selection_rate']:.2f}% → {impact['predicted_selection_rate']:.2f}% ({impact['selection_rate_change_pct']:+.1f}%)")
            report.append(f"- 评分变化: {impact['original_score']:.3f} → {impact['predicted_score']:.3f} ({impact['score_improvement']:+.3f})")
            report.append(f"- 改进评分: {impact['improvement_score']:.1f}/1.0")
        
        # 测试计划
        report.append("\n## 测试计划")
        test_plan = self.generate_test_plan()
        
        for test_case in test_plan:
            report.append(f"\n### 测试 {test_case['priority']}: {test_case['strategy'].upper()}")
            report.append(f"- **优先级**: {test_case['priority']} (改进评分: {test_case['improvement_score']:.1f})")
            report.append(f"- **测试类型**: {test_case['test_type']}")
            report.append(f"- **测试周期**: {test_case['test_duration']}")
            report.append(f"- **成功标准**: {test_case['success_criteria']}")
            report.append(f"- **回滚标准**: {test_case['rollback_criteria']}")
        
        # 监控建议
        report.append("\n## 监控建议")
        report.append("\n### 关键指标")
        report.append("- 每日选中率变化")
        report.append("- 选股质量评分")
        report.append("- 高质量股票数量")
        report.append("- 策略稳定性")
        
        report.append("\n### 预警机制")
        report.append("- 选中率异常波动 (>50%)")
        report.append("- 连续多日无高质量股票")
        report.append("- 评分持续下降")
        report.append("- 策略失效 (连续无选股)")
        
        return "\n".join(report)
    
    def run_validation(self) -> bool:
        """运行完整验证"""
        print("🔍 策略优化验证")
        print("="*30)
        
        if not self.original_results or not self.optimized_config:
            print("❌ 缺少必要的数据文件")
            return False
        
        try:
            # 执行各项分析
            self.compare_parameters()
            self.simulate_optimization_impact()
            self.generate_test_plan()
            
            # 生成报告
            report = self.generate_validation_report()
            with open("optimization_validation_report.md", "w", encoding="utf-8") as f:
                f.write(report)
            
            # 生成监控配置
            dashboard_config = self.create_monitoring_dashboard_config()
            with open("monitoring_config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(dashboard_config, f, default_flow_style=False, allow_unicode=True)
            
            print("\n" + "="*50)
            print("✅ 验证完成!")
            print("\n📄 生成文件:")
            print("  - optimization_validation_report.md (验证报告)")
            print("  - monitoring_config.yaml (监控配置)")
            
            print("\n🎯 下一步行动:")
            print("  1. 审查验证报告")
            print("  2. 按优先级执行测试计划")
            print("  3. 设置监控仪表板")
            print("  4. 准备回滚方案")
            
            return True
            
        except Exception as e:
            print(f"❌ 验证过程出错: {e}")
            return False

def main():
    """主函数"""
    validator = OptimizationValidator()
    validator.run_validation()

if __name__ == "__main__":
    main()