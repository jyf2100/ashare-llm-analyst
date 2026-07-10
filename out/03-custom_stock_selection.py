#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的ML股票预测系统
专门预测RPS5 >85的股票未来5日收益率
这是一个独立条件，不依赖于原来的选股策略条件
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 尝试导入机器学习库
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    import pickle
    import joblib
    import yaml
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("机器学习库不可用，将跳过ML预测功能")

class MLStockPredictor:
    
    
    def __init__(self, data_dir: str = "market_data", rps_dir: str = "rps_results", models_dir: str = "/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/models"):
        """
        初始化ML股票预测器
        
        参数:
        data_dir: str, 股票数据目录
        rps_dir: str, RPS数据目录
        models_dir: str, 训练好的模型目录
        """
        self.data_dir = data_dir
        self.rps_dir = rps_dir
        self.models_dir = models_dir
        
        # 数据存储
        self.stock_data = {}
        self.rps_data = {}
        
        # 结果存储
        self.candidate_stocks = []
        self.strategy_selected_stocks = {}
        self.ml_predictions = {}
        
        # 模型存储
        self.loaded_models = {}
        self.loaded_scalers = {}
        self.model_configs = {}
        
        # 设置日志
        self._setup_logging()
        
        # 加载训练好的模型
        self._load_trained_models()
        
        logging.info("ML股票预测器初始化完成")
    
    def _load_trained_models(self):
        """
        加载训练好的模型和配置
        """
        if not ML_AVAILABLE:
            logging.warning("机器学习库不可用，跳过模型加载")
            return
        
        if not os.path.exists(self.models_dir):
            logging.warning(f"模型目录不存在: {self.models_dir}")
            return
        
        try:
            # 查找模型文件
            model_files = glob.glob(os.path.join(self.models_dir, "*.pkl"))
            config_files = glob.glob(os.path.join(self.models_dir, "*.yaml"))
            
            # 加载配置文件
            for config_file in config_files:
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    config_name = os.path.basename(config_file).replace('.yaml', '')
                    self.model_configs[config_name] = config
                    logging.info(f"加载配置: {config_name}")
                except Exception as e:
                    logging.error(f"加载配置文件失败 {config_file}: {e}")
            
            # 加载模型文件
            for model_file in model_files:
                try:
                    model_name = os.path.basename(model_file).replace('.pkl', '')
                    
                    # 区分模型和标准化器
                    if 'scaler_' in model_name:
                        # 这是标准化器
                        scaler = joblib.load(model_file)
                        scaler_key = model_name.replace('scaler_', '')
                        self.loaded_scalers[scaler_key] = scaler
                        logging.info(f"加载标准化器: {scaler_key}")
                    else:
                        # 这是模型
                        model = joblib.load(model_file)
                        self.loaded_models[model_name] = model
                        logging.info(f"加载模型: {model_name}")
                        
                except Exception as e:
                    logging.error(f"加载模型文件失败 {model_file}: {e}")
            
            logging.info(f"模型加载完成: {len(self.loaded_models)} 个模型, {len(self.loaded_scalers)} 个标准化器, {len(self.model_configs)} 个配置")
            
        except Exception as e:
            logging.error(f"加载训练好的模型时出错: {e}")
    
    def _select_best_model(self, prediction_days: int = 5, task_type: str = "regression") -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
        """
        智能选择最佳模型
        
        参数:
        prediction_days: int, 预测天数
        task_type: str, 任务类型 ("regression" 或 "classification")
        
        返回:
        Tuple[model, scaler, model_name]: 最佳模型、对应的标准化器和模型名称
        """
        if not self.loaded_models:
            logging.warning("没有可用的训练好的模型")
            return None, None, None
        
        # 根据预测天数和任务类型筛选合适的模型
        suitable_models = []
        
        for model_name, model in self.loaded_models.items():
            # 解析模型名称中的信息
            if f"{prediction_days}d" in model_name:
                # 检查任务类型匹配
                if task_type == "regression" and ("return" in model_name and "gt_" not in model_name):
                    suitable_models.append(model_name)
                elif task_type == "classification" and ("gt_" in model_name or "multi_class" in model_name):
                    suitable_models.append(model_name)
        
        if not suitable_models:
            # 如果没有完全匹配的模型，选择最接近的
            logging.warning(f"没有找到完全匹配 {prediction_days}d {task_type} 的模型，尝试选择最接近的模型")
            
            # 按优先级选择模型
            priority_patterns = [
                f"{prediction_days}d",  # 优先匹配天数
                "5d" if prediction_days <= 7 else "10d" if prediction_days <= 15 else "20d",  # 次优天数
                "return" if task_type == "regression" else "class",  # 任务类型
            ]
            
            for pattern in priority_patterns:
                for model_name in self.loaded_models.keys():
                    if pattern in model_name:
                        suitable_models.append(model_name)
                        break
                if suitable_models:
                    break
        
        if not suitable_models:
            logging.error("没有找到合适的模型")
            return None, None, None
        
        # 选择最佳模型（优先选择性能更好的模型）
        logging.info(f"🔍 筛选出的候选模型: {suitable_models}")
        best_model_name = self._rank_models_by_performance(suitable_models)
        
        if best_model_name:
            model = self.loaded_models[best_model_name]
            scaler = self.loaded_scalers.get(best_model_name, None)
            
            logging.info(f"选择模型: {best_model_name}")
            return model, scaler, best_model_name
        
        return None, None, None
    
    def _rank_models_by_performance(self, model_names: List[str]) -> Optional[str]:
        """
        根据模型性能排序选择最佳模型
        
        参数:
        model_names: List[str], 候选模型名称列表
        
        返回:
        Optional[str]: 最佳模型名称
        """
        if not model_names:
            return None
        
        # 优先选择random_forest模型
        for model_name in model_names:
            if "random_forest" in model_name:
                logging.info(f"🌲 优先选择Random Forest模型: {model_name}")
                return model_name
        
        # 如果没有random_forest，按原逻辑选择
        # 模型优先级排序（基于经验和性能）
        model_priority = {
            "random_forest": 1,     # 随机森林优先
            "gradient_boosting": 2,  # 梯度提升次之
            "logistic": 3,          # 逻辑回归
            "svm": 4                # SVM
        }
        
        # 根据配置文件中的性能指标选择
        best_model = None
        best_score = -float('inf')
        
        for model_name in model_names:
            # 查找对应的配置文件
            config_key = f"model_config_{model_name.replace('gradient_boosting_', '').replace('random_forest_', '')}"
            
            if config_key in self.model_configs:
                config = self.model_configs[config_key]
                
                # 从配置中获取性能指标
                if 'best_model' in config and 'performance' in config['best_model']:
                    performance = config['best_model']['performance']
                    
                    # 根据任务类型选择评估指标
                    if 'r2_score' in performance:  # 回归任务
                        score = performance['r2_score']
                    elif 'f1_score' in performance:  # 分类任务
                        score = performance['f1_score']
                    elif 'accuracy' in performance:
                        score = performance['accuracy']
                    else:
                        score = 0
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            # 如果没有配置信息，根据模型类型优先级选择
            if best_model is None:
                for model_type, priority in model_priority.items():
                    if model_type in model_name:
                        if best_model is None or priority < model_priority.get(best_model.split('_')[0], 999):
                            best_model = model_name
                        break
        
        return best_model or model_names[0]  # 如果都没有匹配，返回第一个
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        获取所有模型的性能摘要
        
        返回:
        Dict[str, Any]: 模型性能摘要
        """
        summary = {
            'total_models': len(self.loaded_models),
            'total_scalers': len(self.loaded_scalers),
            'total_configs': len(self.model_configs),
            'models_by_type': {},
            'models_by_period': {},
            'performance_ranking': []
        }
        
        # 按模型类型分组
        for model_name in self.loaded_models.keys():
            if 'gradient_boosting' in model_name:
                model_type = 'gradient_boosting'
            elif 'random_forest' in model_name:
                model_type = 'random_forest'
            elif 'logistic' in model_name:
                model_type = 'logistic_regression'
            elif 'svm' in model_name:
                model_type = 'svm'
            else:
                model_type = 'other'
            
            if model_type not in summary['models_by_type']:
                summary['models_by_type'][model_type] = []
            summary['models_by_type'][model_type].append(model_name)
        
        # 按预测周期分组
        for model_name in self.loaded_models.keys():
            if '5d' in model_name:
                period = '5d'
            elif '10d' in model_name:
                period = '10d'
            elif '20d' in model_name:
                period = '20d'
            else:
                period = 'unknown'
            
            if period not in summary['models_by_period']:
                summary['models_by_period'][period] = []
            summary['models_by_period'][period].append(model_name)
        
        # 性能排序
        performance_list = []
        for model_name in self.loaded_models.keys():
            config_key = f"model_config_{model_name.replace('gradient_boosting_', '').replace('random_forest_', '')}"
            
            if config_key in self.model_configs:
                config = self.model_configs[config_key]
                if 'model_results' in config:
                    # 确定模型类型
                    model_type = None
                    if 'gradient_boosting' in model_name:
                        model_type = 'gradient_boosting'
                    elif 'random_forest' in model_name:
                        model_type = 'random_forest'
                    
                    if model_type and model_type in config['model_results']:
                        performance = config['model_results'][model_type]
                        
                        # 获取主要性能指标
                        if 'f1' in performance:
                            score = performance['f1']
                            metric = 'F1'
                        elif 'auc' in performance:
                            score = performance['auc']
                            metric = 'AUC'
                        elif 'accuracy' in performance:
                            score = performance['accuracy']
                            metric = 'Accuracy'
                        else:
                            score = 0
                            metric = 'Unknown'
                        
                        performance_list.append({
                            'model_name': model_name,
                            'score': score,
                            'metric': metric,
                            'has_scaler': model_name in self.loaded_scalers
                        })
        
        # 按性能排序
        summary['performance_ranking'] = sorted(performance_list, key=lambda x: x['score'], reverse=True)
        
        return summary
    
    def print_model_selection_guide(self):
        """
        打印模型选择指南
        """
        print("\n" + "="*60)
        print("📊 模型选择指南")
        print("="*60)
        
        summary = self.get_model_performance_summary()
        
        print(f"\n📈 模型概览:")
        print(f"  • 总模型数量: {summary['total_models']}")
        print(f"  • 标准化器数量: {summary['total_scalers']}")
        print(f"  • 配置文件数量: {summary['total_configs']}")
        
        print(f"\n🔧 按模型类型分布:")
        for model_type, models in summary['models_by_type'].items():
            print(f"  • {model_type}: {len(models)} 个模型")
            for model in models[:3]:  # 只显示前3个
                print(f"    - {model}")
            if len(models) > 3:
                print(f"    - ... 还有 {len(models)-3} 个")
        
        print(f"\n📅 按预测周期分布:")
        for period, models in summary['models_by_period'].items():
            print(f"  • {period}: {len(models)} 个模型")
        
        print(f"\n🏆 性能排行榜 (前5名):")
        for i, model_info in enumerate(summary['performance_ranking'][:5], 1):
            scaler_status = "✅" if model_info['has_scaler'] else "❌"
            print(f"  {i}. {model_info['model_name']}")
            print(f"     {model_info['metric']}: {model_info['score']:.4f} | 标准化器: {scaler_status}")
        
        print(f"\n💡 选择建议:")
        print(f"  • 回归任务(预测收益率): 优先选择包含'return'的模型")
        print(f"  • 分类任务(涨跌预测): 优先选择包含'class'的模型")
        print(f"  • 短期预测(≤7天): 选择5d模型")
        print(f"  • 中期预测(8-15天): 选择10d模型")
        print(f"  • 长期预测(>15天): 选择20d模型")
        print(f"  • 性能优先级: Gradient Boosting > Random Forest > 其他")
        
        print("="*60)
    
    def backup_existing_results(self, output_dir="results"):
        """备份现有的选股结果文件"""
        if not os.path.exists(output_dir):
            return
        
        # 查找需要备份的文件类型
        backup_extensions = ['.json', '.csv', '.txt', '.pkl']
        files_to_backup = []
        
        # 检查results目录下的文件
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path) and any(file.endswith(ext) for ext in backup_extensions):
                files_to_backup.append(file)
        
        # 检查results_YYYYMMDD目录
        for item in os.listdir('.'):
            if os.path.isdir(item) and item.startswith('results_'):
                for file in os.listdir(item):
                    file_path = os.path.join(item, file)
                    if os.path.isfile(file_path) and any(file.endswith(ext) for ext in backup_extensions):
                        files_to_backup.append(os.path.join(item, file))
        
        if not files_to_backup:
            logging.info(f"选股结果目录中没有需要备份的文件")
            return
        
        # 创建备份目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join('back', f'selection_results_backup_{timestamp}')
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            logging.info(f"创建备份目录: {backup_dir}")
            
            # 移动文件到备份目录
            for file_path in files_to_backup:
                if os.path.exists(file_path):
                    # 保持目录结构
                    if os.path.dirname(file_path):
                        dst_dir = os.path.join(backup_dir, os.path.dirname(file_path))
                        os.makedirs(dst_dir, exist_ok=True)
                        dst_path = os.path.join(backup_dir, file_path)
                    else:
                        dst_path = os.path.join(backup_dir, file_path)
                    
                    shutil.move(file_path, dst_path)
                    logging.info(f"备份文件: {file_path} -> {dst_path}")
            
            # 删除空的results_YYYYMMDD目录
            for item in os.listdir('.'):
                if os.path.isdir(item) and item.startswith('results_'):
                    try:
                        if not os.listdir(item):  # 如果目录为空
                            os.rmdir(item)
                            logging.info(f"删除空目录: {item}")
                    except:
                        pass
            
            logging.info(f"成功备份 {len(files_to_backup)} 个选股结果文件到 {backup_dir}")
            
        except Exception as e:
            logging.error(f"备份选股结果文件时出错: {e}")
            raise
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('independent_stock_selection.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def load_stock_data(self, max_stocks: int = 100) -> bool:
        """
        加载股票数据
        
        参数:
        max_stocks: int, 最大加载股票数量
        
        返回:
        bool: 是否加载成功
        """
        logging.info(f"开始加载股票数据，最大数量: {max_stocks}")
        
        if not os.path.exists(self.data_dir):
            logging.error(f"股票数据目录不存在: {self.data_dir}")
            return False
        
        # 获取所有CSV文件
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        if not csv_files:
            logging.error(f"在目录 {self.data_dir} 中未找到CSV文件")
            return False
        
        # 限制加载数量
        csv_files = csv_files[:max_stocks]
        
        loaded_count = 0
        for csv_file in csv_files:
            try:
                # 从文件名提取股票代码
                filename = os.path.basename(csv_file)
                stock_code = filename.replace('.csv', '')
                
                # 读取CSV数据
                df = pd.read_csv(csv_file)
                
                # 检查必要的列
                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_columns):
                    logging.warning(f"股票 {stock_code} 缺少必要列，跳过")
                    continue
                
                # 处理日期列
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # 确保数据类型正确
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 删除包含NaN的行
                df.dropna(inplace=True)
                
                if len(df) < 30:  # 至少需要30天数据
                    logging.warning(f"股票 {stock_code} 数据不足30天，跳过")
                    continue
                
                self.stock_data[stock_code] = df
                loaded_count += 1
                
                if loaded_count % 10 == 0:
                    logging.info(f"已加载 {loaded_count} 只股票数据")
                
            except Exception as e:
                logging.error(f"加载股票数据失败 {csv_file}: {e}")
                continue
        
        logging.info(f"股票数据加载完成，成功加载 {loaded_count} 只股票")
        return loaded_count > 0
    
    def load_rps_data(self) -> bool:
        """
        加载RPS数据
        
        返回:
        bool: 是否加载成功
        """
        logging.info("开始加载RPS数据")
        
        if not os.path.exists(self.rps_dir):
            logging.error(f"RPS数据目录不存在: {self.rps_dir}，将生成模拟数据")
            return self._generate_mock_rps_data()
        
        # 查找RPS CSV文件
        rps_files = glob.glob(os.path.join(self.rps_dir, "RPS*.csv"))
        
        if not rps_files:
            logging.error(f"在目录 {self.rps_dir} 中未找到RPS文件，将生成模拟数据")
            return self._generate_mock_rps_data()
        
        loaded_periods = []
        
        for rps_file in rps_files:
            try:
                # 从文件名提取周期信息
                filename = os.path.basename(rps_file)
                if 'RPS5' in filename:
                    period = 'rps5'
                elif 'RPS10' in filename:
                    period = 'rps10'
                elif 'RPS20' in filename:
                    period = 'rps20'
                elif 'RPS60' in filename:
                    period = 'rps60'
                elif 'RPS120' in filename:
                    period = 'rps120'
                elif 'RPS250' in filename:
                    period = 'rps250'
                else:
                    continue
                
                # 读取RPS数据
                df = pd.read_csv(rps_file)
                
                # 检查必要的列
                if not all(col in df.columns for col in ['stock_code', 'date', 'rps']):
                    logging.warning(f"RPS文件 {rps_file} 格式不正确，跳过")
                    continue
                
                # 处理数据
                df['date'] = pd.to_datetime(df['date'])
                
                # 按股票代码分组
                for stock_code, group in df.groupby('stock_code'):
                    if stock_code not in self.rps_data:
                        self.rps_data[stock_code] = {}
                    
                    # 创建日期到RPS值的映射
                    rps_dict = dict(zip(group['date'].dt.strftime('%Y-%m-%d'), group['rps']))
                    self.rps_data[stock_code][period] = rps_dict
                
                loaded_periods.append(period)
                logging.info(f"成功加载 {period} 数据，包含 {len(df)} 条记录")
                
            except Exception as e:
                logging.error(f"加载RPS文件失败 {rps_file}: {e}")
                continue
        
        if loaded_periods:
            logging.info(f"RPS数据加载完成，成功加载周期: {loaded_periods}")
            return True
        else:
            logging.warning("未能加载任何RPS数据，将生成模拟数据")
            return self._generate_mock_rps_data()
    
    def _generate_mock_rps_data(self) -> bool:
        """
        生成模拟RPS数据
        
        返回:
        bool: 是否生成成功
        """
        logging.info("生成模拟RPS数据")
        
        for stock_code in self.stock_data.keys():
            stock_df = self.stock_data[stock_code]
            dates = stock_df.index.strftime('%Y-%m-%d').tolist()
            
            # 生成模拟RPS数据，确保递减关系：RPS5 > RPS10 > RPS20 > RPS60 > RPS120 > RPS250
            base_rps = np.random.normal(70, 15, len(dates))  # 基础RPS值
            
            rps5_values = base_rps + np.random.normal(15, 5, len(dates))  # 最高
            rps10_values = base_rps + np.random.normal(10, 5, len(dates))
            rps20_values = base_rps + np.random.normal(5, 5, len(dates))
            rps60_values = base_rps + np.random.normal(0, 5, len(dates))
            rps120_values = base_rps + np.random.normal(-5, 5, len(dates))
            rps250_values = base_rps + np.random.normal(-10, 5, len(dates))  # 最低
            
            # 确保RPS值在合理范围内
            rps5_values = np.clip(rps5_values, 0, 100)
            rps10_values = np.clip(rps10_values, 0, 100)
            rps20_values = np.clip(rps20_values, 0, 100)
            rps60_values = np.clip(rps60_values, 0, 100)
            rps120_values = np.clip(rps120_values, 0, 100)
            rps250_values = np.clip(rps250_values, 0, 100)
            
            self.rps_data[stock_code] = {
                'rps5': dict(zip(dates, rps5_values)),
                'rps10': dict(zip(dates, rps10_values)),
                'rps20': dict(zip(dates, rps20_values)),
                'rps60': dict(zip(dates, rps60_values)),
                'rps120': dict(zip(dates, rps120_values)),
                'rps250': dict(zip(dates, rps250_values))
            }
        
        logging.info(f"模拟RPS数据生成完成，覆盖 {len(self.rps_data)} 只股票，包含周期：rps5, rps10, rps20, rps60, rps120, rps250")
        return True
    
    def filter_by_rps5(self, threshold: float = 80) -> List[str]:
        """
        筛选RPS5 > threshold的股票作为ML预测候选
        
        参数:
        threshold: float, RPS5阈值（默认80）
        
        返回:
        List[str]: 候选股票代码列表
        """
        logging.info(f"🔍 筛选 RPS5 > {threshold} 的股票")
        
        candidates = []
        
        for stock_code in self.stock_data.keys():
            if stock_code in self.rps_data:
                rps_data = self.rps_data[stock_code]
                
                # 检查是否有RPS5数据
                if 'rps5' in rps_data and rps_data['rps5']:
                    # 获取最新日期的RPS5值
                    latest_date = max(rps_data['rps5'].keys())
                    rps5_value = rps_data['rps5'][latest_date]
                    
                    # 检查RPS5阈值条件
                    if  rps5_value > threshold:
                        candidates.append(stock_code)
                        logging.debug(f"股票 {stock_code} 符合RPS5条件: RPS5={rps5_value:.1f}")
        
        self.candidate_stocks = candidates
        logging.info(f"✅ 筛选完成：找到 {len(candidates)} 只RPS5 > {threshold}的股票")
        
        return candidates
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        参数:
        df: pd.DataFrame, 股票数据
        
        返回:
        pd.DataFrame: 包含技术指标的数据
        """
        result_df = df.copy()
        
        # 移动平均线
        result_df['ma5'] = df['close'].rolling(window=5).mean()
        result_df['ma10'] = df['close'].rolling(window=10).mean()
        result_df['ma20'] = df['close'].rolling(window=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result_df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        result_df['macd'] = exp1 - exp2
        result_df['macd_signal'] = result_df['macd'].ewm(span=9).mean()
        
        # 布林带
        result_df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        result_df['bb_upper'] = result_df['bb_middle'] + (bb_std * 2)
        result_df['bb_lower'] = result_df['bb_middle'] - (bb_std * 2)
        
        # 成交量指标
        result_df['volume_ma'] = df['volume'].rolling(window=20).mean()
        result_df['volume_ratio'] = df['volume'] / result_df['volume_ma']
        
        return result_df
    
    def _apply_basic_strategy(self, stock_code: str, df: pd.DataFrame) -> bool:
        """
        应用基础策略
        
        参数:
        stock_code: str, 股票代码
        df: pd.DataFrame, 股票数据
        
        返回:
        bool: 是否选中
        """
        try:
            # 计算技术指标
            df_with_indicators = self._calculate_technical_indicators(df)
            latest = df_with_indicators.iloc[-1]
            
            # 基础策略条件
            conditions = [
                latest['close'] > latest['ma5'],  # 价格在5日均线之上
                latest['ma5'] > latest['ma10'],   # 5日均线在10日均线之上
                latest['rsi'] > 30 and latest['rsi'] < 70,  # RSI在合理范围
                latest['volume_ratio'] > 1.2,    # 成交量放大
            ]
            
            return sum(conditions) >= 3  # 至少满足3个条件
            
        except Exception as e:
            logging.error(f"基础策略计算失败 {stock_code}: {e}")
            return False
    
    def _apply_advanced_strategy(self, stock_code: str, df: pd.DataFrame) -> bool:
        """
        应用高级策略
        
        参数:
        stock_code: str, 股票代码
        df: pd.DataFrame, 股票数据
        
        返回:
        bool: 是否选中
        """
        try:
            # 计算技术指标
            df_with_indicators = self._calculate_technical_indicators(df)
            latest = df_with_indicators.iloc[-1]
            prev = df_with_indicators.iloc[-2]
            
            # 高级策略条件
            conditions = [
                latest['close'] > latest['bb_middle'],  # 价格在布林带中轨之上
                latest['macd'] > latest['macd_signal'],  # MACD金叉
                latest['macd'] > prev['macd'],  # MACD上升
                latest['close'] > prev['close'],  # 价格上涨
                latest['volume'] > prev['volume'],  # 成交量增加
            ]
            
            return sum(conditions) >= 4  # 至少满足4个条件
            
        except Exception as e:
            logging.error(f"高级策略计算失败 {stock_code}: {e}")
            return False
    
    def _apply_momentum_strategy(self, stock_code: str, df: pd.DataFrame) -> bool:
        """
        应用动量策略
        
        参数:
        stock_code: str, 股票代码
        df: pd.DataFrame, 股票数据
        
        返回:
        bool: 是否选中
        """
        try:
            # 计算收益率
            df['return_1d'] = df['close'].pct_change()
            df['return_5d'] = df['close'].pct_change(5)
            df['return_10d'] = df['close'].pct_change(10)
            
            latest = df.iloc[-1]
            
            # 动量策略条件
            conditions = [
                latest['return_1d'] > 0,     # 1日收益率为正
                latest['return_5d'] > 0.02,  # 5日收益率大于2%
                latest['return_10d'] > 0,    # 10日收益率为正
                latest['close'] > df['close'].rolling(20).max().iloc[-2],  # 创20日新高
            ]
            
            return sum(conditions) >= 3  # 至少满足3个条件
            
        except Exception as e:
            logging.error(f"动量策略计算失败 {stock_code}: {e}")
            return False
    
    def step2_apply_strategies(self, analysis_days: int = 30) -> Dict[str, List[str]]:
        """
        第二步：在候选股票中应用策略筛选
        
        参数:
        analysis_days: int, 分析天数，默认30天
        
        返回:
        Dict[str, List[str]]: 各策略筛选出的股票
        """
        logging.info(f"📊 第二步：在 {len(self.candidate_stocks)} 只候选股票中应用策略筛选")
        
        if not self.candidate_stocks:
            logging.warning("没有候选股票，跳过策略筛选")
            return {}
        
        strategy_results = {
            'basic': [],
            'advanced': [],
            'momentum': []
        }
        
        for stock_code in self.candidate_stocks:
            if stock_code not in self.stock_data:
                continue
            
            # 获取最近的数据
            df = self.stock_data[stock_code].tail(analysis_days).copy()
            
            if len(df) < 20:  # 至少需要20天数据进行技术分析
                continue
            
            # 应用各种策略
            try:
                if self._apply_basic_strategy(stock_code, df):
                    strategy_results['basic'].append(stock_code)
                    logging.debug(f"股票 {stock_code} 被基础策略选中")
                
                if self._apply_advanced_strategy(stock_code, df):
                    strategy_results['advanced'].append(stock_code)
                    logging.debug(f"股票 {stock_code} 被高级策略选中")
                
                if self._apply_momentum_strategy(stock_code, df):
                    strategy_results['momentum'].append(stock_code)
                    logging.debug(f"股票 {stock_code} 被动量策略选中")
                    
            except Exception as e:
                logging.error(f"策略分析失败 {stock_code}: {e}")
                continue
        
        self.strategy_selected_stocks = strategy_results
        
        # 输出结果统计
        for strategy_name, selected_stocks in strategy_results.items():
            logging.info(f"✅ {strategy_name} 策略选中 {len(selected_stocks)} 只股票")
        
        return strategy_results
    
    def _prepare_ml_features(self, df: pd.DataFrame, stock_code: str = None) -> pd.DataFrame:
        """
        准备机器学习特征（匹配训练模型的29个特征）
        
        参数:
        df: pd.DataFrame, 股票数据
        stock_code: str, 股票代码（用于获取RPS数据）
        
        返回:
        pd.DataFrame: 包含29个特征的数据框
        """
        # 计算技术指标
        df_with_indicators = self._calculate_technical_indicators(df)
        
        # 计算移动平均线（与训练时一致）
        df_with_indicators['ma10'] = df['close'].rolling(window=10).mean()
        df_with_indicators['ma30'] = df['close'].rolling(window=30).mean()
        df_with_indicators['ma60'] = df['close'].rolling(window=60).mean()
        df_with_indicators['ma120'] = df['close'].rolling(window=120).mean()
        
        # 计算价格位置指标
        high_20 = df['high'].rolling(window=20).max()
        low_20 = df['low'].rolling(window=20).min()
        df_with_indicators['price_position_20'] = (df['close'] - low_20) / (high_20 - low_20)
        
        # 计算MACD直方图
        df_with_indicators['macd_histogram'] = df_with_indicators['macd'] - df_with_indicators['macd_signal']
        
        # 计算布林带位置
        df_with_indicators['bb_position'] = (df['close'] - df_with_indicators['bb_lower']) / (df_with_indicators['bb_upper'] - df_with_indicators['bb_lower'])
        
        # 计算趋势强度（与训练时一致：ma20相对于ma60）
        df_with_indicators['trend_strength'] = (df_with_indicators['ma20'] - df_with_indicators['ma60']) / df_with_indicators['ma60']
        
        # 计算动量指标
        df_with_indicators['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # 计算各种收益率（与训练时一致）
        df_with_indicators['return_1d'] = df['close'].pct_change()
        df_with_indicators['return_3d'] = df['close'].pct_change(3)
        df_with_indicators['return_5d'] = df['close'].pct_change(5)
        df_with_indicators['return_10d'] = df['close'].pct_change(10)
        df_with_indicators['return_20d'] = df['close'].pct_change(20)
        df_with_indicators['return_60d'] = df['close'].pct_change(60)
        
        # 计算波动率（与训练时一致）
        df_with_indicators['volatility_10d'] = df['close'].pct_change().rolling(window=10).std()
        df_with_indicators['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
        df_with_indicators['volatility_60d'] = df['close'].pct_change().rolling(window=60).std()
        
        # 添加RPS数据
        if stock_code and stock_code in self.rps_data:
            rps_data = self.rps_data[stock_code]
            
            # 为每个日期添加RPS值
            for idx in df_with_indicators.index:
                date_str = idx.strftime('%Y-%m-%d')
                
                # 添加各个周期的RPS值
                for rps_period in ['rps5', 'rps10', 'rps20', 'rps60']:
                    if rps_period in rps_data and date_str in rps_data[rps_period]:
                        df_with_indicators.loc[idx, rps_period] = rps_data[rps_period][date_str]
                    else:
                        df_with_indicators.loc[idx, rps_period] = np.nan
        else:
            # 如果没有RPS数据，用默认值填充
            for rps_period in ['rps5', 'rps10', 'rps20', 'rps60']:
                df_with_indicators[rps_period] = 50.0  # 使用中位数作为默认值
        
        # 选择与训练时一致的29个特征（按照config_20250816.yaml中的顺序）
        feature_columns = [
            'close', 'ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma120',
            'price_position_20', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volume_ratio', 'trend_strength', 'momentum_20',
            'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d', 'return_60d',
            'volatility_10d', 'volatility_20d', 'volatility_60d',
            'rps5', 'rps10', 'rps20', 'rps60'
        ]
        
        # 确保所有特征列都存在，如果不存在则填充默认值
        for col in feature_columns:
            if col not in df_with_indicators.columns:
                df_with_indicators[col] = 0.0
        
        # 只返回指定的29个特征
        return df_with_indicators[feature_columns]
    
    def step3_ml_prediction(self, prediction_days: int = 5) -> Dict[str, Dict[str, float]]:
        """
        第三步：使用ML预测选中股票的未来收益率
        
        参数:
        prediction_days: int, 预测天数，默认5天
        
        返回:
        Dict[str, Dict[str, float]]: 各股票的ML预测结果
        """
        logging.info(f"🤖 第三步：使用ML预测选中股票的未来 {prediction_days} 日收益率")
        
        if not ML_AVAILABLE:
            logging.warning("机器学习库不可用，跳过ML预测")
            return {}
        
        # 收集所有被策略选中的股票（去重）
        all_selected_stocks = set()
        for strategy_stocks in self.strategy_selected_stocks.values():
            all_selected_stocks.update(strategy_stocks)
        
        all_selected_stocks = list(all_selected_stocks)
        
        if not all_selected_stocks:
            logging.warning("没有被策略选中的股票，跳过ML预测")
            return {}
        
        logging.info(f"开始对 {len(all_selected_stocks)} 只股票进行ML预测")
        
        ml_predictions = {}
        
        for stock_code in all_selected_stocks:
            try:
                if stock_code not in self.stock_data:
                    continue
                
                df = self.stock_data[stock_code].copy()
                
                if len(df) < 50:  # 至少需要50天数据进行ML训练
                    continue
                
                # 准备特征和标签
                features_df = self._prepare_ml_features(df, stock_code)
                
                if len(features_df) < 30:
                    continue
                
                # 选择数值特征列（排除非数值列）
                numeric_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
                features = features_df[numeric_columns].fillna(0)  # 填充NaN值
                
                if len(features) < 30:
                    continue
                
                # 计算未来收益率作为标签
                future_returns = df['close'].pct_change(prediction_days).shift(-prediction_days)
                
                # 对齐特征和标签
                min_len = min(len(features), len(future_returns))
                features = features[:min_len]
                labels = future_returns.iloc[:min_len].dropna()
                
                if len(labels) < 20:
                    continue
                
                # 对齐数据
                features = features[:len(labels)]
                
                # 选择最佳模型 - 优先使用回归模型预测收益率
                best_model, best_scaler, model_name = self._select_best_model(
                    prediction_days=prediction_days, 
                    task_type="regression"
                )
                
                if best_model is None:
                    # 如果没有训练好的模型，使用简单的临时模型
                    logging.warning(f"股票 {stock_code} 没有合适的训练模型，使用临时模型")
                    
                    # 分割训练和测试数据
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.2, random_state=42
                    )
                    
                    # 特征标准化 - 使用RobustScaler与训练时保持一致
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # 训练临时模型
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    
                    # 预测最新数据
                    latest_features = features[-1:]
                    latest_features_scaled = scaler.transform(latest_features)
                    predicted_return = model.predict(latest_features_scaled)[0]
                    
                    # 计算预测置信度
                    test_score = model.score(X_test_scaled, y_test)
                    confidence = max(0, min(1, test_score))
                    model_info = "临时训练模型"
                    
                else:
                    # 使用训练好的模型进行预测
                    logging.info(f"股票 {stock_code} 使用训练好的模型: {model_name}")
                    
                    # 准备预测特征
                    latest_features = features[-1:]
                    
                    # 使用对应的标准化器
                    if best_scaler is not None:
                        latest_features_scaled = best_scaler.transform(latest_features)
                    else:
                        # 如果没有标准化器，使用当前数据进行标准化 - 使用RobustScaler与训练时保持一致
                        scaler = RobustScaler()
                        scaler.fit(features)
                        latest_features_scaled = scaler.transform(latest_features)
                        logging.warning(f"模型 {model_name} 没有对应的标准化器，使用RobustScaler进行当前数据标准化")
                    
                    # 检查模型类型并进行相应预测
                    if "gt_" in model_name:  # 分类模型
                        # 进行预测 - 分类模型返回概率
                        prediction_proba = best_model.predict_proba(latest_features_scaled)[0]
                        positive_prob = prediction_proba[1]  # 正类概率（收益率>阈值的概率）
                        
                        # 根据模型名称确定阈值
                        if "gt_3pct" in model_name:
                            threshold = 0.03
                        elif "gt_5pct" in model_name:
                            threshold = 0.05
                        elif "gt_8pct" in model_name:
                            threshold = 0.08
                        else:
                            threshold = 0.05  # 默认5%
                        
                        # 基于概率估算预期收益率（更合理的映射）
                        # 概率越高，预期收益率越接近阈值的正值
                        predicted_return = threshold * (positive_prob - 0.5) * 2
                        
                        # 使用概率作为置信度
                        confidence = max(positive_prob, 1 - positive_prob)  # 取更高的概率作为置信度
                        
                    else:  # 回归模型
                        # 直接预测收益率
                        predicted_return = best_model.predict(latest_features_scaled)[0]
                        
                        # 计算置信度（基于预测值的绝对值，但限制在合理范围内）
                        confidence = min(0.9, max(0.1, 1.0 - abs(predicted_return) / 0.2))  # 收益率越接近0，置信度越高
                    
                    model_info = f"训练模型: {model_name}"
                
                ml_predictions[stock_code] = {
                    'predicted_return_5d': predicted_return * 100,  # 转换为百分比
                    'confidence': confidence,
                    'model_info': model_info,
                    'prediction_date': df.index[-1].strftime('%Y-%m-%d'),
                    'features_count': len(features.columns),
                    'data_points': len(features)
                }
                
                logging.debug(f"股票 {stock_code} ML预测完成: 预期收益率={predicted_return*100:.3f}%, 置信度={confidence:.3f}, 模型={model_info}")
                
            except Exception as e:
                logging.error(f"股票 {stock_code} ML预测失败: {e}")
                continue
        
        self.ml_predictions = ml_predictions
        logging.info(f"✅ 第三步完成：成功预测 {len(ml_predictions)} 只股票")
        
        return ml_predictions
    
    def predict_rps_stocks(self, rps5_threshold: float = 85, prediction_days: int = 5, max_stocks: int = 100, enable_strategy_filter: bool = True) -> Dict[str, Any]:
        """
        预测RPS5 >85的股票未来5日收益率
        
        参数:
        rps5_threshold: float, RPS5阈值（默认85）
        prediction_days: int, 预测天数（默认5天）
        max_stocks: int, 最大加载股票数量
        enable_strategy_filter: bool, 是否启用策略筛选（默认True）
        
        返回:
        Dict[str, Any]: 完整的选股和ML预测结果
        """
        logging.info(f"🚀 开始完整选股流程：RPS5 >{rps5_threshold} + 策略筛选 + ML预测{prediction_days}日收益率")
        start_time = datetime.now()
        
        try:
            # 第一步：加载数据
            logging.info("📊 第一步：加载股票和RPS数据")
            if not self.load_stock_data(max_stocks):
                raise Exception("股票数据加载失败")
            
            if not self.load_rps_data():
                raise Exception("RPS数据加载失败")
            
            # 第二步：筛选RPS5 > threshold的股票
            logging.info(f"🔍 第二步：筛选RPS5 > {rps5_threshold}的候选股票")
            candidates = self.filter_by_rps5(rps5_threshold)
            if not candidates:
                logging.warning(f"⚠️ 没有找到符合RPS5 > {rps5_threshold}条件的股票")
                return {}
            
            # 设置候选股票
            self.candidate_stocks = candidates
            logging.info(f"✅ 找到 {len(candidates)} 只符合RPS条件的候选股票")
            
            # 第三步：应用策略筛选（可选）
            strategy_results = {}
            if enable_strategy_filter:
                logging.info("📈 第三步：应用策略筛选")
                strategy_results = self.step2_apply_strategies(analysis_days=30)
                
                # 合并所有策略选中的股票
                all_strategy_stocks = set()
                for strategy_name, stocks in strategy_results.items():
                    all_strategy_stocks.update(stocks)
                
                # 更新strategy_selected_stocks为策略筛选后的股票
                if all_strategy_stocks:
                    self.strategy_selected_stocks = {'combined_strategies': list(all_strategy_stocks)}
                    logging.info(f"✅ 策略筛选完成，共选中 {len(all_strategy_stocks)} 只股票")
                else:
                    logging.warning("⚠️ 策略筛选未选中任何股票，将使用所有候选股票进行ML预测")
                    self.strategy_selected_stocks = {'rps_filtered': candidates}
            else:
                logging.info("⏭️ 跳过策略筛选，直接使用RPS筛选结果")
                self.strategy_selected_stocks = {'rps_filtered': candidates}
            
            # 第四步：ML预测
            logging.info("🤖 第四步：ML预测")
            ml_results = self.step3_ml_prediction(prediction_days)
            
            # 汇总结果
            final_results = {
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'rps5_threshold': rps5_threshold,
                    'prediction_days': prediction_days,
                    'max_stocks': max_stocks,
                    'enable_strategy_filter': enable_strategy_filter
                },
                'step1_candidates': {
                    'count': len(candidates),
                    'stocks': candidates
                },
                'step2_strategy_selection': {
                    strategy_name: {
                        'count': len(stocks),
                        'stocks': stocks
                    } for strategy_name, stocks in strategy_results.items()
                } if strategy_results else {},
                'step3_ml_predictions': ml_results,
                'summary': {
                    'total_loaded_stocks': len(self.stock_data),
                    'total_candidates': len(candidates),
                    'total_strategy_selected': sum(len(stocks) for stocks in strategy_results.values()) if strategy_results else 0,
                    'total_ml_predicted': len(ml_results),
                    'execution_time': (datetime.now() - start_time).total_seconds()
                }
            }
            
            # 保存结果
            self.save_results(final_results)
            
            # 按板块分类保存结果
            self.save_results_by_board(final_results)
            
            # 打印报告
            self.print_selection_report(final_results)
            
            logging.info(f"🎉 完整选股流程完成，总耗时: {final_results['summary']['execution_time']:.2f}秒")
            
            return final_results
            
        except Exception as e:
            logging.error(f"选股流程执行失败: {e}")
            raise
    
    def run_complete_strategy_selection(self, rps5_threshold: float = 85, prediction_days: int = 5, max_stocks: int = 100) -> Dict[str, Any]:
        """
        运行完整的策略选股流程：RPS筛选 + 策略筛选 + ML预测
        
        参数:
        rps5_threshold: float, RPS5阈值（默认85）
        prediction_days: int, 预测天数（默认5天）
        max_stocks: int, 最大加载股票数量
        
        返回:
        Dict[str, Any]: 完整的选股和ML预测结果
        """
        return self.predict_rps_stocks(
            rps5_threshold=rps5_threshold,
            prediction_days=prediction_days,
            max_stocks=max_stocks,
            enable_strategy_filter=True
        )
    
    def run_rps_only_prediction(self, rps5_threshold: float = 85, prediction_days: int = 5, max_stocks: int = 100) -> Dict[str, Any]:
        """
        运行仅基于RPS的ML预测（跳过策略筛选）
        
        参数:
        rps5_threshold: float, RPS5阈值（默认85）
        prediction_days: int, 预测天数（默认5天）
        max_stocks: int, 最大加载股票数量
        
        返回:
        Dict[str, Any]: RPS筛选 + ML预测结果
        """
        return self.predict_rps_stocks(
            rps5_threshold=rps5_threshold,
            prediction_days=prediction_days,
            max_stocks=max_stocks,
            enable_strategy_filter=False
        )
    
    def run_combined_strategy_selection(self, rps5_threshold: float = 85, prediction_days: int = 5, max_stocks: int = 100) -> Dict[str, Any]:
        """
        运行合并策略选股流程：选项1 + 选项2 结果合并去重
        
        参数:
        rps5_threshold: float, RPS5阈值（默认85）
        prediction_days: int, 预测天数（默认5天）
        max_stocks: int, 最大加载股票数量
        
        返回:
        Dict[str, Any]: 合并去重后的选股和ML预测结果
        """
        start_time = datetime.now()
        logging.info(f"🔄 开始合并策略选股流程：选项1 + 选项2 结果合并去重")
        
        try:
            # 运行选项1：完整策略选股流程
            logging.info("📊 执行选项1：完整策略选股流程")
            results1 = self.run_complete_strategy_selection(
                rps5_threshold=rps5_threshold,
                prediction_days=prediction_days,
                max_stocks=max_stocks
            )
            
            # 运行选项2：仅RPS + ML预测
            logging.info("📊 执行选项2：仅RPS + ML预测")
            results2 = self.run_rps_only_prediction(
                rps5_threshold=rps5_threshold,
                prediction_days=prediction_days,
                max_stocks=max_stocks
            )
            
            # 合并ML预测结果并去重
            combined_ml_predictions = {}
            
            # 添加选项1的ML预测结果
            if 'step3_ml_predictions' in results1:
                combined_ml_predictions.update(results1['step3_ml_predictions'])
                logging.info(f"✅ 选项1贡献 {len(results1['step3_ml_predictions'])} 只股票预测")
            
            # 添加选项2的ML预测结果（去重）
            if 'step3_ml_predictions' in results2:
                for stock_code, prediction in results2['step3_ml_predictions'].items():
                    if stock_code not in combined_ml_predictions:
                        combined_ml_predictions[stock_code] = prediction
                logging.info(f"✅ 选项2额外贡献 {len([s for s in results2['step3_ml_predictions'].keys() if s not in results1.get('step3_ml_predictions', {})])} 只股票预测")
            
            # 合并候选股票并去重
            combined_candidates = set()
            if 'step1_candidates' in results1 and 'stocks' in results1['step1_candidates']:
                combined_candidates.update(results1['step1_candidates']['stocks'])
            if 'step1_candidates' in results2 and 'stocks' in results2['step1_candidates']:
                combined_candidates.update(results2['step1_candidates']['stocks'])
            combined_candidates = list(combined_candidates)
            
            # 合并策略筛选结果
            combined_strategy_results = {}
            if 'step2_strategy_selection' in results1:
                combined_strategy_results.update(results1['step2_strategy_selection'])
            
            # 构建合并后的结果
            combined_results = {
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'rps5_threshold': rps5_threshold,
                    'prediction_days': prediction_days,
                    'max_stocks': max_stocks,
                    'strategy_type': 'combined_option1_option2'
                },
                'step1_candidates': {
                    'count': len(combined_candidates),
                    'stocks': combined_candidates
                },
                'step2_strategy_selection': combined_strategy_results,
                'step3_ml_predictions': combined_ml_predictions,
                'option1_results': {
                    'ml_predictions_count': len(results1.get('step3_ml_predictions', {})),
                    'candidates_count': results1.get('step1_candidates', {}).get('count', 0)
                },
                'option2_results': {
                    'ml_predictions_count': len(results2.get('step3_ml_predictions', {})),
                    'candidates_count': results2.get('step1_candidates', {}).get('count', 0)
                },
                'summary': {
                    'total_loaded_stocks': len(self.stock_data),
                    'total_candidates': len(combined_candidates),
                    'total_strategy_selected': sum(len(stocks) for stocks in combined_strategy_results.values()) if combined_strategy_results else 0,
                    'total_ml_predicted': len(combined_ml_predictions),
                    'option1_ml_count': len(results1.get('step3_ml_predictions', {})),
                    'option2_ml_count': len(results2.get('step3_ml_predictions', {})),
                    'unique_ml_count': len(combined_ml_predictions),
                    'execution_time': (datetime.now() - start_time).total_seconds()
                }
            }
            
            # 保存合并结果
            self.save_results(combined_results, "combined_selection_results.json")
            
            # 打印合并报告
            self._print_combined_report(combined_results)
            
            logging.info(f"🎉 合并策略选股流程完成，总耗时: {combined_results['summary']['execution_time']:.2f}秒")
            
            return combined_results
            
        except Exception as e:
            logging.error(f"合并策略选股流程执行失败: {e}")
            raise
    
    def _print_combined_report(self, results: Dict[str, Any]):
        """
        打印合并策略选股报告
        """
        print("\n" + "="*80)
        print("📊 合并策略选股报告 (选项1 + 选项2)")
        print("="*80)
        
        summary = results.get('summary', {})
        option1 = results.get('option1_results', {})
        option2 = results.get('option2_results', {})
        
        print(f"\n📈 执行统计:")
        print(f"  • 加载股票数量: {summary.get('total_loaded_stocks', 0)}")
        print(f"  • 合并候选股票: {summary.get('total_candidates', 0)}")
        print(f"  • 策略筛选股票: {summary.get('total_strategy_selected', 0)}")
        print(f"  • 执行时间: {summary.get('execution_time', 0):.2f}秒")
        
        print(f"\n🔍 选项对比:")
        print(f"  • 选项1 ML预测: {option1.get('ml_predictions_count', 0)} 只")
        print(f"  • 选项2 ML预测: {option2.get('ml_predictions_count', 0)} 只")
        print(f"  • 合并去重后: {summary.get('unique_ml_count', 0)} 只")
        
        # 显示前10只预测股票
        ml_predictions = results.get('step3_ml_predictions', {})
        if ml_predictions:
            print(f"\n🏆 合并后前10只预测股票:")
            sorted_stocks = sorted(ml_predictions.items(), 
                                 key=lambda x: x[1].get('expected_return', 0), 
                                 reverse=True)[:10]
            
            print(f"{'序号':<4} {'股票代码':<12} {'预期收益率':<10} {'置信度':<8} {'投资建议':<8}")
            print("-" * 50)
            
            for i, (stock_code, prediction) in enumerate(sorted_stocks, 1):
                expected_return = prediction.get('expected_return', 0)
                confidence = prediction.get('confidence', 0)
                advice = "买入" if expected_return > 0.02 else "观望" if expected_return > 0 else "回避"
                
                print(f"{i:<4} {stock_code:<12} {expected_return*100:>6.3f}% {confidence:>6.3f} {advice:<8}")
        
        print("\n" + "="*80)
    
    def save_results(self, results: Dict[str, Any], filename: str = "independent_selection_results.json"):
        """
        保存选股结果到文件
        
        参数:
        results: Dict[str, Any], 选股结果
        filename: str, 保存文件名
        """
        try:
            # 备份现有结果
            self.backup_existing_results()
            
            # 创建带日期的结果目录
            from datetime import datetime
            date_str = datetime.now().strftime('%Y%m%d')
            results_dir = os.path.join("results", f"results_{date_str}")
            os.makedirs(results_dir, exist_ok=True)
            
            # 保存到日期目录中
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info(f"选股结果已保存到: {filepath}")
        except Exception as e:
            logging.error(f"保存结果失败: {e}")
    
    def calculate_investment_score(self, prediction_result: Dict[str, float]) -> float:
        """
        计算投资评分
        
        参数:
        prediction_result: Dict[str, float], 预测结果
        
        返回:
        float: 投资评分
        """
        predicted_return = prediction_result.get('predicted_return_5d', 0)
        confidence = prediction_result.get('confidence', 0)
        
        # 基础评分：预期收益率 * 置信度
        base_score = predicted_return * confidence
        
        # 风险调整：如果预期收益率为负，降低评分
        if predicted_return < 0:
            base_score *= 0.5
        
        # 置信度调整：低置信度降低评分
        if confidence < 0.3:
            base_score *= 0.7
        elif confidence > 0.7:
            base_score *= 1.2
        
        return base_score
    
    def select_best_stocks(self, batch_results: Dict[str, Dict[str, float]], top_n: int = 10) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        从批量分析结果中选择最佳股票
        
        参数:
        batch_results: Dict[str, Dict[str, float]], 批量预测结果
        top_n: int, 返回前N只股票
        
        返回:
        List[Tuple[str, float, Dict[str, float]]]: (股票代码, 评分, 预测结果)
        """
        scored_stocks = []
        
        for code, result in batch_results.items():
            if result:
                score = self.calculate_investment_score(result)
                scored_stocks.append((code, score, result))
        
        # 按评分排序
        scored_stocks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_stocks[:top_n]
    
    def print_investment_recommendations(self, best_stocks: List[Tuple[str, float, Dict[str, float]]]):
        """
        打印投资建议
        
        参数:
        best_stocks: List[Tuple[str, float, Dict[str, float]]], 最佳股票列表
        """
        print("\n" + "="*80)
        print("💎 投资组合优化建议")
        print("="*80)
        
        if not best_stocks:
            print("❌ 没有找到符合条件的投资标的")
            return
        
        print(f"\n🏆 推荐投资组合 (前{len(best_stocks)}只股票):")
        print("-" * 80)
        print(f"{'排名':<4} {'股票代码':<12} {'投资评分':<10} {'预期收益率':<12} {'置信度':<10} {'投资建议':<10}")
        print("-" * 80)
        
        for i, (code, score, result) in enumerate(best_stocks, 1):
            predicted_return = result.get('predicted_return_5d', 0)
            confidence = result.get('confidence', 0)
            
            # 生成投资建议
            if score > 1.0 and predicted_return > 3:
                recommendation = "强烈买入"
            elif score > 0.5 and predicted_return > 1:
                recommendation = "买入"
            elif score > 0 and predicted_return > 0:
                recommendation = "关注"
            else:
                recommendation = "回避"
            
            print(f"{i:<4} {code:<12} {score:<10.3f} {predicted_return:<12.3f}% {confidence:<10.3f} {recommendation:<10}")
        
        # 投资组合统计
        total_stocks = len(best_stocks)
        avg_return = sum(result['predicted_return_5d'] for _, _, result in best_stocks) / total_stocks
        avg_confidence = sum(result['confidence'] for _, _, result in best_stocks) / total_stocks
        avg_score = sum(score for _, score, _ in best_stocks) / total_stocks
        
        print("-" * 80)
        print(f"📊 投资组合统计:")
        print(f"   总股票数: {total_stocks}")
        print(f"   平均预期收益率: {avg_return:.3f}%")
        print(f"   平均置信度: {avg_confidence:.3f}")
        print(f"   平均投资评分: {avg_score:.3f}")
        
        # 风险提示
        high_risk_count = sum(1 for _, _, result in best_stocks if result['confidence'] < 0.5)
        if high_risk_count > 0:
            print(f"\n⚠️  风险提示: {high_risk_count} 只股票置信度较低，请谨慎投资")
        
        print("="*80)
    
    def save_results_by_board(self, results: Dict[str, Any]):
        """
        按板块分类保存选股结果
        
        参数:
        results: Dict[str, Any], 选股结果
        """
        try:
            # 获取最终的ML预测结果
            ml_predictions = results.get('step3_ml_predictions', {})
            
            # 分类股票
            chinext_stocks = []  # 创业板
            other_stocks = []    # 其他板块
            
            for stock_code, prediction_data in ml_predictions.items():
                stock_info = {
                    'stock_code': stock_code,
                    'expected_return': prediction_data.get('predicted_return_5d', 0),
                    'confidence': prediction_data.get('confidence', 0)
                }
                
                # 创业板股票代码以sz.30开头
                if stock_code.startswith('sz.30'):
                    chinext_stocks.append(stock_info)
                else:
                    other_stocks.append(stock_info)
            
            # 按预期收益率排序
            chinext_stocks.sort(key=lambda x: x['expected_return'], reverse=True)
            other_stocks.sort(key=lambda x: x['expected_return'], reverse=True)
            
            # 创建带日期的结果目录
            from datetime import datetime
            date_str = datetime.now().strftime('%Y%m%d')
            results_dir = os.path.join("results", f"results_{date_str}")
            os.makedirs(results_dir, exist_ok=True)
            
            # 保存创业板结果
            chinext_results = {
                'timestamp': results['timestamp'],
                'board_type': '创业板',
                'total_count': len(chinext_stocks),
                'stocks': chinext_stocks
            }
            
            chinext_filepath = os.path.join(results_dir, 'chinext_selection_results.json')
            with open(chinext_filepath, 'w', encoding='utf-8') as f:
                json.dump(chinext_results, f, ensure_ascii=False, indent=2)
            
            # 保存其他板块结果
            other_results = {
                'timestamp': results['timestamp'],
                'board_type': '主板/中小板/科创板',
                'total_count': len(other_stocks),
                'stocks': other_stocks
            }
            
            other_filepath = os.path.join(results_dir, 'other_boards_selection_results.json')
            with open(other_filepath, 'w', encoding='utf-8') as f:
                json.dump(other_results, f, ensure_ascii=False, indent=2)
            
            logging.info(f"📊 按板块分类保存完成:")
            logging.info(f"   创业板股票: {len(chinext_stocks)} 只 -> {chinext_filepath}")
            logging.info(f"   其他板块股票: {len(other_stocks)} 只 -> {other_filepath}")
            
        except Exception as e:
            logging.error(f"按板块保存结果失败: {e}")
    
    def print_selection_report(self, results: Dict[str, Any]):
        """
        打印选股报告
        
        参数:
        results: Dict[str, Any], 选股结果
        """
        print("\n" + "="*80)
        print("🎯 ML股票预测系统报告")
        print("="*80)
        
        # 参数信息
        params = results['parameters']
        print(f"📋 预测参数:")
        print(f"   RPS5阈值: {params['rps5_threshold']}")
        print(f"   预测天数: {params['prediction_days']}")
        print(f"   最大股票数: {params['max_stocks']}")
        
        # 第一步结果
        step1 = results['step1_candidates']
        print(f"\n🔍 第一步 - RPS5筛选:")
        print(f"   候选股票数量: {step1['count']}")
        if step1['count'] > 0:
            stocks_list = step1['stocks'] if isinstance(step1['stocks'], list) else list(step1['stocks'])
            print(f"   前10只股票: {stocks_list[:10]}")
        
        # 第二步结果（已跳过策略筛选）
        step2 = results['step2_strategy_selection']
        if step2:  # 如果有策略筛选结果才显示
            print(f"\n📊 第二步 - 策略筛选:")
            for strategy, data in step2.items():
                print(f"   {strategy}策略: {data['count']}只股票")
                if data['count'] > 0:
                    stocks_list = data['stocks'] if isinstance(data['stocks'], list) else list(data['stocks'])
                    print(f"     股票: {stocks_list[:5]}{'...' if len(stocks_list) > 5 else ''}")
        else:
            print(f"\n📊 第二步 - 策略筛选: 已跳过（直接进行ML预测）")
        
        # 第三步结果
        step3 = results['step3_ml_predictions']
        print(f"\n🤖 第三步 - ML预测:")
        print(f"   预测股票数量: {len(step3)}")
        
        if step3:
            # 投资组合优化 - 使用模型组合使用指南中的方法
            best_stocks = self.select_best_stocks(step3, top_n=10)
            
            # 打印投资建议
            self.print_investment_recommendations(best_stocks)
            
            # 按预期收益率排序的传统显示
            sorted_predictions = sorted(step3.items(), 
                                      key=lambda x: x[1].get('predicted_return_5d', 0), 
                                      reverse=True)
            
            print(f"\n🏆 预期收益率排行榜 (前5名):")
            for i, (stock_code, pred) in enumerate(sorted_predictions[:10], 1):
                predicted_return = pred.get('predicted_return_5d', 0)
                confidence = pred.get('confidence', 0)
                print(f"     {i}. {stock_code}: 预期收益率={predicted_return:.3f}%, 置信度={confidence:.3f}")
        
        # 汇总信息
        summary = results['summary']
        print(f"\n📈 汇总统计:")
        print(f"   加载股票总数: {summary['total_loaded_stocks']}")
        print(f"   候选股票数: {summary['total_candidates']}")
        print(f"   ML预测股票: {summary['total_ml_predicted']}")
        print(f"   执行时间: {summary['execution_time']:.2f}秒")
        
        print("="*80)

def main():
    """主函数"""
    import sys
    
    print("🚀 启动ML股票预测系统")
    print("="*60)
    
    # 创建ML预测器实例
    predictor = MLStockPredictor(
        data_dir="market_data",
        rps_dir="rps_results"
    )
    
    # 显示模型选择指南
    print("\n📊 模型选择指南:")
    predictor.print_model_selection_guide()
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
        print(f"\n🎯 使用命令行参数: {choice}")
    else:
        # 提供选择菜单
        print("\n🎯 选择运行模式:")
        print("1. 完整策略选股流程 (RPS筛选 + 策略筛选 + ML预测)")
        print("2. 仅RPS + ML预测 (跳过策略筛选)")
        print("3. 仅RPS筛选 (不进行ML预测)")
        print("4. 仅策略筛选 (基于现有数据)")
        print("5. RPS + 策略筛选 (不进行ML预测)")
        print("6. 合并策略选股 (选项1 + 选项2 结果合并去重)")
        
        # 获取用户选择
        choice = input("\n请输入选择 (1-6): ").strip()
    
    try:
        
        # 根据选择执行相应的流程
        if choice == "1":
            print("\n🔄 运行完整策略选股流程...")
            results = predictor.run_complete_strategy_selection(
                rps5_threshold=80,  # RPS5 > 80
                prediction_days=5,  # 预测5天收益率
                max_stocks=6000     # 最大加载6000只股票
            )
            print("\n✅ 完整策略选股流程完成！")
            
        elif choice == "2":
            print("\n🔄 运行RPS + ML预测流程...")
            results = predictor.run_rps_only_prediction(
                rps5_threshold=80,  # RPS5 > 80
                prediction_days=5,  # 预测5天收益率
                max_stocks=6000     # 最大加载6000只股票
            )
            print("\n✅ RPS + ML预测流程完成！")
            
        elif choice == "3":
            print("\n🔄 运行仅RPS筛选流程...")
            # 加载数据
            if predictor.load_stock_data(max_stocks=6000) and predictor.load_rps_data():
                candidates = predictor.filter_by_rps5(threshold=80)
                print(f"\n✅ RPS筛选完成！找到 {len(candidates)} 只候选股票")
                for i, stock in enumerate(candidates[:10], 1):
                    print(f"  {i}. {stock}")
                if len(candidates) > 10:
                    print(f"  ... 还有 {len(candidates) - 10} 只股票")
            else:
                print("\n❌ 数据加载失败")
            return
            
        elif choice == "4":
            print("\n🔄 运行仅策略筛选流程...")
            # 加载数据
            if predictor.load_stock_data(max_stocks=6000):
                strategy_results = predictor.step2_apply_strategies()
                total_selected = sum(len(stocks) for stocks in strategy_results.values())
                print(f"\n✅ 策略筛选完成！共筛选出 {total_selected} 只股票")
                for strategy, stocks in strategy_results.items():
                    print(f"  {strategy}: {len(stocks)} 只")
            else:
                print("\n❌ 数据加载失败")
            return
            
        elif choice == "5":
            print("\n🔄 运行RPS + 策略筛选流程...")
            # 这里需要实现run_rps_strategy_selection方法
            if hasattr(predictor, 'run_rps_strategy_selection'):
                results = predictor.run_rps_strategy_selection(
                    rps5_threshold=80,
                    max_stocks=6000
                )
                print("\n✅ RPS + 策略筛选流程完成！")
            else:
                print("\n❌ 该功能暂未实现")
                return
            
        elif choice == "6":
            print("\n🔄 运行合并策略选股流程...")
            results = predictor.run_combined_strategy_selection(
                rps5_threshold=80,  # RPS5 > 80
                prediction_days=5,  # 预测5天收益率
                max_stocks=6000     # 最大加载6000只股票
            )
            print("\n✅ 合并策略选股流程完成！")
            
        else:
            print("\n🔄 运行完整策略选股流程...")
            results = predictor.run_complete_strategy_selection(
                rps5_threshold=80,  # RPS5 > 80
                prediction_days=5,  # 预测5天收益率
                max_stocks=6000     # 最大加载6000只股票
            )
            print("\n✅ 完整策略选股流程完成！")
        
        # 显示简要统计（仅对有results的选项）
        if choice in ["1", "2", "5", "6"] and 'results' in locals() and results and 'summary' in results:
            summary = results['summary']
            print(f"\n📊 执行统计:")
            print(f"  • 加载股票数量: {summary.get('total_loaded_stocks', 0)}")
            print(f"  • RPS候选股票: {summary.get('total_candidates', 0)}")
            print(f"  • 策略筛选股票: {summary.get('total_strategy_selected', 0)}")
            print(f"  • ML预测股票: {summary.get('total_ml_predicted', 0)}")
            print(f"  • 执行时间: {summary.get('execution_time', 0):.2f}秒")
            
            if choice == "6":  # 合并选项的特殊统计
                print(f"\n🔍 合并统计:")
                print(f"  • 选项1 ML预测: {summary.get('option1_ml_count', 0)} 只")
                print(f"  • 选项2 ML预测: {summary.get('option2_ml_count', 0)} 只")
                print(f"  • 合并去重后: {summary.get('unique_ml_count', 0)} 只")
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断执行")
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        logging.error(f"执行失败: {e}")

if __name__ == "__main__":
    main()