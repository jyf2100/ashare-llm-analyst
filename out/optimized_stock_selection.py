#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的股票选股分析器
基于优化下载器获取的数据进行选股分析
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OptimizedStockSelector:
    """
    优化的股票选股分析器
    提供多种选股策略和分析功能
    """
    
    def __init__(self, data_dir="market_data"):
        """
        初始化选股器
        
        参数:
        data_dir: str, 数据目录
        """
        self.data_dir = data_dir
        self.stock_data = {}
        self.rps_data = None
        self.analysis_results = {}
    
    def load_data(self, use_optimized=True):
        """
        加载股票数据和RPS数据
        
        参数:
        use_optimized: bool, 是否使用优化版本的数据
        
        返回:
        bool: 加载是否成功
        """
        print("=== 加载股票数据 ===")
        
        # 加载股票数据
        loaded_count = 0
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv') and len(file_name) == 10:  # 格式: 000001.csv
                stock_code = file_name[:-4]
                
                try:
                    file_path = os.path.join(self.data_dir, file_name)
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    if len(df) > 0 and 'close' in df.columns:
                        self.stock_data[stock_code] = df
                        loaded_count += 1
                    
                except Exception as e:
                    print(f"加载股票 {stock_code} 数据失败: {e}")
                    continue
        
        print(f"✓ 成功加载 {loaded_count} 只股票的数据")
        
        # 加载RPS数据
        rps_file = os.path.join(self.data_dir, 'rps_50d_batch.pkl' if use_optimized else 'rps_50d.pkl')
        
        if os.path.exists(rps_file):
            try:
                with open(rps_file, 'rb') as f:
                    self.rps_data = pickle.load(f)
                print(f"✓ 成功加载RPS数据，包含 {len(self.rps_data)} 只股票")
            except Exception as e:
                print(f"✗ 加载RPS数据失败: {e}")
                return False
        else:
            print(f"✗ RPS数据文件不存在: {rps_file}")
            return False
        
        return loaded_count > 0 and self.rps_data is not None
    
    def calculate_technical_indicators(self, stock_code, df):
        """
        计算技术指标
        
        参数:
        stock_code: str, 股票代码
        df: DataFrame, 股票数据
        
        返回:
        DataFrame: 包含技术指标的数据
        """
        try:
            data = df.copy()
            
            # 移动平均线
            data['ma5'] = data['close'].rolling(window=5).mean()
            data['ma10'] = data['close'].rolling(window=10).mean()
            data['ma20'] = data['close'].rolling(window=20).mean()
            data['ma60'] = data['close'].rolling(window=60).mean()
            data['ma120'] = data['close'].rolling(window=120).mean()
            data['ma250'] = data['close'].rolling(window=250).mean()
            
            # 成交量移动平均
            if 'volume' in data.columns:
                data['vol_ma5'] = data['volume'].rolling(window=5).mean()
                data['vol_ma20'] = data['volume'].rolling(window=20).mean()
                
                # 量比
                data['volume_ratio'] = data['volume'] / data['vol_ma5']
            
            # 价格位置
            data['price_position'] = (data['close'] - data['close'].rolling(window=60).min()) / \
                                   (data['close'].rolling(window=60).max() - data['close'].rolling(window=60).min())
            
            # 创新高标记
            data['new_high_20'] = data['close'] >= data['close'].rolling(window=20).max()
            data['new_high_60'] = data['close'] >= data['close'].rolling(window=60).max()
            data['new_high_250'] = data['close'] >= data['close'].rolling(window=250).max()
            
            # 站上年线天数
            data['above_ma250'] = data['close'] > data['ma250']
            data['above_ma250_days'] = data['above_ma250'].rolling(window=20).sum()
            
            # 涨跌幅
            data['pct_change'] = data['close'].pct_change()
            data['pct_change_5d'] = data['close'].pct_change(5)
            data['pct_change_20d'] = data['close'].pct_change(20)
            
            # 波动率
            data['volatility_20d'] = data['pct_change'].rolling(window=20).std()
            
            return data
            
        except Exception as e:
            print(f"计算技术指标失败 {stock_code}: {e}")
            return df
    
    def get_rps_value(self, stock_code, date):
        """
        获取指定日期的RPS值
        
        参数:
        stock_code: str, 股票代码
        date: datetime, 日期
        
        返回:
        float: RPS值或None
        """
        if self.rps_data is None or stock_code not in self.rps_data:
            return None
        
        stock_rps = self.rps_data[stock_code]
        
        # 寻找最接近的日期
        available_dates = list(stock_rps.keys())
        if not available_dates:
            return None
        
        # 转换为datetime对象进行比较
        target_date = pd.to_datetime(date)
        closest_date = None
        min_diff = float('inf')
        
        for rps_date in available_dates:
            rps_datetime = pd.to_datetime(rps_date)
            diff = abs((target_date - rps_datetime).days)
            if diff < min_diff:
                min_diff = diff
                closest_date = rps_date
        
        # 如果最接近的日期在5天内，返回RPS值
        if min_diff <= 5 and closest_date is not None:
            return stock_rps[closest_date]
        
        return None
    
    def basic_selection_strategy(self, stock_code, data, date):
        """
        基础选股策略
        
        参数:
        stock_code: str, 股票代码
        data: DataFrame, 股票数据
        date: datetime, 分析日期
        
        返回:
        dict: 策略结果
        """
        try:
            if date not in data.index:
                return {'selected': False, 'reason': '日期不存在'}
            
            row = data.loc[date]
            
            # 基础条件检查
            conditions = {
                'price_valid': not pd.isna(row['close']) and row['close'] > 0,
                'ma_trend': not pd.isna(row['ma5']) and not pd.isna(row['ma20']) and row['ma5'] > row['ma20'],
                'volume_active': 'volume' in row and not pd.isna(row.get('volume_ratio', 1)) and row.get('volume_ratio', 1) > 1.2,
                'price_position': not pd.isna(row.get('price_position', 0)) and row.get('price_position', 0) > 0.6,
                'above_ma60': not pd.isna(row.get('ma60', 0)) and row['close'] > row.get('ma60', 0)
            }
            
            # 获取RPS值
            rps_value = self.get_rps_value(stock_code, date)
            if rps_value is not None:
                conditions['rps_strong'] = rps_value > 70
            else:
                conditions['rps_strong'] = False
            
            # 判断是否选中
            selected = all(conditions.values())
            
            return {
                'selected': selected,
                'conditions': conditions,
                'rps_value': rps_value,
                'close_price': row['close'],
                'volume_ratio': row.get('volume_ratio', None)
            }
            
        except Exception as e:
            return {'selected': False, 'reason': f'计算错误: {e}'}
    
    def advanced_selection_strategy(self, stock_code, data, date):
        """
        高级选股策略
        
        参数:
        stock_code: str, 股票代码
        data: DataFrame, 股票数据
        date: datetime, 分析日期
        
        返回:
        dict: 策略结果
        """
        try:
            if date not in data.index:
                return {'selected': False, 'reason': '日期不存在'}
            
            row = data.loc[date]
            
            # 高级条件检查
            conditions = {
                'price_valid': not pd.isna(row['close']) and row['close'] > 0,
                'ma_alignment': (
                    not pd.isna(row.get('ma5', 0)) and not pd.isna(row.get('ma10', 0)) and 
                    not pd.isna(row.get('ma20', 0)) and not pd.isna(row.get('ma60', 0)) and
                    row['ma5'] > row['ma10'] > row['ma20'] > row['ma60']
                ),
                'new_high': row.get('new_high_60', False),
                'volume_surge': 'volume' in row and not pd.isna(row.get('volume_ratio', 1)) and row.get('volume_ratio', 1) > 2.0,
                'above_ma250_stable': not pd.isna(row.get('above_ma250_days', 0)) and row.get('above_ma250_days', 0) >= 15,
                'low_volatility': not pd.isna(row.get('volatility_20d', 1)) and row.get('volatility_20d', 1) < 0.05
            }
            
            # 获取RPS值
            rps_value = self.get_rps_value(stock_code, date)
            if rps_value is not None:
                conditions['rps_excellent'] = rps_value > 85
            else:
                conditions['rps_excellent'] = False
            
            # 判断是否选中
            selected = all(conditions.values())
            
            return {
                'selected': selected,
                'conditions': conditions,
                'rps_value': rps_value,
                'close_price': row['close'],
                'volume_ratio': row.get('volume_ratio', None),
                'volatility': row.get('volatility_20d', None)
            }
            
        except Exception as e:
            return {'selected': False, 'reason': f'计算错误: {e}'}
    
    def momentum_selection_strategy(self, stock_code, data, date):
        """
        动量选股策略
        
        参数:
        stock_code: str, 股票代码
        data: DataFrame, 股票数据
        date: datetime, 分析日期
        
        返回:
        dict: 策略结果
        """
        try:
            if date not in data.index:
                return {'selected': False, 'reason': '日期不存在'}
            
            row = data.loc[date]
            
            # 动量条件检查
            conditions = {
                'price_valid': not pd.isna(row['close']) and row['close'] > 0,
                'strong_momentum_5d': not pd.isna(row.get('pct_change_5d', 0)) and row.get('pct_change_5d', 0) > 0.05,
                'strong_momentum_20d': not pd.isna(row.get('pct_change_20d', 0)) and row.get('pct_change_20d', 0) > 0.15,
                'volume_confirmation': 'volume' in row and not pd.isna(row.get('volume_ratio', 1)) and row.get('volume_ratio', 1) > 1.5,
                'price_near_high': not pd.isna(row.get('price_position', 0)) and row.get('price_position', 0) > 0.8
            }
            
            # 获取RPS值
            rps_value = self.get_rps_value(stock_code, date)
            if rps_value is not None:
                conditions['rps_momentum'] = rps_value > 80
            else:
                conditions['rps_momentum'] = False
            
            # 判断是否选中
            selected = all(conditions.values())
            
            return {
                'selected': selected,
                'conditions': conditions,
                'rps_value': rps_value,
                'close_price': row['close'],
                'momentum_5d': row.get('pct_change_5d', None),
                'momentum_20d': row.get('pct_change_20d', None)
            }
            
        except Exception as e:
            return {'selected': False, 'reason': f'计算错误: {e}'}
    
    def analyze_all_stocks(self, analysis_days=30):
        """
        分析所有股票
        
        参数:
        analysis_days: int, 分析天数
        
        返回:
        dict: 分析结果
        """
        print(f"=== 开始分析所有股票（最近{analysis_days}天） ===")
        
        if not self.stock_data:
            print("✗ 没有股票数据")
            return None
        
        # 获取分析日期范围
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=analysis_days)
        
        results = {
            'basic_strategy': {},
            'advanced_strategy': {},
            'momentum_strategy': {},
            'summary': {}
        }
       # 初始化计数器
        processed_count = 0
        skipped_no_data = 0
        skipped_error = 0
        error_details = []  # 存储详细错误信息
        
        for stock_code, df in self.stock_data.items():
            try:
                # 计算技术指标
                data_with_indicators = self.calculate_technical_indicators(stock_code, df)
                
                # 获取分析期间的数据
                analysis_data = data_with_indicators[
                    (data_with_indicators.index.date >= start_date) & 
                    (data_with_indicators.index.date <= end_date)
                ]
                
                if len(analysis_data) == 0:
                    skipped_no_data += 1
                    continue
                
                # 初始化结果
                stock_results = {
                    'basic_selected_days': 0,
                    'advanced_selected_days': 0,
                    'momentum_selected_days': 0,
                    'total_analysis_days': len(analysis_data),
                    'latest_rps': None,
                    'latest_price': None
                }
                
                # 逐日分析
                for date in analysis_data.index:
                    # 基础策略
                    basic_result = self.basic_selection_strategy(stock_code, data_with_indicators, date)
                    if basic_result['selected']:
                        stock_results['basic_selected_days'] += 1
                    
                    # 高级策略
                    advanced_result = self.advanced_selection_strategy(stock_code, data_with_indicators, date)
                    if advanced_result['selected']:
                        stock_results['advanced_selected_days'] += 1
                    
                    # 动量策略
                    momentum_result = self.momentum_selection_strategy(stock_code, data_with_indicators, date)
                    if momentum_result['selected']:
                        stock_results['momentum_selected_days'] += 1
                    
                    # 记录最新数据
                    if date == analysis_data.index[-1]:
                        stock_results['latest_rps'] = basic_result.get('rps_value')
                        stock_results['latest_price'] = basic_result.get('close_price')
                
                # 计算选中率
                if stock_results['total_analysis_days'] > 0:
                    stock_results['basic_selection_rate'] = stock_results['basic_selected_days'] / stock_results['total_analysis_days'] * 100
                    stock_results['advanced_selection_rate'] = stock_results['advanced_selected_days'] / stock_results['total_analysis_days'] * 100
                    stock_results['momentum_selection_rate'] = stock_results['momentum_selected_days'] / stock_results['total_analysis_days'] * 100
                
                # 保存结果
                results['basic_strategy'][stock_code] = stock_results.copy()
                results['advanced_strategy'][stock_code] = stock_results.copy()
                results['momentum_strategy'][stock_code] = stock_results.copy()
                
                processed_count += 1
                
            except Exception as e:
                error_msg = f"分析股票 {stock_code} 失败: {str(e)}"
                print(error_msg)
                error_details.append({
                    'stock_code': stock_code,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'error_details': error_msg
                })
                skipped_error += 1
                continue
        
        print(f"✓ 成功分析 {processed_count} 只股票")
        print(f"✗ 跳过 {skipped_no_data} 只股票（最近{analysis_days}天无数据）")
        print(f"✗ 跳过 {skipped_error} 只股票（分析错误）")
        print(f"📊 总计: {len(self.stock_data)} 只股票，分析 {processed_count} 只，跳过 {skipped_no_data + skipped_error} 只")
        
        # 输出详细错误信息
        if error_details:
            print("\n=== 错误明细信息 ===")
            for i, error in enumerate(error_details, 1):
                print(f"{i}. 股票代码: {error['stock_code']}")
                print(f"   错误类型: {error['error_type']}")
                print(f"   错误信息: {error['error_message']}")
                print("---")
        
        # 将错误详情保存到结果中
        results['error_details'] = {
            'total_errors': skipped_error,
            'no_data_count': skipped_no_data,
            'error_list': error_details
        }
        
        # 生成汇总统计
        results['summary'] = self._generate_summary(results, analysis_days)
        
        # 保存分析结果
        self.analysis_results = results
        
        return results
    
    def _generate_summary(self, results, analysis_days):
        """
        生成汇总统计
        
        参数:
        results: dict, 分析结果
        analysis_days: int, 分析天数
        
        返回:
        dict: 汇总统计
        """
        summary = {
            'analysis_period': f'{analysis_days}天',
            'total_stocks': len(results['basic_strategy']),
            'strategies': {}
        }
        
        for strategy_name in ['basic_strategy', 'advanced_strategy', 'momentum_strategy']:
            strategy_data = results[strategy_name]
            
            if not strategy_data:
                continue
            
            # 统计选中率分布
            selection_rates = [data.get(f'{strategy_name.split("_")[0]}_selection_rate', 0) for data in strategy_data.values()]
            
            summary['strategies'][strategy_name] = {
                'avg_selection_rate': np.mean(selection_rates) if selection_rates else 0,
                'max_selection_rate': np.max(selection_rates) if selection_rates else 0,
                'stocks_with_selections': sum(1 for rate in selection_rates if rate > 0),
                'high_quality_stocks': sum(1 for rate in selection_rates if rate > 50)  # 选中率超过50%的股票
            }
        
        return summary
    
    def get_top_stocks(self, strategy='basic_strategy', top_n=20, min_selection_rate=30):
        """
        获取表现最好的股票
        
        参数:
        strategy: str, 策略名称
        top_n: int, 返回数量
        min_selection_rate: float, 最小选中率
        
        返回:
        list: 排序后的股票列表
        """
        if not self.analysis_results or strategy not in self.analysis_results:
            return []
        
        strategy_data = self.analysis_results[strategy]
        
        # 筛选和排序
        qualified_stocks = []
        
        for stock_code, data in strategy_data.items():
            selection_rate = data.get(f'{strategy.split("_")[0]}_selection_rate', 0)
            
            if selection_rate >= min_selection_rate:
                qualified_stocks.append({
                    'stock_code': stock_code,
                    'selection_rate': selection_rate,
                    'selected_days': data.get(f'{strategy.split("_")[0]}_selected_days', 0),
                    'latest_rps': data.get('latest_rps'),
                    'latest_price': data.get('latest_price')
                })
        
        # 按选中率排序
        qualified_stocks.sort(key=lambda x: x['selection_rate'], reverse=True)
        
        return qualified_stocks[:top_n]
    
    def print_analysis_report(self):
        """
        打印分析报告
        """
        if not self.analysis_results:
            print("✗ 没有分析结果")
            return
        
        print("\n" + "="*80)
        print("                    优化版股票选股分析报告")
        print("="*80)
        
        summary = self.analysis_results['summary']
        print(f"\n📊 分析概况:")
        print(f"   分析期间: {summary['analysis_period']}")
        print(f"   分析股票: {summary['total_stocks']} 只")
        print(f"   数据来源: 优化下载器获取的真实市场数据")
        
        # 策略表现
        print(f"\n📈 策略表现:")
        for strategy_name, strategy_stats in summary['strategies'].items():
            strategy_display = {
                'basic_strategy': '基础策略',
                'advanced_strategy': '高级策略', 
                'momentum_strategy': '动量策略'
            }.get(strategy_name, strategy_name)
            
            print(f"\n   {strategy_display}:")
            print(f"     平均选中率: {strategy_stats['avg_selection_rate']:.1f}%")
            print(f"     最高选中率: {strategy_stats['max_selection_rate']:.1f}%")
            print(f"     有选中的股票: {strategy_stats['stocks_with_selections']} 只")
            print(f"     高质量股票: {strategy_stats['high_quality_stocks']} 只 (选中率>50%)")
        
        # 各策略Top股票
        strategies = [
            ('basic_strategy', '基础策略'),
            ('advanced_strategy', '高级策略'),
            ('momentum_strategy', '动量策略')
        ]
        
        for strategy_key, strategy_name in strategies:
            print(f"\n🏆 {strategy_name} - Top 10 股票:")
            top_stocks = self.get_top_stocks(strategy_key, top_n=10, min_selection_rate=20)
            
            if top_stocks:
                print(f"   {'股票代码':<8} {'选中率':<8} {'选中天数':<8} {'最新RPS':<8} {'最新价格':<10}")
                print(f"   {'-'*50}")
                
                for stock in top_stocks:
                    rps_str = f"{stock['latest_rps']:.1f}" if stock['latest_rps'] is not None else "N/A"
                    price_str = f"{stock['latest_price']:.2f}" if stock['latest_price'] is not None else "N/A"
                    
                    print(f"   {stock['stock_code']:<8} {stock['selection_rate']:<8.1f} "
                          f"{stock['selected_days']:<8} {rps_str:<8} {price_str:<10}")
            else:
                print(f"   暂无符合条件的股票")
        
        # 策略说明
        print(f"\n📋 策略条件说明:")
        
        print(f"\n   基础策略条件:")
        print(f"     • 5日均线 > 20日均线 (短期趋势向上)")
        print(f"     • 量比 > 1.2 (成交量活跃)")
        print(f"     • 价格位置 > 60% (价格相对较高)")
        print(f"     • 收盘价 > 60日均线 (中期趋势向上)")
        print(f"     • RPS > 70 (相对强度较强)")
        
        print(f"\n   高级策略条件:")
        print(f"     • 均线多头排列 (5>10>20>60日均线)")
        print(f"     • 创60日新高")
        print(f"     • 量比 > 2.0 (成交量大幅放大)")
        print(f"     • 站上年线15天以上")
        print(f"     • 20日波动率 < 5% (相对稳定)")
        print(f"     • RPS > 85 (相对强度很强)")
        
        print(f"\n   动量策略条件:")
        print(f"     • 5日涨幅 > 5%")
        print(f"     • 20日涨幅 > 15%")
        print(f"     • 量比 > 1.5 (成交量确认)")
        print(f"     • 价格位置 > 80% (接近高点)")
        print(f"     • RPS > 80 (动量强劲)")
        
        print(f"\n💡 使用建议:")
        print(f"   • 基础策略: 适合稳健投资者，关注趋势和相对强度")
        print(f"   • 高级策略: 适合追求高质量股票的投资者")
        print(f"   • 动量策略: 适合短期交易者，关注价格动量")
        print(f"   • 建议结合多个策略进行综合判断")
        print(f"   • 注意风险控制，设置止损位")
        
        print("\n" + "="*80)

def main():
    """
    主函数
    """
    print("=== 优化版股票选股分析器 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建选股器实例
    selector = OptimizedStockSelector(data_dir="market_data")
    
    # 加载数据
    if not selector.load_data(use_optimized=True):
        print("✗ 数据加载失败")
        return
    
    # 分析所有股票
    print("\n=== 开始股票分析 ===")
    results = selector.analyze_all_stocks(analysis_days=30)
    
    if results:
        # 打印分析报告
        selector.print_analysis_report()
        
        # 保存分析结果
        output_file = os.path.join(selector.data_dir, 'analysis_results_optimized.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            # 转换datetime对象为字符串以便JSON序列化
            json_results = {}
            for key, value in results.items():
                if key not in ['summary', 'error_details'] and isinstance(value, dict):
                    json_results[key] = {}
                    for stock_code, stock_data in value.items():
                        if isinstance(stock_data, dict):
                            json_results[key][stock_code] = {
                                k: v for k, v in stock_data.items() 
                                if not isinstance(v, (pd.Timestamp, datetime))
                            }
                        else:
                            json_results[key][stock_code] = stock_data
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 分析结果已保存到: {output_file}")
    else:
        print("✗ 股票分析失败")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()