#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宽松选股策略 - 适用于IP限制情况下的股票分析
当严格策略没有结果时，使用更宽松的条件找到投资机会
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pickle
import numpy

class RelaxedStockSelector:
    """宽松选股策略类"""
    
    def __init__(self, data_dir: str = 'market_data'):
        """
        初始化宽松选股器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.stock_data = {}
        self.rps_data = None
        
    def load_data(self) -> bool:
        """加载股票数据和RPS数据"""
        print("📊 加载数据...")
        
        # 加载RPS数据
        rps_file = os.path.join(self.data_dir, 'rps_120d_optimized.pkl')
        if os.path.exists(rps_file):
            with open(rps_file, 'rb') as f:
                self.rps_data = pickle.load(f)
            print(f"✅ RPS数据加载成功: {len(self.rps_data)} 只股票")
        else:
            print("❌ 未找到RPS数据文件")
            return False
        
        # 加载股票价格数据
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        print(f"🔍 找到 {len(csv_files)} 个股票数据文件")
        
        loaded_count = 0
        for file in csv_files[:100]:  # 限制加载数量以提高速度
            stock_code = file.replace('.csv', '')
            try:
                df = pd.read_csv(os.path.join(self.data_dir, file))
                if len(df) > 30:  # 确保有足够的数据
                    # 统一列名映射
                    column_mapping = {
                        'date': '日期',
                        'open': '开盘',
                        'close': '收盘',
                        'high': '最高',
                        'low': '最低',
                        'volume': '成交量',
                        'amount': '成交额'
                    }
                    df = df.rename(columns=column_mapping)
                    df['日期'] = pd.to_datetime(df['日期'])
                    df = df.sort_values('日期')
                    self.stock_data[stock_code] = df
                    loaded_count += 1
            except Exception as e:
                continue
        
        print(f"✅ 成功加载 {loaded_count} 只股票数据")
        return loaded_count > 0
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # 移动平均线
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA10'] = df['收盘'].rolling(window=10).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        
        # 价格位置 (当前价格在近期高低点的位置)
        df['最高20'] = df['最高'].rolling(window=20).max()
        df['最低20'] = df['最低'].rolling(window=20).min()
        df['价格位置'] = (df['收盘'] - df['最低20']) / (df['最高20'] - df['最低20']) * 100
        
        # 涨跌幅
        df['涨跌幅'] = df['收盘'].pct_change() * 100
        df['5日涨跌幅'] = (df['收盘'] / df['收盘'].shift(5) - 1) * 100
        df['10日涨跌幅'] = (df['收盘'] / df['收盘'].shift(10) - 1) * 100
        
        # 成交量指标
        df['成交量MA5'] = df['成交量'].rolling(window=5).mean()
        df['量比'] = df['成交量'] / df['成交量MA5']
        
        return df
    
    def relaxed_basic_strategy(self, stock_code: str, df: pd.DataFrame) -> Dict:
        """宽松基础策略"""
        if len(df) < 20:
            return {'selected': False, 'score': 0, 'reasons': ['数据不足']}
        
        latest = df.iloc[-1]
        reasons = []
        score = 0
        
        # 条件1: 短期趋势 (宽松)
        if pd.notna(latest['MA5']) and pd.notna(latest['MA10']):
            if latest['MA5'] > latest['MA10']:
                score += 20
                reasons.append('短期趋势向上')
        
        # 条件2: 价格位置 (降低要求)
        if pd.notna(latest['价格位置']) and latest['价格位置'] > 30:
            score += 15
            reasons.append(f'价格位置较好({latest["价格位置"]:.1f}%)')
        
        # 条件3: 近期表现
        if pd.notna(latest['5日涨跌幅']) and latest['5日涨跌幅'] > -5:
            score += 15
            reasons.append(f'5日表现稳定({latest["5日涨跌幅"]:.1f}%)')
        
        # 条件4: 成交量
        if pd.notna(latest['量比']) and latest['量比'] > 0.8:
            score += 10
            reasons.append(f'成交量正常({latest["量比"]:.2f})')
        
        # 条件5: RPS (降低要求)
        current_rps = self.get_latest_rps(stock_code)
        if current_rps and current_rps > 50:
            score += 20
            reasons.append(f'RPS良好({current_rps:.1f})')
        elif current_rps and current_rps > 30:
            score += 10
            reasons.append(f'RPS一般({current_rps:.1f})')
        
        # 条件6: 价格稳定性
        if len(df) >= 10:
            recent_volatility = df['涨跌幅'].tail(10).std()
            if recent_volatility < 8:  # 10日波动率小于8%
                score += 10
                reasons.append(f'价格相对稳定(波动率{recent_volatility:.1f}%)')
        
        selected = score >= 40  # 降低选中门槛
        
        return {
            'selected': selected,
            'score': score,
            'reasons': reasons,
            'latest_price': latest['收盘'],
            'rps': current_rps,
            'price_position': latest.get('价格位置', 0),
            'volume_ratio': latest.get('量比', 0),
            'change_5d': latest.get('5日涨跌幅', 0)
        }
    
    def value_discovery_strategy(self, stock_code: str, df: pd.DataFrame) -> Dict:
        """价值发现策略 - 寻找被低估的股票"""
        if len(df) < 30:
            return {'selected': False, 'score': 0, 'reasons': ['数据不足']}
        
        latest = df.iloc[-1]
        reasons = []
        score = 0
        
        # 条件1: 价格位置较低 (寻找低位股票)
        if pd.notna(latest['价格位置']) and latest['价格位置'] < 40:
            score += 25
            reasons.append(f'价格位置较低({latest["价格位置"]:.1f}%)')
        
        # 条件2: 近期跌幅较大但开始企稳
        if pd.notna(latest['10日涨跌幅']) and latest['10日涨跌幅'] < -10:
            if pd.notna(latest['5日涨跌幅']) and latest['5日涨跌幅'] > -5:
                score += 20
                reasons.append('跌幅较大但近期企稳')
        
        # 条件3: 成交量萎缩 (可能筑底)
        if pd.notna(latest['量比']) and 0.5 < latest['量比'] < 1.2:
            score += 15
            reasons.append('成交量适中')
        
        # 条件4: RPS不能太差
        current_rps = self.get_latest_rps(stock_code)
        if current_rps and current_rps > 20:
            score += 15
            reasons.append(f'RPS不算太差({current_rps:.1f})')
        
        # 条件5: 价格相对稳定
        if len(df) >= 5:
            recent_volatility = df['涨跌幅'].tail(5).std()
            if recent_volatility < 6:
                score += 15
                reasons.append('近期波动较小')
        
        # 条件6: 均线支撑
        if pd.notna(latest['MA20']) and latest['收盘'] > latest['MA20'] * 0.95:
            score += 10
            reasons.append('接近或站上20日均线')
        
        selected = score >= 50
        
        return {
            'selected': selected,
            'score': score,
            'reasons': reasons,
            'latest_price': latest['收盘'],
            'rps': current_rps,
            'price_position': latest.get('价格位置', 0),
            'volume_ratio': latest.get('量比', 0),
            'change_10d': latest.get('10日涨跌幅', 0)
        }
    
    def momentum_recovery_strategy(self, stock_code: str, df: pd.DataFrame) -> Dict:
        """动量恢复策略 - 寻找开始反弹的股票"""
        if len(df) < 20:
            return {'selected': False, 'score': 0, 'reasons': ['数据不足']}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        reasons = []
        score = 0
        
        # 条件1: 近期开始上涨
        if pd.notna(latest['5日涨跌幅']) and latest['5日涨跌幅'] > 2:
            score += 25
            reasons.append(f'5日涨幅良好({latest["5日涨跌幅"]:.1f}%)')
        elif pd.notna(latest['5日涨跌幅']) and latest['5日涨跌幅'] > 0:
            score += 15
            reasons.append(f'5日小幅上涨({latest["5日涨跌幅"]:.1f}%)')
        
        # 条件2: 成交量配合
        if pd.notna(latest['量比']) and latest['量比'] > 1.1:
            score += 20
            reasons.append(f'成交量放大({latest["量比"]:.2f})')
        
        # 条件3: 价格突破
        if pd.notna(latest['MA5']) and pd.notna(latest['MA10']):
            if latest['MA5'] > latest['MA10'] and prev.get('MA5', 0) <= prev.get('MA10', 0):
                score += 20
                reasons.append('短期均线金叉')
        
        # 条件4: RPS改善
        current_rps = self.get_latest_rps(stock_code)
        if current_rps and current_rps > 40:
            score += 15
            reasons.append(f'RPS表现尚可({current_rps:.1f})')
        
        # 条件5: 价格位置合理
        if pd.notna(latest['价格位置']) and 20 < latest['价格位置'] < 80:
            score += 10
            reasons.append('价格位置合理')
        
        # 条件6: 连续上涨
        if latest['收盘'] > prev['收盘'] and latest['涨跌幅'] > 0:
            score += 10
            reasons.append('当日上涨')
        
        selected = score >= 45
        
        return {
            'selected': selected,
            'score': score,
            'reasons': reasons,
            'latest_price': latest['收盘'],
            'rps': current_rps,
            'price_position': latest.get('价格位置', 0),
            'volume_ratio': latest.get('量比', 0),
            'daily_change': latest.get('涨跌幅', 0)
        }
    
    def get_latest_rps(self, stock_code: str) -> float:
        """获取最新RPS值"""
        if not self.rps_data or stock_code not in self.rps_data:
            return None
        
        rps_data = self.rps_data[stock_code]
        
        # 处理不同的数据格式
        if isinstance(rps_data, pd.Series):
            if len(rps_data) > 0:
                return float(rps_data.iloc[-1])
        elif isinstance(rps_data, list):
            if len(rps_data) > 0:
                # 如果是字典列表格式 [{'date': '2025-08-05', 'rps': 41.21}, ...]
                last_item = rps_data[-1]
                if isinstance(last_item, dict) and 'rps' in last_item:
                    return float(last_item['rps'])
                else:
                    return float(last_item)
        elif isinstance(rps_data, (int, float)):
            return float(rps_data)
        
        return None
    
    def run_analysis(self) -> Dict:
        """运行宽松选股分析"""
        print("\n=== 开始宽松选股分析 ===")
        
        if not self.load_data():
            return {'error': '数据加载失败'}
        
        strategies = {
            '宽松基础策略': self.relaxed_basic_strategy,
            '价值发现策略': self.value_discovery_strategy,
            '动量恢复策略': self.momentum_recovery_strategy
        }
        
        results = {}
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\n📊 执行 {strategy_name}...")
            strategy_results = []
            
            for stock_code, df in self.stock_data.items():
                df_with_indicators = self.calculate_technical_indicators(df)
                result = strategy_func(stock_code, df_with_indicators)
                
                if result['selected']:
                    result['stock_code'] = stock_code
                    strategy_results.append(result)
            
            # 按分数排序
            strategy_results.sort(key=lambda x: x['score'], reverse=True)
            results[strategy_name] = strategy_results
            
            print(f"✅ {strategy_name} 找到 {len(strategy_results)} 只符合条件的股票")
        
        return results
    
    def print_results(self, results: Dict):
        """打印分析结果"""
        print("\n" + "="*80)
        print("🎯 宽松选股分析结果")
        print("="*80)
        
        for strategy_name, stocks in results.items():
            print(f"\n🏆 {strategy_name} - Top 10 股票:")
            
            if not stocks:
                print("   暂无符合条件的股票")
                continue
            
            for i, stock in enumerate(stocks[:10], 1):
                print(f"   {i:2d}. {stock['stock_code']} - 分数: {stock['score']:2d}")
                print(f"       价格: {stock['latest_price']:.2f}, RPS: {stock.get('rps', 'N/A')}")
                print(f"       理由: {', '.join(stock['reasons'])}")
                print()
        
        # 综合推荐
        all_stocks = {}
        for strategy_name, stocks in results.items():
            for stock in stocks:
                code = stock['stock_code']
                if code not in all_stocks:
                    all_stocks[code] = {'strategies': [], 'total_score': 0}
                all_stocks[code]['strategies'].append(strategy_name)
                all_stocks[code]['total_score'] += stock['score']
                all_stocks[code]['stock_info'] = stock
        
        # 按总分排序
        comprehensive_ranking = sorted(all_stocks.items(), 
                                     key=lambda x: x[1]['total_score'], reverse=True)
        
        print("\n🌟 综合推荐 (多策略选中):")
        for i, (code, info) in enumerate(comprehensive_ranking[:15], 1):
            if len(info['strategies']) >= 2:  # 至少被两个策略选中
                print(f"   {i:2d}. {code} - 总分: {info['total_score']}")
                print(f"       选中策略: {', '.join(info['strategies'])}")
                print(f"       价格: {info['stock_info']['latest_price']:.2f}")
                print()

def main():
    """主函数"""
    print("=== 宽松选股策略分析 ===")
    print("适用于IP限制情况下的股票筛选")
    
    selector = RelaxedStockSelector()
    results = selector.run_analysis()
    
    if 'error' in results:
        print(f"❌ 分析失败: {results['error']}")
        return
    
    selector.print_results(results)
    
    # 保存结果
    output_file = 'market_data/relaxed_analysis_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        # 转换为可序列化的格式
        serializable_results = {}
        for strategy, stocks in results.items():
            serializable_results[strategy] = []
            for stock in stocks:
                stock_copy = stock.copy()
                # 处理可能的NaN值
                for key, value in stock_copy.items():
                    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
                        stock_copy[key] = None
                    elif isinstance(value, np.ndarray):
                        stock_copy[key] = value.tolist()
                    elif hasattr(value, 'item'):  # numpy scalar
                        stock_copy[key] = value.item()
                serializable_results[strategy].append(stock_copy)
        
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 分析结果已保存到: {output_file}")
    print(f"\n⏰ 分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()