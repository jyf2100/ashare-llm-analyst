#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Stock Data Analyzer
从 market_data 目录获取日线数据并进行技术分析
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import warnings
import talib
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DailyStockDataAnalyzer:
    def __init__(self, data_dir: str = "/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/market_data"):
        self.data_dir = data_dir
        self.stock_data = {}
        self.analysis_results = {}
    
    def load_stock_data(self, stock_codes: Optional[List[str]] = None, limit: int = 50) -> Dict:
        """
        加载股票数据
        
        Args:
            stock_codes: 指定股票代码列表，如果为None则加载所有
            limit: 最大加载数量
        
        Returns:
            股票数据字典
        """
        print(f"正在从 {self.data_dir} 加载股票数据...")
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 获取所有CSV文件
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if stock_codes:
            # 过滤指定的股票代码
            target_files = []
            for code in stock_codes:
                pattern = f"{code.lower()}.csv"
                matching_files = [f for f in csv_files if f.startswith(pattern)]
                target_files.extend(matching_files)
        else:
            # 加载所有文件，但限制数量
            target_files = csv_files[:limit]
        
        self.stock_data = {}
        
        for file_name in target_files:
            try:
                file_path = os.path.join(self.data_dir, file_name)
                # 从文件名提取股票代码
                stock_code = file_name.split('.')[0].upper() + '.' + file_name.split('.')[1]
                
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 转换日期列
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                # 计算技术指标
                df = self.calculate_technical_indicators(df)
                
                self.stock_data[stock_code] = df
                print(f"已加载: {stock_code} - {len(df)} 条记录")
                
            except Exception as e:
                print(f"加载 {file_name} 时出错: {e}")
        
        print(f"成功加载 {len(self.stock_data)} 只股票数据")
        return self.stock_data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 复制数据避免修改原数据
        df = df.copy()
        
        # 移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        df['MA120'] = df['close'].rolling(window=120).mean()
        
        # 指数移动平均线
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['EMA50'] = df['close'].ewm(span=50).mean()
        
        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
        
        # 成交量指标
        df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
        df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA20']
        
        # 价格变化
        df['price_change_pct'] = df['close'].pct_change() * 100
        df['price_change_5d'] = df['close'].pct_change(5) * 100
        df['price_change_20d'] = df['close'].pct_change(20) * 100
        
        # 波动率指标
        df['ATR'] = self._calculate_atr(df)
        df['Volatility_20d'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean() * 100
        
        # 动量指标
        df['Momentum_10d'] = df['close'].pct_change(10) * 100
        df['Rate_of_Change'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # 价格位置指标
        df['Price_to_MA20_Ratio'] = df['close'] / df['MA20']
        df['Price_to_BB_Middle_Ratio'] = df['close'] / df['BB_Middle']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算平均真实波幅(ATR)"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def analyze_stock_trend(self, stock_code: str) -> Dict:
        """分析单只股票趋势"""
        if stock_code not in self.stock_data:
            return {}
        
        df = self.stock_data[stock_code]
        if len(df) < 30:  # 需要足够的数据
            return {}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        analysis = {
            'stock_code': stock_code,
            'latest_date': latest['date'],
            'close_price': latest['close'],
            'price_change': latest['close'] - prev['close'],
            'price_change_pct': ((latest['close'] - prev['close']) / prev['close']) * 100,
            'volume': latest['volume'],
            'volume_change': latest['volume'] - prev['volume'],
            'trend': self._determine_trend(df),
            'signals': self._generate_signals(df),
            'support_resistance': self._find_support_resistance(df),
            'technical_score': self._calculate_technical_score(df)
        }
        
        self.analysis_results[stock_code] = analysis
        return analysis
    
    def _determine_trend(self, df: pd.DataFrame) -> Dict:
        """确定趋势方向和多时间框架分析"""
        latest = df.iloc[-1]
        
        # 多时间框架趋势判断
        trends = {
            'very_short_term': '上升' if latest['close'] > df['close'].iloc[-3] else '下降',
            'short_term': '上升' if latest['close'] > df['close'].iloc[-5] else '下降',
            'medium_term': '上升' if latest['close'] > df['close'].iloc[-20] else '下降',
            'long_term': '上升' if latest['close'] > df['close'].iloc[-60] else '下降'
        }
        
        # MA排列分析
        ma_ranking = []
        if latest['MA5'] > latest['MA10'] > latest['MA20'] > latest['MA60']:
            ma_ranking = '完美多头排列'
        elif latest['MA5'] < latest['MA10'] < latest['MA20'] < latest['MA60']:
            ma_ranking = '完美空头排列'
        else:
            ma_ranking = '混合排列'
        
        # 趋势强度评估
        trend_strength = self._assess_trend_strength(df)
        
        # 价格位置分析
        price_position = self._analyze_price_position(df)
        
        return {
            'trend_summary': f"超短{trends['very_short_term']}/短{trends['short_term']}/中{trends['medium_term']}/长{trends['long_term']}",
            'ma_ranking': ma_ranking,
            'trend_strength': trend_strength,
            'price_position': price_position,
            'detailed_trends': trends
        }
    
    def _assess_trend_strength(self, df: pd.DataFrame) -> str:
        """评估趋势强度"""
        latest = df.iloc[-1]
        
        # 基于多个指标评估趋势强度
        strength_score = 0
        
        # MA排列加分
        if latest['MA5'] > latest['MA10'] > latest['MA20']:
            strength_score += 3
        elif latest['MA5'] < latest['MA10'] < latest['MA20']:
            strength_score -= 3
        
        # 价格在MA之上加分
        if latest['close'] > latest['MA20']:
            strength_score += 2
        if latest['close'] > latest['MA60']:
            strength_score += 3
        
        # 近期涨幅加分
        if df['price_change_5d'].iloc[-1] > 5:
            strength_score += 2
        elif df['price_change_5d'].iloc[-1] < -5:
            strength_score -= 2
        
        if strength_score >= 5:
            return '强势上升'
        elif strength_score >= 2:
            return '温和上升'
        elif strength_score <= -5:
            return '强势下降'
        elif strength_score <= -2:
            return '温和下降'
        else:
            return '震荡整理'
    
    def _analyze_price_position(self, df: pd.DataFrame) -> Dict:
        """分析价格位置"""
        latest = df.iloc[-1]
        
        return {
            'above_ma20': latest['close'] > latest['MA20'],
            'above_ma60': latest['close'] > latest['MA60'],
            'ma20_distance': ((latest['close'] - latest['MA20']) / latest['MA20']) * 100,
            'ma60_distance': ((latest['close'] - latest['MA60']) / latest['MA60']) * 100,
            'in_upper_bb_half': latest['close'] > latest['BB_Middle'],
            'bb_position': (latest['close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) * 100
        }
    
    def _generate_signals(self, df: pd.DataFrame) -> Dict:
        """生成交易信号和模式识别"""
        signals = []
        patterns = []
        warnings = []
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # MACD信号
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append({'type': 'bullish', 'signal': 'MACD金叉', 'strength': '强'})
        elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals.append({'type': 'bearish', 'signal': 'MACD死叉', 'strength': '强'})
        
        # RSI信号
        if latest['RSI'] < 30:
            signals.append({'type': 'bullish', 'signal': 'RSI超卖', 'strength': '强'})
        elif latest['RSI'] > 70:
            signals.append({'type': 'bearish', 'signal': 'RSI超买', 'strength': '强'})
        elif 30 <= latest['RSI'] <= 50:
            signals.append({'type': 'neutral', 'signal': 'RSI偏弱', 'strength': '中'})
        elif 50 < latest['RSI'] <= 70:
            signals.append({'type': 'neutral', 'signal': 'RSI偏强', 'strength': '中'})
        
        # 布林带信号
        bb_position = (latest['close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) * 100
        if bb_position < 20:
            signals.append({'type': 'bullish', 'signal': '布林带下轨超卖', 'strength': '强'})
        elif bb_position > 80:
            signals.append({'type': 'bearish', 'signal': '布林带上轨超买', 'strength': '强'})
        
        # 成交量信号
        volume_ratio = latest['volume'] / latest['Volume_MA20']
        if volume_ratio > 2.0:
            signals.append({'type': 'volume', 'signal': '巨量', 'strength': '很强'})
        elif volume_ratio > 1.5:
            signals.append({'type': 'volume', 'signal': '放量', 'strength': '强'})
        elif volume_ratio < 0.5:
            signals.append({'type': 'volume', 'signal': '缩量', 'strength': '中'})
        
        # 价格突破信号
        if latest['close'] > latest['BB_Upper']:
            signals.append({'type': 'breakout', 'signal': '突破布林带上轨', 'strength': '强'})
        elif latest['close'] < latest['BB_Lower']:
            signals.append({'type': 'breakout', 'signal': '跌破布林带下轨', 'strength': '强'})
        
        # MA交叉信号
        if latest['MA5'] > latest['MA10'] and prev['MA5'] <= prev['MA10']:
            signals.append({'type': 'bullish', 'signal': 'MA5上穿MA10', 'strength': '中'})
        elif latest['MA5'] < latest['MA10'] and prev['MA5'] >= prev['MA10']:
            signals.append({'type': 'bearish', 'signal': 'MA5下穿MA10', 'strength': '中'})
        
        # 价格模式识别
        patterns.extend(self._identify_price_patterns(df))
        
        # 风险警告
        if latest['Volatility_20d'] > 30:
            warnings.append('高波动率警告')
        if latest['ATR'] > latest['close'] * 0.03:
            warnings.append('高ATR警告')
        
        return {
            'signals': signals,
            'patterns': patterns,
            'warnings': warnings,
            'signal_count': len(signals),
            'bullish_count': len([s for s in signals if s['type'] == 'bullish']),
            'bearish_count': len([s for s in signals if s['type'] == 'bearish'])
        }
    
    def _identify_price_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """识别价格模式"""
        patterns = []
        latest = df.iloc[-1]
        
        # 简单的模式识别（可以进一步扩展）
        
        # 锤子线识别
        if self._is_hammer_candle(df.iloc[-1]):
            patterns.append({'pattern': '锤子线', 'direction': 'bullish', 'reliability': '中'})
        
        # 吞没模式
        if len(df) >= 2 and self._is_engulfing_pattern(df.iloc[-2], df.iloc[-1]):
            patterns.append({'pattern': '吞没模式', 'direction': 'bullish' if df.iloc[-1]['close'] > df.iloc[-1]['open'] else 'bearish', 'reliability': '高'})
        
        # 十字星
        if self._is_doji_candle(df.iloc[-1]):
            patterns.append({'pattern': '十字星', 'direction': 'neutral', 'reliability': '中'})
        
        return patterns
    
    def _is_hammer_candle(self, candle: pd.Series) -> bool:
        """判断是否为锤子线"""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return False
            
        # 锤子线条件：下影线至少是实体的2倍，上影线很短
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        
        return (lower_shadow >= 2 * body_size and 
                upper_shadow <= body_size * 0.3 and
                body_size / total_range <= 0.3)
    
    def _is_engulfing_pattern(self, prev_candle: pd.Series, curr_candle: pd.Series) -> bool:
        """判断是否为吞没模式"""
        prev_body = abs(prev_candle['close'] - prev_candle['open'])
        curr_body = abs(curr_candle['close'] - curr_candle['open'])
        
        # 吞没条件：当前K线完全包含前一根K线
        return (min(curr_candle['open'], curr_candle['close']) < min(prev_candle['open'], prev_candle['close']) and
                max(curr_candle['open'], curr_candle['close']) > max(prev_candle['open'], prev_candle['close']) and
                curr_body > prev_body * 1.5)
    
    def _is_doji_candle(self, candle: pd.Series) -> bool:
        """判断是否为十字星"""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return False
            
        # 十字星条件：实体很小，上下影线较长
        return (body_size / total_range <= 0.1 and
                (candle['high'] - max(candle['open'], candle['close'])) / total_range >= 0.3 and
                (min(candle['open'], candle['close']) - candle['low']) / total_range >= 0.3)
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """寻找支撑阻力位"""
        # 简单的支撑阻力识别
        recent_data = df.tail(50)
        
        support = recent_data['low'].min()
        resistance = recent_data['high'].max()
        
        return {
            'support': support,
            'resistance': resistance,
            'distance_to_support': ((df['close'].iloc[-1] - support) / support) * 100,
            'distance_to_resistance': ((resistance - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100
        }
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """计算技术评分（0-100）"""
        score = 50  # 基准分
        latest = df.iloc[-1]
        
        # 价格在MA之上加分
        if latest['close'] > latest['MA5']:
            score += 5
        if latest['close'] > latest['MA10']:
            score += 5
        if latest['close'] > latest['MA20']:
            score += 10
        
        # MACD金叉加分
        if latest['MACD'] > latest['MACD_Signal']:
            score += 10
        
        # RSI在合理区间加分
        if 30 <= latest['RSI'] <= 70:
            score += 5
        elif latest['RSI'] < 30:  # 超卖区域
            score += 10
        
        # 成交量放大加分
        if latest['volume'] > latest['Volume_MA20']:
            score += 5
        
        return min(max(score, 0), 100)
    
    def analyze_all_stocks(self) -> pd.DataFrame:
        """分析所有已加载的股票"""
        results = []
        
        for stock_code in self.stock_data.keys():
            analysis = self.analyze_stock_trend(stock_code)
            if analysis:
                results.append(analysis)
        
        return pd.DataFrame(results)
    
    def get_top_performers(self, n: int = 10, by: str = 'technical_score') -> pd.DataFrame:
        """获取表现最好的股票"""
        if not self.analysis_results:
            self.analyze_all_stocks()
        
        df = pd.DataFrame(list(self.analysis_results.values()))
        if df.empty:
            return df
        
        return df.nlargest(n, by)
    
    def plot_stock_chart(self, stock_code: str, save_path: Optional[str] = None):
        """绘制股票图表"""
        if stock_code not in self.stock_data:
            print(f"未找到股票数据: {stock_code}")
            return
        
        df = self.stock_data[stock_code]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{stock_code} 技术分析', fontsize=16)
        
        # 价格图表
        ax1.plot(df['date'], df['close'], label='收盘价', linewidth=2)
        ax1.plot(df['date'], df['MA5'], label='MA5', alpha=0.7)
        ax1.plot(df['date'], df['MA10'], label='MA10', alpha=0.7)
        ax1.plot(df['date'], df['MA20'], label='MA20', alpha=0.7)
        ax1.set_title('价格走势')
        ax1.legend()
        ax1.grid(True)
        
        # MACD
        ax2.plot(df['date'], df['MACD'], label='MACD', linewidth=2)
        ax2.plot(df['date'], df['MACD_Signal'], label='Signal', linewidth=2)
        ax2.bar(df['date'], df['MACD_Histogram'], label='Histogram', alpha=0.3)
        ax2.set_title('MACD')
        ax2.legend()
        ax2.grid(True)
        
        # RSI
        ax3.plot(df['date'], df['RSI'], label='RSI', linewidth=2, color='purple')
        ax3.axhline(70, linestyle='--', alpha=0.3, color='red')
        ax3.axhline(30, linestyle='--', alpha=0.3, color='green')
        ax3.set_title('RSI')
        ax3.legend()
        ax3.grid(True)
        
        # 成交量
        ax4.bar(df['date'], df['volume'], alpha=0.3, label='成交量')
        ax4.plot(df['date'], df['Volume_MA5'], label='成交量MA5', color='orange')
        ax4.set_title('成交量')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
    
    def export_analysis_report(self, output_path: str):
        """导出分析报告"""
        if not self.analysis_results:
            self.analyze_all_stocks()
        
        df = pd.DataFrame(list(self.analysis_results.values()))
        
        # 按技术评分排序
        df = df.sort_values('technical_score', ascending=False)
        
        # 保存到CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"分析报告已导出至: {output_path}")

def analyze_specific_stocks(stock_codes):
    """分析指定股票代码"""
    # 创建分析器实例
    analyzer = DailyStockDataAnalyzer()
    
    try:
        # 加载指定股票数据
        analyzer.load_stock_data(stock_codes=stock_codes)
        
        if not analyzer.stock_data:
            print("未找到指定的股票数据")
            return
        
        # 分析所有已加载股票
        analysis_df = analyzer.analyze_all_stocks()
        
        if not analysis_df.empty:
            print(f"\n指定股票分析结果 (共{len(analysis_df)}只):")
            print(analysis_df[['stock_code', 'close_price', 'price_change_pct', 'technical_score', 'trend']])
            
            # 导出分析报告
            report_path = "/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/specific_stock_analysis_report.csv"
            analyzer.export_analysis_report(report_path)
            print(f"分析报告已导出至: {report_path}")
            
            # 绘制每只股票的图表
            for stock_code in analyzer.stock_data.keys():
                chart_path = f"/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/{stock_code}_analysis.png"
                analyzer.plot_stock_chart(stock_code, chart_path)
                
        else:
            print("没有可分析的数据")
            
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    # 创建分析器实例
    analyzer = DailyStockDataAnalyzer()
    
    try:
        # 加载股票数据（默认加载前50只）
        analyzer.load_stock_data(limit=50)
        
        # 分析所有股票
        analysis_df = analyzer.analyze_all_stocks()
        
        if not analysis_df.empty:
            # 显示前10只技术评分最高的股票
            top_stocks = analyzer.get_top_performers(10)
            print("\n技术评分最高的10只股票:")
            print(top_stocks[['stock_code', 'close_price', 'price_change_pct', 'technical_score', 'trend']])
            
            # 导出分析报告
            report_path = "/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/stock_analysis_report.csv"
            analyzer.export_analysis_report(report_path)
            
            # 绘制第一只股票的图表
            first_stock = list(analyzer.stock_data.keys())[0]
            chart_path = f"/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/{first_stock}_analysis.png"
            analyzer.plot_stock_chart(first_stock, chart_path)
            
        else:
            print("没有可分析的数据")
            
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()