#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版日K线详细分析器
提供更深入的技术分析和多时间框架分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime, timedelta
import json
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class DetailedDailyAnalyzer:
    """增强版日K线详细分析器"""
    
    def __init__(self, data_dir: str = "/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/market_data"):
        self.data_dir = data_dir
        self.stock_data = {}
        self.analysis_results = {}
    
    def load_stock_data(self, stock_codes: List[str] = None, limit: int = 50) -> None:
        """加载股票数据"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if stock_codes:
            # 加载指定股票代码
            files_to_load = []
            for code in stock_codes:
                code_lower = code.lower()
                matching_files = [f for f in csv_files if f.startswith(code_lower)]
                if matching_files:
                    files_to_load.extend(matching_files)
                else:
                    print(f"警告: 未找到股票 {code} 的数据文件")
        else:
            # 加载前limit只股票
            files_to_load = csv_files[:limit]
        
        for file in files_to_load:
            try:
                file_path = os.path.join(self.data_dir, file)
                df = pd.read_csv(file_path)
                
                # 确保有足够的日期数据
                if len(df) < 60:  # 至少需要60个交易日的数据
                    print(f"跳过 {file}: 数据不足 ({len(df)} 个交易日)")
                    continue
                
                # 处理日期格式
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                # 计算技术指标
                df = self._calculate_detailed_indicators(df)
                
                stock_code = file.replace('.csv', '').upper()
                self.stock_data[stock_code] = df
                
                print(f"已加载: {stock_code} ({len(df)} 个交易日)")
                
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")
    
    def _calculate_detailed_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算详细技术指标"""
        df = df.copy()
        
        # 多周期移动平均线
        periods = [5, 10, 20, 30, 60, 120]
        for period in periods:
            df[f'MA{period}'] = df['close'].rolling(window=period).mean()
        
        # 指数移动平均线
        ema_periods = [12, 26, 50, 100]
        for period in ema_periods:
            df[f'EMA{period}'] = df['close'].ewm(span=period).mean()
        
        # MACD指标
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI指标（多周期）
        for period in [6, 14, 24]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI{period}'] = 100 - (100 / (1 + rs))
        
        # 布林带（多周期）
        for period in [20, 50]:
            df[f'BB_Middle_{period}'] = df['close'].rolling(window=period).mean()
            bb_std = df['close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + 2 * bb_std
            df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - 2 * bb_std
            df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / df[f'BB_Middle_{period}'] * 100
        
        # 成交量指标
        volume_periods = [5, 10, 20, 60]
        for period in volume_periods:
            df[f'Volume_MA{period}'] = df['volume'].rolling(window=period).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA20']
        
        # 价格变化（多周期）
        change_periods = [1, 3, 5, 10, 20, 60]
        for period in change_periods:
            df[f'price_change_{period}d'] = df['close'].pct_change(period) * 100
        
        # 波动率指标
        df['ATR'] = self._calculate_atr(df)
        for period in [20, 60]:
            df[f'Volatility_{period}d'] = df['close'].rolling(window=period).std() / df['close'].rolling(window=period).mean() * 100
        
        # 动量指标
        df['Momentum_10d'] = df['close'].pct_change(10) * 100
        df['Rate_of_Change'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # 价格位置指标
        for period in [20, 60, 120]:
            df[f'Price_to_MA{period}_Ratio'] = df['close'] / df[f'MA{period}']
        
        # 价格通道
        df['Donchian_Upper_20'] = df['high'].rolling(window=20).max()
        df['Donchian_Lower_20'] = df['low'].rolling(window=20).min()
        df['Donchian_Middle_20'] = (df['Donchian_Upper_20'] + df['Donchian_Lower_20']) / 2
        
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
    
    def analyze_stock_detailed(self, stock_code: str) -> Dict:
        """详细分析单只股票"""
        if stock_code not in self.stock_data:
            print(f"未找到股票数据: {stock_code}")
            return None
        
        df = self.stock_data[stock_code]
        latest = df.iloc[-1]
        
        # 多时间框架趋势分析
        trend_analysis = self._detailed_trend_analysis(df)
        
        # 技术信号分析
        signal_analysis = self._detailed_signal_analysis(df)
        
        # 波动率分析
        volatility_analysis = self._volatility_analysis(df)
        
        # 支撑阻力分析
        support_resistance = self._detailed_support_resistance(df)
        
        # 风险评估
        risk_assessment = self._risk_assessment(df)
        
        # 综合评分
        overall_score = self._calculate_overall_score(df, trend_analysis, signal_analysis)
        
        analysis = {
            'stock_code': stock_code,
            'latest_date': latest['date'].strftime('%Y-%m-%d') if hasattr(latest['date'], 'strftime') else str(latest['date']),
            'close_price': latest['close'],
            'open_price': latest['open'],
            'high_price': latest['high'],
            'low_price': latest['low'],
            'volume': latest['volume'],
            'price_change_1d': latest['price_change_1d'],
            'price_change_5d': latest['price_change_5d'],
            'price_change_20d': latest['price_change_20d'],
            'trend_analysis': trend_analysis,
            'signal_analysis': signal_analysis,
            'volatility_analysis': volatility_analysis,
            'support_resistance': support_resistance,
            'risk_assessment': risk_assessment,
            'overall_score': overall_score,
            'technical_indicators': {
                'MACD': latest['MACD'],
                'MACD_Signal': latest['MACD_Signal'],
                'RSI14': latest['RSI14'],
                'RSI24': latest['RSI24'],
                'BB_Width_20': latest['BB_Width_20'],
                'ATR': latest['ATR'],
                'Volatility_20d': latest['Volatility_20d']
            }
        }
        
        self.analysis_results[stock_code] = analysis
        return analysis
    
    def _detailed_trend_analysis(self, df: pd.DataFrame) -> Dict:
        """详细趋势分析"""
        latest = df.iloc[-1]
        
        # 多时间框架趋势
        timeframes = {
            'very_short_term': (1, 3),    # 1-3天
            'short_term': (3, 10),       # 3-10天
            'medium_term': (10, 30),     # 10-30天
            'long_term': (30, 60)        # 30-60天
        }
        
        trend_results = {}
        for timeframe, (start, end) in timeframes.items():
            if len(df) > end:
                current_price = latest['close']
                past_price = df['close'].iloc[-end]
                trend_direction = '上升' if current_price > past_price else '下降'
                trend_percentage = ((current_price - past_price) / past_price) * 100
                
                trend_results[timeframe] = {
                    'direction': trend_direction,
                    'percentage': trend_percentage,
                    'strength': self._assess_trend_strength(trend_percentage)
                }
        
        # MA排列分析
        ma_ranking = self._analyze_ma_ranking(df)
        
        return {
            'multi_timeframe_trends': trend_results,
            'ma_ranking': ma_ranking,
            'trend_consistency': self._assess_trend_consistency(trend_results)
        }
    
    def _analyze_ma_ranking(self, df: pd.DataFrame) -> Dict:
        """分析MA排列"""
        latest = df.iloc[-1]
        
        ma_values = {
            'MA5': latest['MA5'],
            'MA10': latest['MA10'],
            'MA20': latest['MA20'],
            'MA60': latest['MA60'],
            'MA120': latest['MA120']
        }
        
        # 检查多头排列
        if (latest['MA5'] > latest['MA10'] > latest['MA20'] > latest['MA60'] > latest['MA120']):
            ranking = '完美多头排列'
        elif (latest['MA5'] < latest['MA10'] < latest['MA20'] < latest['MA60'] < latest['MA120']):
            ranking = '完美空头排列'
        else:
            ranking = '混合排列'
        
        return {
            'ranking': ranking,
            'values': ma_values,
            'price_vs_ma': {
                'above_MA5': latest['close'] > latest['MA5'],
                'above_MA20': latest['close'] > latest['MA20'],
                'above_MA60': latest['close'] > latest['MA60']
            }
        }
    
    def _assess_trend_strength(self, percentage: float) -> str:
        """评估趋势强度"""
        abs_percentage = abs(percentage)
        
        if abs_percentage > 15:
            return '非常强'
        elif abs_percentage > 8:
            return '强'
        elif abs_percentage > 3:
            return '中等'
        elif abs_percentage > 1:
            return '弱'
        else:
            return '非常弱'
    
    def _assess_trend_consistency(self, trends: Dict) -> str:
        """评估趋势一致性"""
        directions = [trend['direction'] for trend in trends.values()]
        
        if all(d == '上升' for d in directions):
            return '完全一致上升'
        elif all(d == '下降' for d in directions):
            return '完全一致下降'
        elif directions.count('上升') > directions.count('下降'):
            return '多数上升'
        elif directions.count('下降') > directions.count('上升'):
            return '多数下降'
        else:
            return '分歧'
    
    def _detailed_signal_analysis(self, df: pd.DataFrame) -> Dict:
        """详细信号分析"""
        signals = []
        patterns = []
        divergences = []
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # MACD信号
        macd_signals = self._analyze_macd_signals(df)
        signals.extend(macd_signals)
        
        # RSI信号
        rsi_signals = self._analyze_rsi_signals(df)
        signals.extend(rsi_signals)
        
        # 布林带信号
        bb_signals = self._analyze_bollinger_signals(df)
        signals.extend(bb_signals)
        
        # 成交量信号
        volume_signals = self._analyze_volume_signals(df)
        signals.extend(volume_signals)
        
        # MA交叉信号
        ma_signals = self._analyze_ma_crossovers(df)
        signals.extend(ma_signals)
        
        # 价格模式
        price_patterns = self._identify_price_patterns(df)
        patterns.extend(price_patterns)
        
        # 背离检测
        divergences.extend(self._detect_divergences(df))
        
        return {
            'signals': signals,
            'patterns': patterns,
            'divergences': divergences,
            'signal_strength': self._calculate_signal_strength(signals),
            'bullish_count': len([s for s in signals if s.get('type') == 'bullish']),
            'bearish_count': len([s for s in signals if s.get('type') == 'bearish'])
        }
    
    def _analyze_macd_signals(self, df: pd.DataFrame) -> List[Dict]:
        """分析MACD信号"""
        signals = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # MACD金叉/死叉
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append({'type': 'bullish', 'signal': 'MACD金叉', 'strength': '强', 'indicator': 'MACD'})
        elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals.append({'type': 'bearish', 'signal': 'MACD死叉', 'strength': '强', 'indicator': 'MACD'})
        
        # MACD柱状图变化
        if latest['MACD_Histogram'] > 0 and prev['MACD_Histogram'] <= 0:
            signals.append({'type': 'bullish', 'signal': 'MACD柱转正', 'strength': '中', 'indicator': 'MACD'})
        elif latest['MACD_Histogram'] < 0 and prev['MACD_Histogram'] >= 0:
            signals.append({'type': 'bearish', 'signal': 'MACD柱转负', 'strength': '中', 'indicator': 'MACD'})
        
        return signals
    
    def _analyze_rsi_signals(self, df: pd.DataFrame) -> List[Dict]:
        """分析RSI信号"""
        signals = []
        latest = df.iloc[-1]
        
        # RSI超买超卖
        if latest['RSI14'] < 30:
            signals.append({'type': 'bullish', 'signal': 'RSI14超卖', 'strength': '强', 'indicator': 'RSI'})
        elif latest['RSI14'] > 70:
            signals.append({'type': 'bearish', 'signal': 'RSI14超买', 'strength': '强', 'indicator': 'RSI'})
        
        if latest['RSI24'] < 30:
            signals.append({'type': 'bullish', 'signal': 'RSI24超卖', 'strength': '中', 'indicator': 'RSI'})
        elif latest['RSI24'] > 70:
            signals.append({'type': 'bearish', 'signal': 'RSI24超买', 'strength': '中', 'indicator': 'RSI'})
        
        # RSI背离（简化版）
        if len(df) > 20:
            # 这里可以添加RSI背离检测逻辑
            pass
        
        return signals
    
    def _analyze_bollinger_signals(self, df: pd.DataFrame) -> List[Dict]:
        """分析布林带信号"""
        signals = []
        latest = df.iloc[-1]
        
        # 布林带位置信号
        bb_position_20 = (latest['close'] - latest['BB_Lower_20']) / (latest['BB_Upper_20'] - latest['BB_Lower_20']) * 100
        
        if bb_position_20 < 10:
            signals.append({'type': 'bullish', 'signal': '布林带下轨超卖', 'strength': '强', 'indicator': 'Bollinger'})
        elif bb_position_20 > 90:
            signals.append({'type': 'bearish', 'signal': '布林带上轨超买', 'strength': '强', 'indicator': 'Bollinger'})
        
        # 带宽收缩扩张
        if latest['BB_Width_20'] > df['BB_Width_20'].rolling(20).mean().iloc[-1] * 1.5:
            signals.append({'type': 'volatility', 'signal': '布林带扩张', 'strength': '中', 'indicator': 'Bollinger'})
        elif latest['BB_Width_20'] < df['BB_Width_20'].rolling(20).mean().iloc[-1] * 0.5:
            signals.append({'type': 'volatility', 'signal': '布林带收缩', 'strength': '中', 'indicator': 'Bollinger'})
        
        return signals
    
    def _analyze_volume_signals(self, df: pd.DataFrame) -> List[Dict]:
        """分析成交量信号"""
        signals = []
        latest = df.iloc[-1]
        
        volume_ratio = latest['volume'] / latest['Volume_MA20']
        
        if volume_ratio > 3.0:
            signals.append({'type': 'volume', 'signal': '天量', 'strength': '很强', 'indicator': 'Volume'})
        elif volume_ratio > 2.0:
            signals.append({'type': 'volume', 'signal': '巨量', 'strength': '强', 'indicator': 'Volume'})
        elif volume_ratio > 1.5:
            signals.append({'type': 'volume', 'signal': '放量', 'strength': '中', 'indicator': 'Volume'})
        elif volume_ratio < 0.3:
            signals.append({'type': 'volume', 'signal': '极度缩量', 'strength': '中', 'indicator': 'Volume'})
        elif volume_ratio < 0.5:
            signals.append({'type': 'volume', 'signal': '缩量', 'strength': '弱', 'indicator': 'Volume'})
        
        # 量价关系
        if volume_ratio > 1.5 and latest['close'] > latest['open']:
            signals.append({'type': 'bullish', 'signal': '放量上涨', 'strength': '强', 'indicator': 'Volume'})
        elif volume_ratio > 1.5 and latest['close'] < latest['open']:
            signals.append({'type': 'bearish', 'signal': '放量下跌', 'strength': '强', 'indicator': 'Volume'})
        
        return signals
    
    def _analyze_ma_crossovers(self, df: pd.DataFrame) -> List[Dict]:
        """分析MA交叉信号"""
        signals = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # MA5与MA10交叉
        if latest['MA5'] > latest['MA10'] and prev['MA5'] <= prev['MA10']:
            signals.append({'type': 'bullish', 'signal': 'MA5上穿MA10', 'strength': '中', 'indicator': 'MA'})
        elif latest['MA5'] < latest['MA10'] and prev['MA5'] >= prev['MA10']:
            signals.append({'type': 'bearish', 'signal': 'MA5下穿MA10', 'strength': '中', 'indicator': 'MA'})
        
        # MA10与MA20交叉
        if latest['MA10'] > latest['MA20'] and prev['MA10'] <= prev['MA20']:
            signals.append({'type': 'bullish', 'signal': 'MA10上穿MA20', 'strength': '强', 'indicator': 'MA'})
        elif latest['MA10'] < latest['MA20'] and prev['MA10'] >= prev['MA20']:
            signals.append({'type': 'bearish', 'signal': 'MA10下穿MA20', 'strength': '强', 'indicator': 'MA'})
        
        return signals
    
    def _identify_price_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """识别价格模式"""
        patterns = []
        
        # 使用最近60天的数据进行模式识别
        recent_data = df.tail(60)
        
        # 识别头肩顶模式
        head_shoulder_pattern = self._detect_head_shoulder_top(recent_data)
        if head_shoulder_pattern:
            patterns.append(head_shoulder_pattern)
        
        # 识别头肩底模式
        inverse_head_shoulder = self._detect_head_shoulder_bottom(recent_data)
        if inverse_head_shoulder:
            patterns.append(inverse_head_shoulder)
        
        # 识别双顶模式
        double_top = self._detect_double_top(recent_data)
        if double_top:
            patterns.append(double_top)
        
        # 识别双底模式
        double_bottom = self._detect_double_bottom(recent_data)
        if double_bottom:
            patterns.append(double_bottom)
        
        # 识别上升三角形
        ascending_triangle = self._detect_ascending_triangle(recent_data)
        if ascending_triangle:
            patterns.append(ascending_triangle)
        
        # 识别下降三角形
        descending_triangle = self._detect_descending_triangle(recent_data)
        if descending_triangle:
            patterns.append(descending_triangle)
        
        # 识别对称三角形
        symmetrical_triangle = self._detect_symmetrical_triangle(recent_data)
        if symmetrical_triangle:
            patterns.append(symmetrical_triangle)
        
        # 识别旗形整理
        flag_pattern = self._detect_flag_pattern(recent_data)
        if flag_pattern:
            patterns.append(flag_pattern)
        
        return patterns
    
    def _detect_divergences(self, df: pd.DataFrame) -> List[Dict]:
        """检测技术指标背离"""
        divergences = []
        
        # 这里可以添加背离检测逻辑
        # 例如: 价格创新高但RSI未创新高（顶背离）
        #       价格创新低但RSI未创新低（底背离）
        
        return divergences
    
    def _calculate_signal_strength(self, signals: List[Dict]) -> str:
        """计算信号强度"""
        if not signals:
            return '无信号'
        
        bullish_signals = [s for s in signals if s.get('type') == 'bullish']
        bearish_signals = [s for s in signals if s.get('type') == 'bearish']
        
        # 考虑信号强度权重
        bullish_strength = sum(3 if s.get('strength') == '很强' else 
                              2 if s.get('strength') == '强' else 
                              1 for s in bullish_signals)
        bearish_strength = sum(3 if s.get('strength') == '很强' else 
                              2 if s.get('strength') == '强' else 
                              1 for s in bearish_signals)
        
        # 只有当看多信号明显强于看空信号时才给出强烈看多
        # 同时考虑信号数量，避免单个强信号就给出强烈看多
        if bullish_strength > 0 and bearish_strength == 0:
            # 只有看多信号，没有看空信号
            if bullish_strength >= 5:  # 需要多个强信号
                return '强烈看多'
            else:
                return '偏多'
        elif bearish_strength > 0 and bullish_strength == 0:
            # 只有看空信号，没有看多信号
            if bearish_strength >= 5:  # 需要多个强信号
                return '强烈看空'
            else:
                return '偏空'
        elif bullish_strength > bearish_strength * 3:
            return '强烈看多'
        elif bearish_strength > bullish_strength * 3:
            return '强烈看空'
        elif bullish_strength > bearish_strength * 1.5:
            return '偏多'
        elif bearish_strength > bullish_strength * 1.5:
            return '偏空'
        else:
            return '中性'
    
    def _volatility_analysis(self, df: pd.DataFrame) -> Dict:
        """波动率分析"""
        latest = df.iloc[-1]
        
        return {
            'current_volatility_20d': latest['Volatility_20d'],
            'volatility_trend': self._analyze_volatility_trend(df),
            'atr_value': latest['ATR'],
            'atr_percentage': (latest['ATR'] / latest['close']) * 100,
            'volatility_regime': self._determine_volatility_regime(df)
        }
    
    def _analyze_volatility_trend(self, df: pd.DataFrame) -> str:
        """分析波动率趋势"""
        if len(df) < 30:
            return '数据不足'
        
        current_vol = df['Volatility_20d'].iloc[-1]
        prev_vol = df['Volatility_20d'].iloc[-10]
        
        if current_vol > prev_vol * 1.5:
            return '波动率上升'
        elif current_vol < prev_vol * 0.7:
            return '波动率下降'
        else:
            return '波动率稳定'
    
    def _determine_volatility_regime(self, df: pd.DataFrame) -> str:
        """确定波动率状态"""
        current_vol = df['Volatility_20d'].iloc[-1]
        
        if current_vol > 30:
            return '高波动'
        elif current_vol > 15:
            return '中高波动'
        elif current_vol > 8:
            return '中波动'
        elif current_vol > 3:
            return '低波动'
        else:
            return '极低波动'
    
    def _detailed_support_resistance(self, df: pd.DataFrame) -> Dict:
        """详细支撑阻力分析"""
        # 使用多种方法识别支撑阻力
        recent_data = df.tail(100)
        
        # 方法1: 近期高点和低点
        support1 = recent_data['low'].min()
        resistance1 = recent_data['high'].max()
        
        # 方法2: 移动平均线支撑阻力
        ma_levels = {
            'MA20': df['MA20'].iloc[-1],
            'MA60': df['MA60'].iloc[-1],
            'MA120': df['MA120'].iloc[-1]
        }
        
        # 方法3: 布林带支撑阻力
        bb_levels = {
            'BB_Upper_20': df['BB_Upper_20'].iloc[-1],
            'BB_Middle_20': df['BB_Middle_20'].iloc[-1],
            'BB_Lower_20': df['BB_Lower_20'].iloc[-1]
        }
        
        current_price = df['close'].iloc[-1]
        
        return {
            'key_levels': {
                'support': support1,
                'resistance': resistance1,
                'distance_to_support': ((current_price - support1) / support1) * 100,
                'distance_to_resistance': ((resistance1 - current_price) / current_price) * 100
            },
            'ma_levels': ma_levels,
            'bb_levels': bb_levels,
            'closest_support': self._find_closest_support(current_price, support1, ma_levels, bb_levels),
            'closest_resistance': self._find_closest_resistance(current_price, resistance1, ma_levels, bb_levels)
        }
    
    def _find_closest_support(self, current_price: float, support: float, ma_levels: Dict, bb_levels: Dict) -> Dict:
        """找到最近的支撑位"""
        all_supports = {
            'recent_low': support,
            'MA20': ma_levels['MA20'],
            'MA60': ma_levels['MA60'],
            'BB_Lower': bb_levels['BB_Lower_20']
        }
        
        # 只考虑低于当前价格的支撑位
        valid_supports = {k: v for k, v in all_supports.items() if v < current_price}
        
        if not valid_supports:
            return {'level': None, 'distance': None, 'type': None}
        
        # 找到最近的支撑位
        closest_level = min(valid_supports.items(), key=lambda x: abs(current_price - x[1]))
        
        return {
            'level': closest_level[1],
            'distance': ((current_price - closest_level[1]) / current_price) * 100,
            'type': closest_level[0]
        }
    
    def _find_closest_resistance(self, current_price: float, resistance: float, ma_levels: Dict, bb_levels: Dict) -> Dict:
        """找到最近的阻力位"""
        all_resistances = {
            'recent_high': resistance,
            'MA20': ma_levels['MA20'],
            'MA60': ma_levels['MA60'],
            'BB_Upper': bb_levels['BB_Upper_20']
        }
        
        # 只考虑高于当前价格的阻力位
        valid_resistances = {k: v for k, v in all_resistances.items() if v > current_price}
        
        if not valid_resistances:
            return {'level': None, 'distance': None, 'type': None}
        
        # 找到最近的阻力位
        closest_level = min(valid_resistances.items(), key=lambda x: abs(current_price - x[1]))
        
        return {
            'level': closest_level[1],
            'distance': ((closest_level[1] - current_price) / current_price) * 100,
            'type': closest_level[0]
        }
    
    def _risk_assessment(self, df: pd.DataFrame) -> Dict:
        """风险评估"""
        latest = df.iloc[-1]
        
        risk_factors = []
        
        # 高波动率风险
        if latest['Volatility_20d'] > 25:
            risk_factors.append({'factor': '高波动率', 'level': '高', 'description': '价格波动剧烈，风险较高'})
        
        # 高ATR风险
        if latest['ATR'] > latest['close'] * 0.04:
            risk_factors.append({'factor': '高ATR', 'level': '中', 'description': '平均真实波幅较大，日内波动风险'})
        
        # RSI极端值风险
        if latest['RSI14'] > 80 or latest['RSI14'] < 20:
            risk_factors.append({'factor': 'RSI极端值', 'level': '中', 'description': 'RSI处于极端区域，可能反转'})
        
        # 价格远离均线风险
        ma20_distance = abs((latest['close'] - latest['MA20']) / latest['MA20']) * 100
        if ma20_distance > 15:
            risk_factors.append({'factor': '远离均线', 'level': '中', 'description': '价格远离20日均线，回归风险'})
        
        # 总体风险评估
        risk_level = self._determine_overall_risk(risk_factors)
        
        return {
            'risk_factors': risk_factors,
            'risk_level': risk_level,
            'risk_score': len(risk_factors) * 10  # 简单风险评分
        }
    
    def _determine_overall_risk(self, risk_factors: List[Dict]) -> str:
        """确定总体风险等级"""
        if not risk_factors:
            return '低风险'
        
        high_risk_count = sum(1 for factor in risk_factors if factor['level'] == '高')
        medium_risk_count = sum(1 for factor in risk_factors if factor['level'] == '中')
        
        if high_risk_count >= 2:
            return '高风险'
        elif high_risk_count == 1 or medium_risk_count >= 2:
            return '中高风险'
        elif medium_risk_count == 1:
            return '中等风险'
        else:
            return '低风险'
    
    def _calculate_overall_score(self, df: pd.DataFrame, trend_analysis: Dict, signal_analysis: Dict) -> Dict:
        """计算综合评分"""
        latest = df.iloc[-1]
        
        # 趋势评分 (0-30分)
        trend_score = self._score_trend(trend_analysis)
        
        # 技术指标评分 (0-30分)
        indicator_score = self._score_indicators(latest)
        
        # 信号强度评分 (0-20分)
        signal_score = self._score_signals(signal_analysis)
        
        # 风险调整评分 (0-20分)
        risk_score = 20 - min(len(signal_analysis.get('risk_factors', [])) * 2, 20)
        
        total_score = trend_score + indicator_score + signal_score + risk_score
        
        return {
            'total_score': total_score,
            'breakdown': {
                'trend_score': trend_score,
                'indicator_score': indicator_score,
                'signal_score': signal_score,
                'risk_score': risk_score
            },
            'rating': self._get_rating(total_score)
        }
    
    def _score_trend(self, trend_analysis: Dict) -> int:
        """趋势评分"""
        score = 0
        
        # 多时间框架趋势一致性
        trends = trend_analysis.get('multi_timeframe_trends', {})
        consistency = trend_analysis.get('trend_consistency', '')
        
        if consistency == '完全一致上升':
            score += 15
        elif consistency == '多数上升':
            score += 10
        elif consistency == '分歧':
            score += 5
        
        # MA排列评分
        ma_ranking = trend_analysis.get('ma_ranking', {}).get('ranking', '')
        if ma_ranking == '完美多头排列':
            score += 15
        elif '多头' in ma_ranking:
            score += 10
        
        return min(score, 30)
    
    def _score_indicators(self, latest: pd.Series) -> int:
        """技术指标评分"""
        score = 0
        
        # MACD评分
        if latest['MACD'] > latest['MACD_Signal']:
            score += 5
        if latest['MACD_Histogram'] > 0:
            score += 3
        
        # RSI评分
        if 40 <= latest['RSI14'] <= 60:
            score += 5
        elif 30 <= latest['RSI14'] <= 70:
            score += 3
        
        # 布林带位置评分
        bb_position = (latest['close'] - latest['BB_Lower_20']) / (latest['BB_Upper_20'] - latest['BB_Lower_20']) * 100
        if 40 <= bb_position <= 60:
            score += 5
        elif 30 <= bb_position <= 70:
            score += 3
        
        return min(score, 30)
    
    def _score_signals(self, signal_analysis: Dict) -> int:
        """信号强度评分"""
        bullish_count = signal_analysis.get('bullish_count', 0)
        bearish_count = signal_analysis.get('bearish_count', 0)
        
        net_signals = bullish_count - bearish_count
        
        if net_signals >= 4:
            return 20
        elif net_signals >= 2:
            return 15
        elif net_signals >= 0:
            return 10
        else:
            return max(20 + net_signals * 2, 0)  # 负信号扣分
    
    def _get_rating(self, score: int) -> str:
        """根据评分获取评级"""
        if score >= 85:
            return 'A+ (优秀)'
        elif score >= 75:
            return 'A (很好)'
        elif score >= 65:
            return 'B+ (良好)'
        elif score >= 55:
            return 'B (一般)'
        elif score >= 45:
            return 'C+ (谨慎)'
        elif score >= 35:
            return 'C (较差)'
        else:
            return 'D (很差)'
    
    def analyze_all_stocks(self) -> List[Dict]:
        """分析所有已加载的股票"""
        results = []
        
        for stock_code in self.stock_data.keys():
            try:
                analysis = self.analyze_stock_detailed(stock_code)
                if analysis:
                    results.append(analysis)
                    print(f"已分析: {stock_code}")
            except Exception as e:
                print(f"分析股票 {stock_code} 时出错: {e}")
        
        # 按综合评分排序
        results.sort(key=lambda x: x['overall_score']['total_score'], reverse=True)
        
        return results
    
    def export_analysis_report(self, results: List[Dict], filename: str = 'detailed_stock_analysis_report.json'):
        """导出详细分析报告"""
        # 转换pandas数值类型为Python原生类型
        def convert_pandas_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d')
            elif isinstance(obj, dict):
                return {k: convert_pandas_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_pandas_types(item) for item in obj]
            else:
                return obj
        
        # 转换所有结果
        converted_results = convert_pandas_types(results)
        
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_stocks_analyzed': len(results),
            'stocks': converted_results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"详细分析报告已导出到: {filename}")
    
    def plot_detailed_chart(self, stock_code: str, save_path: str = None):
        """绘制详细分析图表"""
        if stock_code not in self.stock_data:
            print(f"未找到股票数据: {stock_code}")
            return
        
        df = self.stock_data[stock_code]
        
        # 彻底解决字体显示问题
        import matplotlib as mpl
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['pdf.fonttype'] = 42  # 确保字体可嵌入
        
        # 创建4个子图的图表
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        fig.suptitle(f'{stock_code} - 详细技术分析', fontsize=16, fontweight='bold')
        
        # 子图1: Price and Moving Averages
        ax1 = axes[0]
        ax1.plot(df['date'], df['close'], label='Close Price', linewidth=2, color='black')
        ax1.plot(df['date'], df['MA20'], label='MA20', linestyle='--', alpha=0.7)
        ax1.plot(df['date'], df['MA60'], label='MA60', linestyle='--', alpha=0.7)
        ax1.plot(df['date'], df['MA120'], label='MA120', linestyle='--', alpha=0.7)
        ax1.fill_between(df['date'], df['BB_Lower_20'], df['BB_Upper_20'], alpha=0.2, label='Bollinger Bands')
        ax1.set_title('Price and Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: MACD
        ax2 = axes[1]
        ax2.plot(df['date'], df['MACD'], label='MACD', linewidth=2)
        ax2.plot(df['date'], df['MACD_Signal'], label='Signal', linewidth=2)
        ax2.bar(df['date'], df['MACD_Histogram'], label='Histogram', alpha=0.5)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('MACD Indicator')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3: RSI
        ax3 = axes[2]
        ax3.plot(df['date'], df['RSI14'], label='RSI14', linewidth=2)
        ax3.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.axhline(50, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('RSI Indicator')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4: Volume
        ax4 = axes[3]
        ax4.bar(df['date'], df['volume'], alpha=0.7, label='Volume')
        ax4.plot(df['date'], df['Volume_MA20'], label='Volume MA20', linewidth=2, color='orange')
        ax4.set_title('Volume Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()

    def _detect_head_shoulder_top(self, df: pd.DataFrame) -> Optional[Dict]:
        """识别头肩顶模式"""
        if len(df) < 25:
            return None
        
        # 使用高低点数据
        highs = df['high'].values
        
        # 寻找局部高点
        peaks = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                peaks.append((i, highs[i]))
        
        if len(peaks) < 3:
            return None
        
        # 按价格排序，找到最高的点（头部）
        peaks.sort(key=lambda x: x[1], reverse=True)
        head_idx, head_price = peaks[0]
        
        # 找到两个肩部（价格低于头部，时间顺序正确）
        shoulders = []
        for idx, price in peaks[1:]:
            if price < head_price * 0.97:  # 肩部价格低于头部的97%
                shoulders.append((idx, price))
        
        if len(shoulders) < 2:
            return None
        
        # 检查时间顺序：左肩 -> 头部 -> 右肩
        shoulders.sort(key=lambda x: x[0])
        left_shoulder, right_shoulder = shoulders[0], shoulders[1]
        
        # 检查颈线 - 两个肩部之间的低点
        neckline_start = left_shoulder[0]
        neckline_end = right_shoulder[0]
        neckline_low = min(df['low'].iloc[neckline_start:neckline_end])
        
        # 验证模式条件
        if (left_shoulder[0] < head_idx < right_shoulder[0] and
            neckline_low < head_price * 0.9):  # 颈线明显低于头部
            
            # 计算模式完成后的价格目标
            price_target = neckline_low - (head_price - neckline_low)
            
            return {
                'pattern': '头肩顶',
                'type': 'bearish',
                'confidence': '中',
                'head_price': head_price,
                'left_shoulder_price': left_shoulder[1],
                'right_shoulder_price': right_shoulder[1],
                'neckline_price': neckline_low,
                'price_target': price_target,
                'description': '反转形态，通常出现在上升趋势末端，预示下跌'
            }
        
        return None

    def _detect_head_shoulder_bottom(self, df: pd.DataFrame) -> Optional[Dict]:
        """识别头肩底模式"""
        if len(df) < 25:
            return None
        
        # 使用低点数据
        lows = df['low'].values
        
        # 寻找局部低点
        troughs = []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                troughs.append((i, lows[i]))
        
        if len(troughs) < 3:
            return None
        
        # 按价格排序，找到最低的点（头部）
        troughs.sort(key=lambda x: x[1])
        head_idx, head_price = troughs[0]
        
        # 找到两个肩部（价格高于头部，时间顺序正确）
        shoulders = []
        for idx, price in troughs[1:]:
            if price > head_price * 1.03:  # 肩部价格高于头部的103%
                shoulders.append((idx, price))
        
        if len(shoulders) < 2:
            return None
        
        # 检查时间顺序：左肩 -> 头部 -> 右肩
        shoulders.sort(key=lambda x: x[0])
        left_shoulder, right_shoulder = shoulders[0], shoulders[1]
        
        # 检查颈线 - 两个肩部之间的高点
        neckline_start = left_shoulder[0]
        neckline_end = right_shoulder[0]
        neckline_high = max(df['high'].iloc[neckline_start:neckline_end])
        
        # 验证模式条件
        if (left_shoulder[0] < head_idx < right_shoulder[0] and
            neckline_high > head_price * 1.1):  # 颈线明显高于头部
            
            # 计算模式完成后的价格目标
            price_target = neckline_high + (neckline_high - head_price)
            
            return {
                'pattern': '头肩底',
                'type': 'bullish',
                'confidence': '中',
                'head_price': head_price,
                'left_shoulder_price': left_shoulder[1],
                'right_shoulder_price': right_shoulder[1],
                'neckline_price': neckline_high,
                'price_target': price_target,
                'description': '反转形态，通常出现在下降趋势末端，预示上涨'
            }
        
        return None

    def _detect_double_top(self, df: pd.DataFrame) -> Optional[Dict]:
        """识别双顶模式"""
        if len(df) < 15:
            return None
        
        prices = df['close'].values
        
        # 寻找两个相近的高点
        peaks = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
        
        if len(peaks) < 2:
            return None
        
        # 按价格排序
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # 检查前两个高点是否相近（价格差异在3%以内）
        top1_idx, top1_price = peaks[0]
        top2_idx, top2_price = peaks[1]
        
        if abs(top1_price - top2_price) / top1_price <= 0.03:
            # 检查时间顺序和中间的低点（颈线）
            if top1_idx < top2_idx:
                # 第一个顶在前，第二个顶在后
                neckline = min(prices[top1_idx:top2_idx])
                if neckline < top1_price * 0.95:  # 颈线明显低于顶部
                    return {
                        'pattern': '双顶',
                        'type': 'bearish',
                        'confidence': '中',
                        'top1_price': top1_price,
                        'top2_price': top2_price,
                        'neckline_price': neckline,
                        'description': '反转形态，两个相近的高点，预示下跌'
                    }
        
        return None

    def _detect_double_bottom(self, df: pd.DataFrame) -> Optional[Dict]:
        """识别双底模式"""
        if len(df) < 15:
            return None
        
        prices = df['close'].values
        
        # 寻找两个相近的低点
        troughs = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append((i, prices[i]))
        
        if len(troughs) < 2:
            return None
        
        # 按价格排序
        troughs.sort(key=lambda x: x[1])
        
        # 检查前两个低点是否相近（价格差异在3%以内）
        bottom1_idx, bottom1_price = troughs[0]
        bottom2_idx, bottom2_price = troughs[1]
        
        if abs(bottom1_price - bottom2_price) / bottom1_price <= 0.03:
            # 检查时间顺序和中间的高点（颈线）
            if bottom1_idx < bottom2_idx:
                # 第一个底在前，第二个底在后
                neckline = max(prices[bottom1_idx:bottom2_idx])
                if neckline > bottom1_price * 1.05:  # 颈线明显高于底部
                    return {
                        'pattern': '双底',
                        'type': 'bullish',
                        'confidence': '中',
                        'bottom1_price': bottom1_price,
                        'bottom2_price': bottom2_price,
                        'neckline_price': neckline,
                        'description': '反转形态，两个相近的低点，预示上涨'
                    }
        
        return None

    def _detect_ascending_triangle(self, df: pd.DataFrame) -> Optional[Dict]:
        """识别上升三角形"""
        if len(df) < 15:
            return None
        
        # 上升三角形：水平阻力线 + 上升支撑线
        highs = df['high'].values
        lows = df['low'].values
        
        # 寻找水平阻力
        recent_highs = highs[-10:]
        resistance_level = np.mean(recent_highs)
        resistance_std = np.std(recent_highs)
        
        # 检查高点是否在阻力线附近
        if resistance_std / resistance_level > 0.02:  # 波动太大，不是水平阻力
            return None
        
        # 检查低点是否逐步抬高
        recent_lows = lows[-10:]
        low_slope, _ = np.polyfit(range(len(recent_lows)), recent_lows, 1)
        
        if low_slope > 0:  # 低点上升
            return {
                'pattern': '上升三角形',
                'type': 'bullish',
                'confidence': '中',
                'resistance_level': resistance_level,
                'support_trend': '上升',
                'description': '整理形态，通常向上突破，预示继续上涨'
            }
        
        return None

    def _detect_descending_triangle(self, df: pd.DataFrame) -> Optional[Dict]:
        """识别下降三角形"""
        if len(df) < 15:
            return None
        
        # 下降三角形：水平支撑线 + 下降阻力线
        highs = df['high'].values
        lows = df['low'].values
        
        # 寻找水平支撑
        recent_lows = lows[-10:]
        support_level = np.mean(recent_lows)
        support_std = np.std(recent_lows)
        
        # 检查低点是否在支撑线附近
        if support_std / support_level > 0.02:  # 波动太大，不是水平支撑
            return None
        
        # 检查高点是否逐步降低
        recent_highs = highs[-10:]
        high_slope, _ = np.polyfit(range(len(recent_highs)), recent_highs, 1)
        
        if high_slope < 0:  # 高点下降
            return {
                'pattern': '下降三角形',
                'type': 'bearish',
                'confidence': '中',
                'support_level': support_level,
                'resistance_trend': '下降',
                'description': '整理形态，通常向下突破，预示继续下跌'
            }
        
        return None

    def _detect_symmetrical_triangle(self, df: pd.DataFrame) -> Optional[Dict]:
        """识别对称三角形"""
        if len(df) < 15:
            return None
        
        # 对称三角形：收敛的高低点
        highs = df['high'].values
        lows = df['low'].values
        
        # 计算高点和低点的趋势
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        high_slope, _ = np.polyfit(range(len(recent_highs)), recent_highs, 1)
        low_slope, _ = np.polyfit(range(len(recent_lows)), recent_lows, 1)
        
        # 对称三角形：高点下降，低点上升
        if high_slope < 0 and low_slope > 0:
            return {
                'pattern': '对称三角形',
                'type': '中性',
                'confidence': '低',
                'high_trend': '下降',
                'low_trend': '上升',
                'description': '整理形态，突破方向不确定，需要等待确认'
            }
        
        return None

    def _detect_flag_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """识别旗形整理模式"""
        if len(df) < 20:
            return None
        
        # 旗形：快速上涨/下跌后的整理
        prices = df['close'].values
        
        # 检查前期的趋势
        first_half = prices[:len(prices)//2]
        second_half = prices[len(prices)//2:]
        
        first_slope, _ = np.polyfit(range(len(first_half)), first_half, 1)
        second_slope, _ = np.polyfit(range(len(second_half)), second_half, 1)
        
        # 前期有明显趋势，后期整理
        if abs(first_slope) > abs(second_slope) * 3:
            trend = '上涨' if first_slope > 0 else '下跌'
            return {
                'pattern': f'{trend}旗形',
                'type': 'bullish' if trend == '上涨' else 'bearish',
                'confidence': '中',
                'trend_strength': abs(first_slope),
                'consolidation_strength': abs(second_slope),
                'description': f'{trend}趋势后的整理，通常延续原趋势'
            }
        
        return None


def main():
    """主函数 - 支持命令行参数指定股票代码"""
    import argparse
    
    parser = argparse.ArgumentParser(description='详细股票技术分析器')
    parser.add_argument('--stocks', '-s', nargs='+', help='指定要分析的股票代码列表（例如：SH.600000 SH.600006）')
    parser.add_argument('--file', '-f', help='包含股票代码列表的文件路径')
    parser.add_argument('--limit', '-l', type=int, default=20, help='当不指定股票时，加载的股票数量限制（默认：20）')
    parser.add_argument('--output', '-o', default='detailed_stock_analysis_report.json', help='输出报告文件名')
    
    args = parser.parse_args()
    
    analyzer = DetailedDailyAnalyzer()
    
    # 确定要分析的股票代码
    stock_codes = []
    
    if args.stocks:
        # 从命令行参数获取股票代码
        stock_codes = args.stocks
        print(f"指定分析股票: {', '.join(stock_codes)}")
    elif args.file:
        # 从文件读取股票代码
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                stock_codes = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"从文件 {args.file} 读取 {len(stock_codes)} 只股票")
        except FileNotFoundError:
            print(f"错误: 文件 {args.file} 不存在")
            return
    
    # 加载数据
    print("正在加载股票数据...")
    if stock_codes:
        analyzer.load_stock_data(stock_codes=stock_codes)
    else:
        analyzer.load_stock_data(limit=args.limit)
        print(f"加载前 {args.limit} 只股票")
    
    if not analyzer.stock_data:
        print("错误: 没有可分析的股票数据")
        return
    
    # 分析所有股票
    print(f"\n正在分析 {len(analyzer.stock_data)} 只股票...")
    results = analyzer.analyze_all_stocks()
    
    # 导出报告
    analyzer.export_analysis_report(results, args.output)
    
    # 显示分析结果
    print(f"\n=== 分析完成 ===")
    print(f"总共分析: {len(results)} 只股票")
    print(f"报告文件: {args.output}")
    
    # 显示前5只股票的详细分析
    if results:
        print("\n=== 前5只股票分析结果 ===")
        for i, result in enumerate(results[:5]):
            print(f"\n{i+1}. {result['stock_code']} - 评分: {result['overall_score']['total_score']}/100 ({result['overall_score']['rating']})")
            print(f"   最新价: {result['close_price']:.2f}")
            print(f"   趋势: {result['trend_analysis']['trend_consistency']}")
            print(f"   信号强度: {result['signal_analysis']['signal_strength']}")
            print(f"   风险等级: {result['risk_assessment']['risk_level']}")
    
        # 为第一只股票绘制详细图表
        top_stock = results[0]['stock_code']
        print(f"\n正在为 {top_stock} 绘制详细图表...")
        analyzer.plot_detailed_chart(top_stock, f"{top_stock}_detailed_analysis.png")
    else:
        print("没有分析结果")


if __name__ == "__main__":
    main()