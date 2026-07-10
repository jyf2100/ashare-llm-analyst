#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立K线形态分析器
不依赖其他模块，完全独立的K线形态识别脚本
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os
import sys


class IndependentCandlePatternAnalyzer:
    """独立的K线形态分析器，不依赖其他模块"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化分析器
        
        Args:
            df: 包含OHLCV数据的DataFrame，必须有以下列：
                'open', 'high', 'low', 'close', 'volume'
        """
        self.df = df.copy()
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # 验证数据完整性
        self._validate_data()
        
        # 计算K线实体和影线
        self.df['body'] = abs(self.df['close'] - self.df['open'])
        self.df['upper_shadow'] = self.df['high'] - np.maximum(self.df['open'], self.df['close'])
        self.df['lower_shadow'] = np.minimum(self.df['open'], self.df['close']) - self.df['low']
        self.df['total_range'] = self.df['high'] - self.df['low']
        
    def _validate_data(self):
        """验证数据完整性"""
        missing_cols = [col for col in self.required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        
        if len(self.df) < 3:
            raise ValueError("数据不足，至少需要3根K线进行分析")
    
    def detect_hammer(self) -> bool:
        """检测锤子线形态"""
        latest = self.df.iloc[-1]
        
        # 锤子线条件：小实体，长下影线，短上影线
        body_ratio = latest['body'] / latest['total_range']
        lower_shadow_ratio = latest['lower_shadow'] / latest['total_range']
        upper_shadow_ratio = latest['upper_shadow'] / latest['total_range']
        
        return (body_ratio < 0.3 and 
                lower_shadow_ratio > 0.6 and 
                upper_shadow_ratio < 0.1 and
                latest['close'] > latest['open'])  # 阳线
    
    def detect_inverted_hammer(self) -> bool:
        """检测倒锤子线形态"""
        latest = self.df.iloc[-1]
        
        # 倒锤子线条件：小实体，长上影线，短下影线
        body_ratio = latest['body'] / latest['total_range']
        upper_shadow_ratio = latest['upper_shadow'] / latest['total_range']
        lower_shadow_ratio = latest['lower_shadow'] / latest['total_range']
        
        return (body_ratio < 0.3 and 
                upper_shadow_ratio > 0.6 and 
                lower_shadow_ratio < 0.1 and
                latest['close'] > latest['open'])  # 阳线
    
    def detect_bullish_engulfing(self) -> bool:
        """检测看涨吞没形态"""
        if len(self.df) < 2:
            return False
            
        prev = self.df.iloc[-2]
        latest = self.df.iloc[-1]
        
        # 看涨吞没条件：前一根阴线，当前阳线完全吞没前一根
        return (prev['close'] < prev['open'] and  # 前一根阴线
                latest['close'] > latest['open'] and  # 当前阳线
                latest['close'] > prev['open'] and  # 收盘价高于前一根开盘价
                latest['open'] < prev['close'])   # 开盘价低于前一根收盘价
    
    def detect_bearish_engulfing(self) -> bool:
        """检测看跌吞没形态"""
        if len(self.df) < 2:
            return False
            
        prev = self.df.iloc[-2]
        latest = self.df.iloc[-1]
        
        # 看跌吞没条件：前一根阳线，当前阴线完全吞没前一根
        return (prev['close'] > prev['open'] and  # 前一根阳线
                latest['close'] < latest['open'] and  # 当前阴线
                latest['close'] < prev['open'] and  # 收盘价低于前一根开盘价
                latest['open'] > prev['close'])   # 开盘价高于前一根收盘价
    
    def detect_doji(self) -> bool:
        """检测十字星形态"""
        latest = self.df.iloc[-1]
        
        # 十字星条件：实体非常小，影线相对较长
        body_ratio = latest['body'] / latest['total_range']
        return body_ratio < 0.1 and latest['total_range'] > 0
    
    def detect_morning_star(self) -> bool:
        """检测早晨之星形态"""
        if len(self.df) < 3:
            return False
            
        prev2 = self.df.iloc[-3]
        prev = self.df.iloc[-2]
        latest = self.df.iloc[-1]
        
        # 早晨之星条件：
        # 1. 第一根大阴线
        # 2. 第二根小实体（十字星或小阳小阴）
        # 3. 第三根大阳线
        body_prev2 = abs(prev2['close'] - prev2['open'])
        body_prev = abs(prev['close'] - prev['open'])
        body_latest = abs(latest['close'] - latest['open'])
        
        return (prev2['close'] < prev2['open'] and  # 第一根阴线
                body_prev2 > body_prev * 2 and  # 第一根实体远大于第二根
                body_latest > body_prev * 2 and  # 第三根实体远大于第二根
                latest['close'] > latest['open'] and  # 第三根阳线
                latest['close'] > prev2['close'])   # 收盘价高于第一根收盘价
    
    def detect_evening_star(self) -> bool:
        """检测黄昏之星形态"""
        if len(self.df) < 3:
            return False
            
        prev2 = self.df.iloc[-3]
        prev = self.df.iloc[-2]
        latest = self.df.iloc[-1]
        
        # 黄昏之星条件：
        # 1. 第一根大阳线
        # 2. 第二根小实体（十字星或小阳小阴）
        # 3. 第三根大阴线
        body_prev2 = abs(prev2['close'] - prev2['open'])
        body_prev = abs(prev['close'] - prev['open'])
        body_latest = abs(latest['close'] - latest['open'])
        
        return (prev2['close'] > prev2['open'] and  # 第一根阳线
                body_prev2 > body_prev * 2 and  # 第一根实体远大于第二根
                body_latest > body_prev * 2 and  # 第三根实体远大于第二根
                latest['close'] < latest['open'] and  # 第三根阴线
                latest['close'] < prev2['close'])   # 收盘价低于第一根收盘价
    
    def detect_shooting_star(self) -> bool:
        """检测射击之星形态"""
        latest = self.df.iloc[-1]
        
        # 射击之星条件：小实体，长上影线，短下影线，出现在上升趋势中
        body_ratio = latest['body'] / latest['total_range']
        upper_shadow_ratio = latest['upper_shadow'] / latest['total_range']
        lower_shadow_ratio = latest['lower_shadow'] / latest['total_range']
        
        return (body_ratio < 0.3 and 
                upper_shadow_ratio > 0.6 and 
                lower_shadow_ratio < 0.1 and
                latest['close'] < latest['open'])  # 阴线
    
    def detect_hanging_man(self) -> bool:
        """检测吊颈线形态"""
        latest = self.df.iloc[-1]
        
        # 吊颈线条件：小实体，长下影线，短上影线，出现在上升趋势中
        body_ratio = latest['body'] / latest['total_range']
        lower_shadow_ratio = latest['lower_shadow'] / latest['total_range']
        upper_shadow_ratio = latest['upper_shadow'] / latest['total_range']
        
        return (body_ratio < 0.3 and 
                lower_shadow_ratio > 0.6 and 
                upper_shadow_ratio < 0.1 and
                latest['close'] < latest['open'])  # 阴线
    
    def analyze_all_patterns(self) -> List[str]:
        """分析所有K线形态，返回检测到的形态列表"""
        signals = []
        
        # 检测各种形态
        if self.detect_hammer():
            signals.append("🔨 锤子线形态 - 看涨信号")
        
        if self.detect_inverted_hammer():
            signals.append("🔨 倒锤子线形态 - 看涨信号")
        
        if self.detect_bullish_engulfing():
            signals.append("📈 看涨吞没形态 - 强烈看涨信号")
        
        if self.detect_bearish_engulfing():
            signals.append("📉 看跌吞没形态 - 强烈看跌信号")
        
        if self.detect_doji():
            signals.append("➕ 十字星形态 - 趋势可能反转")
        
        if self.detect_morning_star():
            signals.append("🌅 早晨之星形态 - 强烈看涨信号")
        
        if self.detect_evening_star():
            signals.append("🌇 黄昏之星形态 - 强烈看跌信号")
        
        if self.detect_shooting_star():
            signals.append("⭐ 射击之星形态 - 看跌信号")
        
        if self.detect_hanging_man():
            signals.append("👨‍💼 吊颈线形态 - 看跌信号")
        
        return signals
    
    def get_trading_recommendation(self) -> Dict:
        """获取交易建议"""
        patterns = self.analyze_all_patterns()
        
        # 根据形态数量和质量给出建议
        bullish_count = sum(1 for p in patterns if any(keyword in p for keyword in ['看涨', '锤子', '早晨']))
        bearish_count = sum(1 for p in patterns if any(keyword in p for keyword in ['看跌', '射击', '黄昏', '吊颈']))
        
        if bullish_count > bearish_count:
            recommendation = "买入"
            confidence = min(100, bullish_count * 20)
        elif bearish_count > bullish_count:
            recommendation = "卖出"
            confidence = min(100, bearish_count * 20)
        else:
            recommendation = "观望"
            confidence = 50
        
        return {
            'patterns': patterns,
            'recommendation': recommendation,
            'confidence': confidence,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count
        }


def load_stock_data_from_csv(file_path: str) -> pd.DataFrame:
    """从CSV文件加载股票数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # 确保有必要的列
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"CSV文件缺少必要的列: {missing_cols}")
    
    # 按日期排序
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def main():
    """主函数 - 独立运行示例"""
    print("=" * 60)
    print("独立K线形态分析器")
    print("=" * 60)
    
    # 示例：使用测试数据
    test_data = {
        'date': pd.date_range('2024-01-01', periods=10),
        'open': [100, 102, 98, 95, 96, 94, 92, 90, 88, 85],
        'high': [105, 104, 100, 97, 98, 96, 94, 92, 90, 87],
        'low': [95, 96, 94, 92, 93, 90, 88, 86, 84, 82],
        'close': [102, 98, 95, 96, 94, 92, 90, 88, 85, 87],
        'volume': [1000, 1200, 800, 900, 700, 600, 500, 400, 300, 500]
    }
    
    df = pd.DataFrame(test_data)
    
    try:
        # 创建分析器实例
        analyzer = IndependentCandlePatternAnalyzer(df)
        
        # 分析所有形态
        patterns = analyzer.analyze_all_patterns()
        
        # 获取交易建议
        recommendation = analyzer.get_trading_recommendation()
        
        print("\n📊 检测到的K线形态:")
        if patterns:
            for pattern in patterns:
                print(f"  • {pattern}")
        else:
            print("  未检测到明显的K线形态")
        
        print(f"\n💡 交易建议: {recommendation['recommendation']}")
        print(f"📈 置信度: {recommendation['confidence']}%")
        print(f"📊 看涨信号: {recommendation['bullish_signals']}")
        print(f"📉 看跌信号: {recommendation['bearish_signals']}")
        
    except Exception as e:
        print(f"❌ 分析出错: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("分析完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())