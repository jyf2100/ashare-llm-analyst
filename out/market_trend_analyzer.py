#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票行情识别系统
基于行情时间周期与级别大小分析理论

功能：
1. 识别大多头、中级、次级、技术四种行情级别
2. 分析趋势方向和强度
3. 计算趋势持续时间
4. 生成交易建议
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import Ashare as as_api
import MyTT as mt
import os
import json
from dotenv import load_dotenv


class MarketTrendAnalyzer:
    """
    市场趋势分析器
    """
    
    def __init__(self, stock_code: str):
        """
        初始化分析器
        
        Args:
            stock_code: 股票代码
        """
        self.stock_code = stock_code
        self.data_cache = {}  # 数据缓存
        
        # 行情级别定义
        self.trend_levels = {
            'major_bull': {
                'name': '大多头行情',
                'duration_range': (365, float('inf')),
                'characteristics': ['持续时间1年以上', '大级别趋势', '主要上升浪'],
                'timeframes': ['大时间框架', '主时间框架']
            },
            'intermediate': {
                'name': '中级行情',
                'duration_range': (90, 365),
                'characteristics': ['持续时间3-12个月', '中级别趋势', '次要调整浪'],
                'timeframes': ['主时间框架', '小时间框架']
            },
            'minor': {
                'name': '次级行情',
                'duration_range': (14, 90),
                'characteristics': ['持续时间2周-3个月', '小级别趋势', '短期波动'],
                'timeframes': ['小时间框架', '分钟级别']
            },
            'technical': {
                'name': '技术行情',
                'duration_range': (2, 14),
                'characteristics': ['持续时间2-20天', '技术性反弹', '超短期波动'],
                'timeframes': ['分钟级别', '秒级别']
            }
        }
    
    def fetch_stock_data(self, period: str = 'daily', count: int = 500) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            period: 数据周期 ('daily', 'weekly', 'monthly', '60min', '30min', '15min', '5min')
            count: 获取数据条数
            
        Returns:
            股票数据DataFrame
        """
        cache_key = f"{self.stock_code}_{period}_{count}"
        
        if cache_key in self.data_cache:
            cached_item = self.data_cache[cache_key]
            if isinstance(cached_item, dict) and 'data' in cached_item:
                return cached_item['data']
            else:
                return cached_item
        
        try:
            # 根据周期选择相应的API
            if period == 'daily':
                df = as_api.get_price(self.stock_code, count=count)
            elif period == 'weekly':
                df = as_api.get_price(self.stock_code, count=count, frequency='w')
            elif period == 'monthly':
                df = as_api.get_price(self.stock_code, count=count, frequency='m')
            elif period in ['60min', '30min', '15min', '5min']:
                # 分钟级数据
                freq_map = {'60min': '60', '30min': '30', '15min': '15', '5min': '5'}
                df = as_api.get_price(self.stock_code, count=count, frequency=freq_map[period])
            else:
                df = as_api.get_price(self.stock_code, count=count)

            if df is not None and not df.empty and 'close' in df.columns:
                # 计算技术指标
                df = self._calculate_technical_indicators(df)
                # 缓存优化：过期时间设置为1小时
                if cache_key in self.data_cache:
                    cache_age = datetime.now() - self.data_cache[cache_key]['timestamp']
                    if cache_age.total_seconds() > 3600:
                        del self.data_cache[cache_key]
                self.data_cache[cache_key] = {
                    'timestamp': datetime.now(),
                    'data': df
                }
                return df

            # 处理空数据或缺少必要列的情况
            if period not in ['weekly', '60min', '15min']:
                print(f"⚠️ 无法获取 {self.stock_code} 的 {period} 数据")
            return pd.DataFrame()

        except Exception as e:
            # 只对非weekly数据显示错误，weekly数据失败是常见情况
            if period != 'weekly':
                print(f"❌ 获取股票数据失败: {e}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            包含技术指标的DataFrame
        """
        try:
            if df.empty or len(df) < 30:
                return df
            
            # 计算MACD
            try:
                macd_result = mt.MACD(df['close'])
                if isinstance(macd_result, dict):
                    df['macd'] = macd_result.get('MACD', 0)
                    df['macd_signal'] = macd_result.get('SIGNAL', 0)
                    df['macd_hist'] = macd_result.get('HIST', 0)
                elif isinstance(macd_result, tuple) and len(macd_result) >= 3:
                    df['macd'] = macd_result[0]
                    df['macd_signal'] = macd_result[1]
                    df['macd_hist'] = macd_result[2]
                else:
                    df['macd'] = df['macd_signal'] = df['macd_hist'] = 0
            except:
                df['macd'] = df['macd_signal'] = df['macd_hist'] = 0
            
            # 计算RSI
            try:
                rsi_result = mt.RSI(df['close'])
                if isinstance(rsi_result, (pd.Series, np.ndarray)):
                    df['rsi'] = rsi_result
                else:
                    df['rsi'] = 50
            except:
                df['rsi'] = 50
            
            # 计算布林带
            try:
                boll_result = mt.BOLL(df['close'])
                if isinstance(boll_result, dict):
                    df['boll_upper'] = boll_result.get('UPPER', df['close'])
                    df['boll_middle'] = boll_result.get('MIDDLE', df['close'])
                    df['boll_lower'] = boll_result.get('LOWER', df['close'])
                elif isinstance(boll_result, tuple) and len(boll_result) >= 3:
                    df['boll_upper'] = boll_result[0]
                    df['boll_middle'] = boll_result[1]
                    df['boll_lower'] = boll_result[2]
                else:
                    df['boll_upper'] = df['boll_middle'] = df['boll_lower'] = df['close']
            except:
                df['boll_upper'] = df['boll_middle'] = df['boll_lower'] = df['close']
            
            # 计算KDJ
            try:
                kdj_result = mt.KDJ(df['close'], df['high'], df['low'])
                if isinstance(kdj_result, dict):
                    df['kdj_k'] = kdj_result.get('K', 50)
                    df['kdj_d'] = kdj_result.get('D', 50)
                    df['kdj_j'] = kdj_result.get('J', 50)
                elif isinstance(kdj_result, tuple) and len(kdj_result) >= 3:
                    df['kdj_k'] = kdj_result[0]
                    df['kdj_d'] = kdj_result[1]
                    df['kdj_j'] = kdj_result[2]
                else:
                    df['kdj_k'] = df['kdj_d'] = df['kdj_j'] = 50
            except:
                df['kdj_k'] = df['kdj_d'] = df['kdj_j'] = 50
            
            # 计算移动平均线
            df['ma5'] = mt.MA(df['close'], 5)
            df['ma10'] = mt.MA(df['close'], 10)
            df['ma20'] = mt.MA(df['close'], 20)
            df['ma60'] = mt.MA(df['close'], 60)
            
            return df
            
        except Exception as e:
            print(f"❌ 计算技术指标失败: {e}")
            return df
    
    def identify_trend_direction(self, df: pd.DataFrame) -> Dict[str, str]:
        # 新增多时间框架验证
        weekly_df = self.fetch_stock_data('weekly', 26)
        hour_df = self.fetch_stock_data('60min', 240)
        """
        识别趋势方向
        
        Args:
            df: 包含技术指标的DataFrame
            
        Returns:
            趋势方向分析结果
        """
        if df.empty or len(df) < 20 or 'close' not in df.columns:
            return {
                'trend': 'unknown',
                'strength': 'unknown',
                'confidence': 0
            }
        
        try:
            # 获取最新数据
            latest = df.iloc[-1]
            prev_5 = df.iloc[-6:-1] if len(df) >= 6 else df.iloc[:-1]
            
            # 价格趋势分析
            price_trend_score = 0
            
            # 动态权重系数（根据波动率调整）
            volatility = df['close'].pct_change().std() * 100
            weight = 1.8 if volatility < 1.5 else 1.2 if volatility < 3 else 0.8
            
            # 分钟级趋势验证（15分钟）- 避免递归调用
            minute_df = self.fetch_stock_data('15min', 120)
            if not minute_df.empty and len(minute_df) >= 20:
                # 简化的分钟级趋势判断，避免递归
                minute_latest = minute_df.iloc[-1]
                if 'ma5' in minute_latest and 'ma10' in minute_latest:
                    if minute_latest['ma5'] > minute_latest['ma10']:
                        price_trend_score += 0.5
                    elif minute_latest['ma5'] < minute_latest['ma10']:
                        price_trend_score -= 0.5
            
            # 移动平均线排列
            if 'ma5' in latest and 'ma10' in latest and 'ma20' in latest:
                if latest['ma5'] > latest['ma10'] > latest['ma20']:
                    price_trend_score += 2  # 多头排列
                elif latest['ma5'] < latest['ma10'] < latest['ma20']:
                    price_trend_score -= 2  # 空头排列
            
            # 价格相对于均线位置
            if 'ma20' in latest:
                if latest['close'] > latest['ma20']:
                    price_trend_score += 1
                else:
                    price_trend_score -= 1
            
            # MACD趋势
            if 'macd_hist' in latest and not pd.isna(latest['macd_hist']):
                # 添加布林带收口状态检测
                boll_width = (df['boll_upper'] - df['boll_lower']) / df['boll_middle']
                is_boll_squeeze = boll_width.iloc[-1] < 0.1
                
                if latest['macd_hist'] > 0 and not is_boll_squeeze:
                    price_trend_score += 1
                else:
                    price_trend_score -= 1
            
            # RSI趋势
            if 'rsi' in latest and not pd.isna(latest['rsi']):
                if latest['rsi'] > 50:
                    price_trend_score += 0.5
                else:
                    price_trend_score -= 0.5
            
            # 判断趋势方向
            if price_trend_score >= 2:
                trend = 'upward'
            elif price_trend_score <= -2:
                trend = 'downward'
            else:
                trend = 'sideways'
            
            # 判断趋势强度
            abs_score = abs(price_trend_score)
            if abs_score >= 4:
                strength = 'strong'
            elif abs_score >= 2:
                strength = 'moderate'
            else:
                strength = 'weak'
            
            # 计算置信度
            confidence = min(abs_score / 5 * 100, 100)
            
            return {
                'trend': trend,
                'strength': strength,
                'confidence': round(confidence, 1)
            }
            
        except Exception as e:
            print(f"❌ 趋势方向识别失败: {e}")
            return {
                'trend': 'unknown',
                'strength': 'unknown',
                'confidence': 0
            }
    
    def calculate_trend_duration(self, df: pd.DataFrame, trend_type: str) -> int:
        """
        计算趋势持续时间
        
        Args:
            df: 股票数据DataFrame
            trend_type: 趋势类型 ('upward', 'downward', 'sideways')
            
        Returns:
            趋势持续天数
        """
        if df.empty or len(df) < 2:
            return 1
        
        try:
            # 简化的趋势持续时间计算
            # 基于价格相对于移动平均线的位置
            duration = 1
            
            if 'ma20' in df.columns:
                current_position = df.iloc[-1]['close'] > df.iloc[-1]['ma20']
                
                # 向前查找趋势开始点
                for i in range(len(df) - 2, -1, -1):
                    if pd.isna(df.iloc[i]['ma20']):
                        break
                    
                    position = df.iloc[i]['close'] > df.iloc[i]['ma20']
                    if position == current_position:
                        duration += 1
                    else:
                        break
            
                # 基于波动率的持续时间修正
            volatility = df['close'].pct_change().std() * 100
            volatility_factor = 1 + (volatility / 10)
            return min(int(duration * volatility_factor), len(df))
            
        except Exception as e:
            print(f"❌ 趋势持续时间计算失败: {e}")
            return 1
    
    def identify_market_level(self) -> Dict[str, any]:
        """
        识别当前市场行情级别
        
        Returns:
            行情级别识别结果
        """
        try:
            # 获取不同周期的数据
            daily_data = self.fetch_stock_data('daily', 250)  # 约1年日线数据
            weekly_data = self.fetch_stock_data('weekly', 52)  # 约1年周线数据
            
            if daily_data.empty:
                return {'level': 'unknown', 'description': '数据获取失败'}
            
            # 分析日线趋势
            daily_trend = self.identify_trend_direction(daily_data)
            daily_duration = self.calculate_trend_duration(daily_data, daily_trend['trend'])
            
            # 分析周线趋势
            if not weekly_data.empty:
                weekly_trend = self.identify_trend_direction(weekly_data)
            else:
                weekly_trend = {'trend': 'unknown', 'strength': 'unknown'}
            
            # 根据趋势持续时间和强度判断行情级别
            if daily_duration >= 365:  # 1年以上
                level = 'major_bull'
            elif daily_duration >= 90:  # 3个月以上
                level = 'intermediate'
            elif daily_duration >= 14:  # 2周以上
                level = 'minor'
            else:  # 2-20天
                level = 'technical'
            
            # 获取行情级别信息
            level_info = self.trend_levels[level]
            
            # 计算价格变化幅度
            if len(daily_data) >= daily_duration:
                start_price = daily_data.iloc[-(daily_duration+1)]['close']
                current_price = daily_data.iloc[-1]['close']
                price_change = (current_price - start_price) / start_price * 100
            else:
                price_change = 0
            
            return {
                'level': level,
                'level_name': level_info['name'],
                'duration_days': daily_duration,
                'trend_direction': daily_trend['trend'],
                'trend_strength': daily_trend['strength'],
                'price_change_pct': round(price_change, 2),
                'characteristics': level_info['characteristics'],
                'recommended_timeframes': level_info['timeframes'],
                'daily_trend': daily_trend,
                'weekly_trend': weekly_trend
            }
            
        except Exception as e:
            print(f"❌ 行情级别识别失败: {e}")
            return {
                'level': 'unknown',
                'description': f'识别失败: {e}'
            }
    
    def generate_trading_suggestions(self, market_analysis: Dict) -> List[str]:
        """
        生成交易建议
        
        Args:
            market_analysis: 市场分析结果
            
        Returns:
            交易建议列表
        """
        suggestions = []
        
        try:
            level = market_analysis.get('level', 'unknown')
            trend_direction = market_analysis.get('trend_direction', 'unknown')
            trend_strength = market_analysis.get('trend_strength', 'unknown')
            
            # 基于行情级别的建议
            if level == 'major_bull':
                suggestions.append("🔴 大多头行情：建议长期持有，关注日线级别")
                suggestions.append("📈 主升浪行情，适合趋势跟踪策略")
            elif level == 'intermediate':
                suggestions.append("🟠 中级行情：建议中期操作，关注4小时线")
                suggestions.append("⚡ 适合波段交易，注意回调风险")
            elif level == 'minor':
                suggestions.append("🟡 次级行情：建议短期操作，关注30分钟线")
                suggestions.append("⚡ 适合波段交易，注意及时止盈止损")
            elif level == 'technical':
                suggestions.append("🟢 技术行情：建议超短线操作，关注15分钟线")
                suggestions.append("💨 技术性反弹，操作要快进快出")
                suggestions.append("🎯 设置严格止损，避免追高")
            
            # 基于趋势方向的建议
            if trend_direction == 'upward':
                if trend_strength == 'strong':
                    suggestions.append("🚀 强势上涨趋势，可考虑逢低买入")
                else:
                    suggestions.append("📊 上涨趋势，谨慎追高")
            elif trend_direction == 'downward':
                if trend_strength == 'strong':
                    suggestions.append("⬇️ 强势下跌趋势，建议观望或减仓")
                else:
                    suggestions.append("📉 下跌趋势，等待反弹机会")
            else:
                suggestions.append("➡️ 横盘整理，等待方向选择")
            
            # 基于趋势强度的建议
            if trend_strength == 'weak':
                suggestions.append("⚠️ 趋势较弱，建议谨慎操作")
            
            # 通用风险提示
            suggestions.append("⚠️ 风险提示：市场有风险，投资需谨慎")
            suggestions.append("📋 建议结合基本面分析，制定完整交易计划")
            
            return suggestions
            
        except Exception as e:
            print(f"❌ 生成交易建议失败: {e}")
            return ["❌ 无法生成交易建议，请检查数据"]
    
    def analyze_market_trend(self) -> Dict[str, any]:
        """
        综合分析市场趋势
        
        Returns:
            完整的市场趋势分析结果
        """
        try:
            # 识别行情级别
            market_level = self.identify_market_level()
            
            # 生成交易建议
            trading_suggestions = self.generate_trading_suggestions(market_level)
            
            # 组合结果
            result = {
                'stock_code': self.stock_code,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_level': market_level,
                'trading_suggestions': trading_suggestions
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 市场趋势分析失败: {e}")
            return {
                'stock_code': self.stock_code,
                'error': str(e)
            }


def analyze_stock_trend(stock_code: str) -> Dict[str, any]:
    """
    分析单只股票的趋势
    
    Args:
        stock_code: 股票代码
        
    Returns:
        分析结果
    """
    analyzer = MarketTrendAnalyzer(stock_code)
    return analyzer.analyze_market_trend()


def batch_analyze_trends(stock_codes: List[str]) -> Dict[str, Dict]:
    """
    批量分析股票趋势
    
    Args:
        stock_codes: 股票代码列表
        
    Returns:
        批量分析结果
    """
    results = {}
    
    for stock_code in stock_codes:
        print(f"\n{'='*50}")
        print(f"分析股票: {stock_code}")
        print(f"{'='*50}")
        print(f"🔍 开始分析股票 {stock_code} 的市场趋势...")
        
        try:
            result = analyze_stock_trend(stock_code)
            results[stock_code] = result
            
            # 显示分析结果
            if 'error' not in result:
                market_level = result['market_level']
                print(f"\n📊 {stock_code} 行情分析结果:")
                print(f"行情级别: {market_level.get('level_name', 'unknown')}")
                print(f"趋势方向: {market_level.get('trend_direction', 'unknown')}")
                print(f"趋势强度: {market_level.get('trend_strength', 'unknown')}")
                print(f"持续时间: {market_level.get('duration_days', 0)} 天")
                print(f"价格变化: {market_level.get('price_change_pct', 0)}%")
                
                print(f"\n💡 交易建议:")
                for suggestion in result['trading_suggestions']:
                    print(f"  {suggestion}")
            else:
                print(f"❌ 分析失败: {result['error']}")
                
        except Exception as e:
            print(f"❌ 分析股票 {stock_code} 时发生错误: {e}")
            results[stock_code] = {'error': str(e)}
    
    return results


def load_stocks_from_config() -> List[str]:
    """
    从.env配置文件加载股票列表
    
    Returns:
        股票代码列表
    """
    try:
        # 加载.env文件
        load_dotenv()
        
        # 获取STOCKS_CONFIG配置
        stocks_config = os.getenv('STOCKS_CONFIG', '{}')
        
        # 解析JSON配置
        stocks_dict = json.loads(stocks_config)
        
        # 提取股票代码列表
        stock_codes = list(stocks_dict.values())
        
        print(f"📋 从配置文件加载股票:")
        for name, code in stocks_dict.items():
            print(f"  {name}: {code}")
        
        return stock_codes
        
    except Exception as e:
        print(f"⚠️ 加载股票配置失败: {e}")
        print("使用默认股票列表")
        return ['sz002487', 'sz000001', 'sh000001']


if __name__ == "__main__":
    # 从配置文件加载股票列表
    test_stocks = load_stocks_from_config()
    
    print("🚀 股票行情识别系统启动")
    print("基于行情时间周期与级别大小分析理论")
    print(f"分析股票数量: {len(test_stocks)} 只")
    print(f"{'='*60}")
    
    # 批量分析
    results = batch_analyze_trends(test_stocks)
    
    print(f"\n{'='*60}")
    print("📈 批量分析完成")
    print(f"成功分析: {len([r for r in results.values() if 'error' not in r])} 只股票")
    print(f"分析失败: {len([r for r in results.values() if 'error' in r])} 只股票")
    print(f"{'='*60}")