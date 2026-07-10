#!/usr/bin/env python3
"""
真实趋势分析工具
基于东方财富API获取历史财务数据，进行趋势分析
"""
import requests
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from time_fix import get_current_timestamp, get_current_iso_timestamp, get_current_time
from data_filter import filter_historical_data, get_latest_annual_data, get_latest_quarterly_data

class RealTrendAnalysis:
    def __init__(self, proxies=None):
        self.proxies = proxies or {
            'http': 'http://172.32.147.190:7890',
            'https': 'http://172.32.147.190:7890'
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'http://quote.eastmoney.com/'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.proxies.update(self.proxies)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def generate_trend_analysis(self, stock_code: str, company_name: str = "") -> Dict:
        """生成趋势分析报告"""
        start_time = time.time()
        self.logger.info(f"开始获取 {company_name}({stock_code}) 的趋势分析数据")
        
        try:
            # 获取历史财务数据
            historical_data = await self._get_historical_financial_data(stock_code)
            
            # 评估数据质量
            data_quality = self._assess_data_quality(historical_data)
            
            # 如果数据质量太低，抛出异常
            if data_quality['quality_score'] == 0.0:
                raise ValueError(f"无法获取股票代码 {stock_code} 的有效财务数据，请检查股票代码是否正确")
            
            # 计算趋势指标
            trend_metrics = self._calculate_trend_metrics(historical_data)
            
            # 生成趋势分析
            analysis_result = self._generate_analysis_summary(trend_metrics, company_name)
            
            fetch_time = time.time() - start_time
            
            return {
                **analysis_result,
                'data_fetch_time': round(fetch_time, 2),
                'data_source': 'eastmoney_api',
                'update_time': get_current_timestamp(),
                'analysis_timestamp': get_current_iso_timestamp(),
                'data_quality': data_quality
            }
            
        except Exception as e:
            self.logger.error(f"趋势分析失败: {str(e)}")
            raise
    
    async def _get_historical_financial_data(self, stock_code: str) -> Dict:
        """获取历史财务数据（近5年）"""
        try:
            # 构建API URL - 获取主要财务指标历史数据
            base_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
            
            # 参数配置
            params = {
                'sortColumns': 'NOTICE_DATE,SECURITY_CODE',
                'sortTypes': '-1,-1',
                'pageSize': '20',  # 获取近5年数据（年报+中报）
                'pageNumber': '1',
                'reportName': 'RPT_LICO_FN_CPD',
                'columns': 'SECURITY_CODE,SECURITY_NAME_ABBR,REPORTDATE,TOTAL_OPERATE_INCOME,PARENT_NETPROFIT,WEIGHTAVG_ROE,NOTICE_DATE',
                'filter': f'(SECURITY_CODE="{stock_code}")'
            }
            
            response = self.session.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('success') and data.get('result', {}).get('data'):
                financial_data = data['result']['data']
                
                # 解析数据
                parsed_data = self._parse_financial_history(financial_data)
                
                # 应用数据过滤，移除不合理的未来年份数据
                filtered_data = filter_historical_data(parsed_data)
                
                # 获取最新季报数据信息
                latest_quarterly = get_latest_quarterly_data(filtered_data)
                if latest_quarterly:
                    self.logger.info(f"最新季报数据: {latest_quarterly['year']}年{latest_quarterly['period_type']} ({latest_quarterly['date']})")
                
                # 同时获取最新年报数据信息作为参考
                latest_annual = get_latest_annual_data(filtered_data)
                if latest_annual:
                    self.logger.info(f"最新年报数据: {latest_annual['year']}年")
                
                self.logger.info(f"成功获取 {len(financial_data)} 期财务数据，过滤后保留 {len(filtered_data.get('report_dates', []))} 期")
                return filtered_data
            else:
                self.logger.warning("未获取到有效的历史财务数据")
                return {}
                
        except Exception as e:
            self.logger.error(f"获取历史财务数据失败: {str(e)}")
            return {}
    
    def _parse_financial_history(self, raw_data: List[Dict]) -> Dict:
        """解析历史财务数据"""
        parsed = {
            'revenue_history': [],
            'profit_history': [],
            'roe_history': [],
            'report_dates': [],
            'years': []
        }
        
        for item in raw_data:
            try:
                # 报告日期
                report_date = item.get('REPORTDATE', '') or item.get('NOTICE_DATE', '')
                if report_date:
                    parsed['report_dates'].append(report_date)
                    year = str(report_date)[:4]
                    if year not in parsed['years']:
                        parsed['years'].append(year)
                
                # 营业收入 (单位：万元)
                revenue = item.get('TOTAL_OPERATE_INCOME')
                if revenue is not None:
                    parsed['revenue_history'].append(float(revenue))
                
                # 净利润 (单位：万元)
                profit = item.get('PARENT_NETPROFIT')
                if profit is not None:
                    parsed['profit_history'].append(float(profit))
                
                # ROE (加权平均净资产收益率)
                roe = item.get('WEIGHTAVG_ROE')
                if roe is not None:
                    parsed['roe_history'].append(float(roe) / 100)  # 转换为小数
                    
            except (ValueError, TypeError) as e:
                self.logger.warning(f"解析财务数据项失败: {str(e)}")
                continue
        
        return parsed
    
    def _calculate_trend_metrics(self, historical_data: Dict) -> Dict:
        """计算趋势指标"""
        metrics = {}
        
        # 计算收入增长率
        revenue_growth = self._calculate_growth_rates(historical_data.get('revenue_history', []))
        metrics['revenue_growth_rates'] = revenue_growth
        
        # 计算利润增长率
        profit_growth = self._calculate_growth_rates(historical_data.get('profit_history', []))
        metrics['profit_growth_rates'] = profit_growth
        
        # ROE趋势
        roe_history = historical_data.get('roe_history', [])
        metrics['roe_trend'] = roe_history
        
        # 计算平均增长率
        metrics['avg_revenue_growth'] = self._calculate_average_growth(revenue_growth)
        metrics['avg_profit_growth'] = self._calculate_average_growth(profit_growth)
        
        # 趋势方向判断
        metrics['revenue_trend_direction'] = self._determine_trend_direction(revenue_growth)
        metrics['profit_trend_direction'] = self._determine_trend_direction(profit_growth)
        metrics['roe_trend_direction'] = self._determine_trend_direction(roe_history)
        
        return metrics
    
    def _calculate_growth_rates(self, values: List[float]) -> List[float]:
        """计算同比增长率"""
        if len(values) < 2:
            return []
        
        growth_rates = []
        # 按时间倒序排列，计算同比增长
        for i in range(1, len(values)):
            if values[i] != 0:  # 避免除零
                growth_rate = (values[i-1] - values[i]) / abs(values[i])
                growth_rates.append(growth_rate)
        
        return growth_rates
    
    def _calculate_average_growth(self, growth_rates: List[float]) -> float:
        """计算平均增长率"""
        if not growth_rates:
            return 0.0
        return sum(growth_rates) / len(growth_rates)
    
    def _determine_trend_direction(self, values: List[float]) -> str:
        """判断趋势方向"""
        if len(values) < 3:
            return "数据不足"
        
        # 计算最近3期的趋势
        recent_values = values[:3]
        
        if recent_values[0] > recent_values[1] > recent_values[2]:
            return "上升"
        elif recent_values[0] < recent_values[1] < recent_values[2]:
            return "下降"
        else:
            return "波动"
    
    def _generate_analysis_summary(self, metrics: Dict, company_name: str) -> Dict:
        """生成趋势分析总结"""
        current_year = get_current_time().year
        analysis_period = f"{current_year-4}-{current_year}年"
        
        # 生成趋势总结
        trend_summary = self._generate_trend_summary(metrics)
        
        # 判断成长阶段
        growth_stage = self._determine_growth_stage(metrics)
        
        return {
            'analysis_period': analysis_period,
            'revenue_growth_5y': metrics.get('revenue_growth_rates', [])[:5],
            'profit_growth_5y': metrics.get('profit_growth_rates', [])[:5],
            'roe_trend_5y': metrics.get('roe_trend', [])[:5],
            'debt_ratio_trend_5y': metrics.get('debt_ratio_trend', [])[:5],
            'avg_revenue_growth': round(metrics.get('avg_revenue_growth', 0), 3),
            'avg_profit_growth': round(metrics.get('avg_profit_growth', 0), 3),
            'revenue_trend_direction': metrics.get('revenue_trend_direction', '未知'),
            'profit_trend_direction': metrics.get('profit_trend_direction', '未知'),
            'roe_trend_direction': metrics.get('roe_trend_direction', '未知'),
            'trend_summary': trend_summary,
            'growth_stage': growth_stage,
            'company_name': company_name
        }
    
    def _generate_trend_summary(self, metrics: Dict) -> str:
        """生成趋势总结"""
        revenue_direction = metrics.get('revenue_trend_direction', '未知')
        profit_direction = metrics.get('profit_trend_direction', '未知')
        roe_direction = metrics.get('roe_trend_direction', '未知')
        
        avg_revenue_growth = metrics.get('avg_revenue_growth', 0)
        avg_profit_growth = metrics.get('avg_profit_growth', 0)
        
        summary_parts = []
        
        # 收入趋势
        if revenue_direction == "上升":
            summary_parts.append(f"收入呈上升趋势，平均增长率{avg_revenue_growth:.1%}")
        elif revenue_direction == "下降":
            summary_parts.append(f"收入呈下降趋势，平均下降率{abs(avg_revenue_growth):.1%}")
        else:
            summary_parts.append("收入增长波动较大")
        
        # 利润趋势
        if profit_direction == "上升":
            summary_parts.append(f"盈利能力持续改善，平均增长率{avg_profit_growth:.1%}")
        elif profit_direction == "下降":
            summary_parts.append(f"盈利能力有所下降，平均下降率{abs(avg_profit_growth):.1%}")
        else:
            summary_parts.append("盈利能力波动")
        
        # ROE趋势
        if roe_direction == "上升":
            summary_parts.append("净资产收益率呈上升趋势")
        elif roe_direction == "下降":
            summary_parts.append("净资产收益率有所下降")
        
        return "，".join(summary_parts)
    
    def _determine_growth_stage(self, metrics: Dict) -> str:
        """判断企业成长阶段"""
        avg_revenue_growth = metrics.get('avg_revenue_growth', 0)
        avg_profit_growth = metrics.get('avg_profit_growth', 0)
        
        if avg_revenue_growth > 0.2 and avg_profit_growth > 0.2:
            return "高速成长期"
        elif avg_revenue_growth > 0.1 and avg_profit_growth > 0.1:
            return "成长期"
        elif avg_revenue_growth > 0.05 and avg_profit_growth > 0.05:
            return "稳定成长期"
        elif avg_revenue_growth > 0 and avg_profit_growth > 0:
            return "成熟期"
        else:
            return "调整期"
    
    def _assess_data_quality(self, historical_data: Dict) -> Dict:
        """评估数据质量"""
        total_expected = 10  # 期望获取近5年的年报和中报数据
        
        revenue_count = len(historical_data.get('revenue_history', []))
        profit_count = len(historical_data.get('profit_history', []))
        roe_count = len(historical_data.get('roe_history', []))
        
        return {
            'data_completeness': min(revenue_count / total_expected, 1.0),
            'revenue_data_points': revenue_count,
            'profit_data_points': profit_count,
            'roe_data_points': roe_count,
            'data_years': len(set(historical_data.get('years', []))),
            'quality_score': min((revenue_count + profit_count + roe_count) / (total_expected * 3), 1.0)
        }
    


# 测试函数
async def test_trend_analysis():
    """测试趋势分析功能"""
    analyzer = RealTrendAnalysis()
    
    # 测试平安银行
    result = await analyzer.generate_trend_analysis("000001", "平安银行")
    
    print("=== 趋势分析测试结果 ===")
    print(f"分析期间: {result['analysis_period']}")
    print(f"收入增长率: {result['revenue_growth_5y']}")
    print(f"利润增长率: {result['profit_growth_5y']}")
    print(f"ROE趋势: {result['roe_trend_5y']}")
    print(f"平均收入增长: {result['avg_revenue_growth']:.1%}")
    print(f"平均利润增长: {result['avg_profit_growth']:.1%}")
    print(f"趋势总结: {result['trend_summary']}")
    print(f"成长阶段: {result['growth_stage']}")
    print(f"数据获取耗时: {result['data_fetch_time']}秒")
    print(f"数据质量评分: {result['data_quality']['quality_score']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_trend_analysis())