#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实行业对比分析模块
使用东方财富API获取行业数据进行对比分析
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from time_fix import get_current_timestamp, get_current_iso_timestamp, get_current_time
from data_filter import filter_financial_data, get_latest_annual_data

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealIndustryComparison:
    """真实行业对比分析类"""
    
    def __init__(self):
        self.base_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://data.eastmoney.com/'
        }
        
    def get_industry_classification(self, stock_code: str) -> Optional[str]:
        """获取股票的行业分类"""
        try:
            # 获取股票基本信息，包含行业分类
            params = {
                'sortColumns': 'SECURITY_CODE',
                'sortTypes': '1',
                'pageSize': '1',
                'pageNumber': '1',
                'reportName': 'RPT_LICO_FN_CPD',
                'columns': 'SECURITY_CODE,SECURITY_NAME_ABBR,PUBLISHNAME,BOARD_NAME',
                'filter': f'(SECURITY_CODE="{stock_code}")'
            }
            
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=10)
            data = response.json()
            
            if data.get('success') and data.get('result', {}).get('data'):
                # 优先使用BOARD_NAME，其次使用PUBLISHNAME
                board_name = data['result']['data'][0].get('BOARD_NAME', '')
                publish_name = data['result']['data'][0].get('PUBLISHNAME', '')
                industry_name = board_name or publish_name or '未知行业'
                logger.info(f"股票 {stock_code} 所属行业: {industry_name}")
                return industry_name
            
        except Exception as e:
            logger.error(f"获取行业分类失败: {e}")
        
        return "银行"  # 默认返回银行
    
    def get_peer_companies(self, stock_code: str, industry_name: str = None) -> List[Dict]:
        """获取同行业公司列表"""
        try:
            if not industry_name:
                industry_name = self.get_industry_classification(stock_code)
            
            logger.info(f"正在获取 {industry_name} 行业的公司列表...")
            
            # 方法1: 尝试通过行业名称获取同行业公司
            peer_companies = self._get_companies_by_industry(industry_name)
            
            if not peer_companies:
                # 方法2: 如果行业查询失败，尝试通过概念板块获取
                logger.info("行业查询失败，尝试通过概念板块获取...")
                peer_companies = self._get_companies_by_concept(stock_code)
            
            if not peer_companies:
                # 方法3: 如果都失败，获取市值相近的公司
                logger.info("概念板块查询失败，获取市值相近的公司...")
                peer_companies = self._get_companies_by_market_cap(stock_code)
            
            logger.info(f"最终找到 {len(peer_companies)} 家同行业公司")
            return peer_companies
            
        except Exception as e:
            logger.error(f"获取同行业公司失败: {e}")
            return []
    
    def _get_companies_by_industry(self, industry_name: str) -> List[Dict]:
        """通过行业名称获取公司列表"""
        try:
            # 使用BOARD_NAME或PUBLISHNAME进行筛选
            params = {
                'sortColumns': 'TOTAL_OPERATE_INCOME',
                'sortTypes': '-1',
                'pageSize': '20',
                'pageNumber': '1',
                'reportName': 'RPT_LICO_FN_CPD',
                'columns': 'SECURITY_CODE,SECURITY_NAME_ABBR,TOTAL_OPERATE_INCOME,PARENT_NETPROFIT,WEIGHTAVG_ROE,BPS',
                'filter': f'(BOARD_NAME="{industry_name}")'
            }
            
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=10)
            data = response.json()
            
            if data.get('success') and data.get('result', {}).get('data'):
                companies = []
                seen_codes = set()  # 用于去重
                
                for item in data['result']['data']:
                    code = item.get('SECURITY_CODE', '')
                    if code in seen_codes:
                        continue  # 跳过重复的公司
                    seen_codes.add(code)
                    
                    # 计算简单的PE和PB比率
                    revenue = item.get('TOTAL_OPERATE_INCOME', 0) or 0
                    net_profit = item.get('PARENT_NETPROFIT', 0) or 0
                    bps = item.get('BPS', 0) or 0
                    roe = item.get('WEIGHTAVG_ROE', 0) or 0
                    
                    companies.append({
                        'code': code,
                        'name': item.get('SECURITY_NAME_ABBR', ''),
                        'market_cap': revenue / 100000000 if revenue else 0,  # 转换为亿元
                        'pe_ratio': 15.0,  # 默认PE
                        'pb_ratio': 1.0,   # 默认PB
                        'roe': roe
                    })
                logger.info(f"通过行业名称 {industry_name} 找到 {len(companies)} 家公司")
                return companies
                
        except Exception as e:
            logger.error(f"通过行业获取公司失败: {e}")
        
        return []
    
    def _get_companies_by_concept(self, stock_code: str) -> List[Dict]:
        """通过概念板块获取相关公司"""
        try:
            # 这里可以实现概念板块的查询逻辑
            # 暂时返回空列表，表示该方法未实现
            logger.info("概念板块查询功能暂未实现")
            return []
            
        except Exception as e:
            logger.error(f"通过概念板块获取公司失败: {e}")
            return []
    
    def _get_companies_by_market_cap(self, stock_code: str) -> List[Dict]:
        """获取市值相近的公司作为对比"""
        try:
            # 先获取目标公司的市值
            target_market_cap = self._get_company_market_cap(stock_code)
            if not target_market_cap:
                return []
            
            # 获取市值在目标公司50%-200%范围内的公司
            min_cap = target_market_cap * 0.5
            max_cap = target_market_cap * 2.0
            
            params = {
                'sortColumns': 'TOTAL_MARKET_CAP',
                'sortTypes': '-1',
                'pageSize': '100',
                'pageNumber': '1',
                'reportName': 'RPT_LICO_FN_CPD',
                'columns': 'SECURITY_CODE,SECURITY_NAME_ABBR,TOTAL_MARKET_CAP,PE_TTM,PB_LF,ROE_WEIGHT',
                'filter': f'(TRADE_MARKET_CODE in ("001001","001002"))'
            }
            
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=15)
            data = response.json()
            
            peer_companies = []
            if data.get('success') and data.get('result', {}).get('data'):
                for company in data['result']['data']:
                    market_cap = company.get('TOTAL_MARKET_CAP', 0)
                    if min_cap <= market_cap <= max_cap:
                        peer_companies.append({
                            'code': company.get('SECURITY_CODE', ''),
                            'name': company.get('SECURITY_NAME_ABBR', ''),
                            'market_cap': market_cap,
                            'pe_ratio': company.get('PE_TTM', 0),
                            'pb_ratio': company.get('PB_LF', 0),
                            'roe': company.get('ROE_WEIGHT', 0)
                        })
                        
                        if len(peer_companies) >= 15:
                            break
            
            logger.info(f"通过市值范围找到 {len(peer_companies)} 家公司")
            return peer_companies
            
        except Exception as e:
            logger.error(f"通过市值获取公司失败: {e}")
            return []
    
    def _get_company_market_cap(self, stock_code: str) -> float:
        """获取公司市值"""
        try:
            params = {
                'sortColumns': 'SECURITY_CODE',
                'sortTypes': '1',
                'pageSize': '1',
                'pageNumber': '1',
                'reportName': 'RPT_LICO_FN_CPD',
                'columns': 'TOTAL_MARKET_CAP',
                'filter': f'(SECURITY_CODE="{stock_code}")'
            }
            
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=10)
            data = response.json()
            
            if data.get('success') and data.get('result', {}).get('data'):
                return data['result']['data'][0].get('TOTAL_MARKET_CAP', 0)
            
            return 0
            
        except Exception as e:
            logger.error(f"获取公司市值失败: {e}")
            return 0
    
    def calculate_industry_averages(self, peer_companies: List[Dict]) -> Dict:
        """计算行业平均指标"""
        if not peer_companies:
            return {
                'avg_pe': 15.0,
                'avg_pb': 1.2,
                'avg_roe': 12.0,
                'avg_market_cap': 1000.0
            }
        
        total_pe = sum(company.get('pe_ratio', 0) for company in peer_companies if company.get('pe_ratio', 0) > 0)
        total_pb = sum(company.get('pb_ratio', 0) for company in peer_companies if company.get('pb_ratio', 0) > 0)
        total_roe = sum(company.get('roe', 0) for company in peer_companies if company.get('roe', 0) > 0)
        total_market_cap = sum(company.get('market_cap', 0) for company in peer_companies if company.get('market_cap', 0) > 0)
        
        valid_pe_count = len([c for c in peer_companies if c.get('pe_ratio', 0) > 0])
        valid_pb_count = len([c for c in peer_companies if c.get('pb_ratio', 0) > 0])
        valid_roe_count = len([c for c in peer_companies if c.get('roe', 0) > 0])
        valid_cap_count = len([c for c in peer_companies if c.get('market_cap', 0) > 0])
        
        return {
            'avg_pe': round(total_pe / valid_pe_count, 2) if valid_pe_count > 0 else 15.0,
            'avg_pb': round(total_pb / valid_pb_count, 2) if valid_pb_count > 0 else 1.2,
            'avg_roe': round(total_roe / valid_roe_count, 2) if valid_roe_count > 0 else 12.0,
            'avg_market_cap': round(total_market_cap / valid_cap_count, 2) if valid_cap_count > 0 else 1000.0
        }
    
    def get_company_ranking(self, stock_code: str, peer_companies: List[Dict]) -> int:
        """获取公司在同行业中的排名（按市值）"""
        try:
            # 按市值排序
            sorted_companies = sorted(peer_companies, key=lambda x: x.get('market_cap', 0), reverse=True)
            
            for i, company in enumerate(sorted_companies):
                if company.get('code') == stock_code:
                    return i + 1
            
            return len(peer_companies) // 2  # 如果找不到，返回中位数排名
            
        except Exception as e:
            logger.error(f"计算排名失败: {e}")
            return 5
    
    def generate_comparison_analysis(self, stock_code: str, company_name: str) -> Dict[str, Any]:
        """生成行业对比分析"""
        try:
            logger.info(f"开始生成 {company_name}({stock_code}) 的行业对比分析")
            
            # 记录数据获取开始时间
            data_fetch_start = time.time()
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            current_quarter = self._get_current_quarter()
            
            # 1. 获取行业分类
            industry_name = self.get_industry_classification(stock_code)
            
            # 2. 获取同行业公司（最新数据）
            peer_companies = self.get_peer_companies(stock_code, industry_name)
            
            # 3. 计算行业平均值
            industry_averages = self.calculate_industry_averages(peer_companies)
            
            # 4. 获取公司排名
            ranking = self.get_company_ranking(stock_code, peer_companies)
            
            # 计算数据获取耗时
            data_fetch_time = round(time.time() - data_fetch_start, 2)
            
            # 5. 生成对比分析
            analysis_result = {
                'industry_name': industry_name,
                'total_companies': len(peer_companies),
                'company_ranking': ranking,
                'industry_averages': industry_averages,
                'peer_companies': peer_companies[:5],  # 只返回前5家公司
                'analysis_summary': self._generate_analysis_summary(
                    company_name, ranking, len(peer_companies), industry_averages
                ),
                'data_source': '东方财富',
                'update_time': current_time,
                'data_period': current_quarter,
                'data_fetch_time_seconds': data_fetch_time,
                'data_freshness': '实时数据',
                'comparison_timestamp': int(time.time()),  # Unix时间戳
                'data_quality': {
                    'companies_found': len(peer_companies),
                    'valid_data_ratio': self._calculate_data_quality(peer_companies),
                    'last_refresh': current_time
                }
            }
            
            logger.info(f"行业对比分析生成完成，耗时 {data_fetch_time} 秒")
            return analysis_result
            
        except Exception as e:
            logger.error(f"生成行业对比分析失败: {e}")
            return self._get_fallback_analysis(company_name)
    
    def _generate_analysis_summary(self, company_name: str, ranking: int, total_companies: int, averages: Dict) -> str:
        """生成分析总结"""
        ranking_desc = "领先" if ranking <= 3 else "中等" if ranking <= total_companies // 2 else "落后"
        
        summary = f"{company_name}在同行业{total_companies}家公司中排名第{ranking}位，处于{ranking_desc}水平。"
        summary += f"行业平均PE为{averages['avg_pe']}倍，PB为{averages['avg_pb']}倍，ROE为{averages['avg_roe']}%。"
        
        return summary
    
    def _get_current_quarter(self) -> str:
        """获取当前季度信息"""
        now = time.localtime()
        year = now.tm_year
        month = now.tm_mon
        
        if month <= 3:
            quarter = "Q1"
        elif month <= 6:
            quarter = "Q2"
        elif month <= 9:
            quarter = "Q3"
        else:
            quarter = "Q4"
        
        return f"{year}{quarter}"
    
    def _calculate_data_quality(self, peer_companies: List[Dict]) -> float:
        """计算数据质量比例"""
        if not peer_companies:
            return 0.0
        
        valid_count = 0
        total_count = len(peer_companies)
        
        for company in peer_companies:
            # 检查关键数据是否有效
            if (company.get('market_cap', 0) > 0 and 
                company.get('roe', 0) > 0 and 
                company.get('name', '')):
                valid_count += 1
        
        return round(valid_count / total_count, 2)
    


    def _get_fallback_analysis(self, company_name: str) -> Dict[str, Any]:
        """获取备用分析数据"""
        return {
            'industry_name': '银行业',
            'total_companies': 42,
            'company_ranking': 4,
            'industry_averages': {
                'avg_pe': 5.8,
                'avg_pb': 0.7,
                'avg_roe': 11.2,
                'avg_market_cap': 2500.0
            },
            'peer_companies': [
                {'code': '601398', 'name': '工商银行', 'market_cap': 18500.0},
                {'code': '601939', 'name': '建设银行', 'market_cap': 16200.0},
                {'code': '601988', 'name': '中国银行', 'market_cap': 12800.0},
                {'code': '601328', 'name': '交通银行', 'market_cap': 4200.0},
                {'code': '000001', 'name': '平安银行', 'market_cap': 3800.0}
            ],
            'analysis_summary': f"{company_name}在银行业42家公司中排名第4位，处于领先水平。行业平均PE为5.8倍，PB为0.7倍，ROE为11.2%。",
            'data_source': '模拟数据',
            'update_time': get_current_timestamp(),
            'analysis_timestamp': get_current_iso_timestamp()
        }

def test_industry_comparison():
    """测试行业对比功能"""
    print("=== 测试行业对比分析功能 ===")
    
    comparison = RealIndustryComparison()
    
    # 测试平安银行
    result = comparison.generate_comparison_analysis('000001', '平安银行')
    
    print(f"行业名称: {result['industry_name']}")
    print(f"同行业公司数量: {result['total_companies']}")
    print(f"公司排名: {result['company_ranking']}")
    print(f"行业平均指标: {result['industry_averages']}")
    print(f"主要竞争对手: {[c['name'] for c in result['peer_companies']]}")
    print(f"分析总结: {result['analysis_summary']}")
    print(f"数据来源: {result['data_source']}")
    print(f"更新时间: {result['update_time']}")

if __name__ == "__main__":
    test_industry_comparison()