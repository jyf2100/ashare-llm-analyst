#!/usr/bin/env python3
"""
真实财报数据获取工具
"""
import requests
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Optional
import asyncio
from time_fix import get_current_timestamp, get_current_iso_timestamp, get_report_date

class RealFinancialDataTool:
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
    
    async def get_financial_reports(self, stock_code: str) -> Dict:
        """获取真实的财报数据"""
        print(f"📊 正在获取 {stock_code} 的财报数据...")
        
        # 获取基本财务数据
        basic_data = await self._get_basic_financial_data(stock_code)
        
        # 获取财务指标
        financial_ratios = await self._get_financial_ratios(stock_code)
        
        # 获取现金流数据
        cashflow_data = await self._get_cashflow_data(stock_code)
        
        return {
            'stock_code': stock_code,
            'basic_data': basic_data,
            'financial_ratios': financial_ratios,
            'cashflow_data': cashflow_data,
            'data_source': 'eastmoney_api',
            'update_time': get_current_timestamp(),
            'analysis_timestamp': get_current_iso_timestamp()
        }
    
    async def _get_basic_financial_data(self, stock_code: str) -> Dict:
        """获取基本财务数据"""
        try:
            market_prefix = "1" if stock_code.startswith('6') else "0"
            url = "http://push2.eastmoney.com/api/qt/stock/get"
            params = {
                'secid': f"{market_prefix}.{stock_code}",
                'fields': 'f57,f58,f162,f163,f164,f165,f166,f167,f168,f169,f170,f46,f44,f45,f47,f260,f116'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    stock_data = data['data']
                    return {
                        'current_price': stock_data.get('f43', 0) / 100 if stock_data.get('f43') else 0,
                        'change_percent': stock_data.get('f170', 0) / 100 if stock_data.get('f170') else 0,
                        'volume': stock_data.get('f47', 0),
                        'market_cap': stock_data.get('f116', 0),
                        'pe_ratio': stock_data.get('f162', 0) / 100 if stock_data.get('f162') else 0,
                        'pb_ratio': stock_data.get('f167', 0) / 100 if stock_data.get('f167') else 0,
                        'total_market_value': stock_data.get('f116', 0),
                        'circulation_market_value': stock_data.get('f117', 0)
                    }
        except Exception as e:
            print(f"获取基本财务数据失败: {e}")
        
        return {}
    
    async def _get_financial_ratios(self, stock_code: str) -> Dict:
        """获取财务指标"""
        try:
            # 使用新的东方财富财务指标API
            url = "http://datacenter-web.eastmoney.com/api/data/v1/get"
            params = {
                'sortColumns': 'NOTICE_DATE,SECURITY_CODE',
                'sortTypes': '-1,-1',
                'pageSize': '5',
                'pageNumber': '1',
                'reportName': 'RPT_LICO_FN_CPD',
                'columns': 'SECURITY_CODE,SECURITY_NAME_ABBR,REPORTDATE,BASIC_EPS,BPS,MGJYXJJE,TOTAL_OPERATE_INCOME,PARENT_NETPROFIT',
                'filter': f'(SECURITY_CODE="{stock_code}")'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success') and data.get('result', {}).get('data'):
                    latest_data = data['result']['data'][0]
                    
                    ratios = {
                        'basic_eps': latest_data.get('BASIC_EPS'),  # 基本每股收益
                        'bps': latest_data.get('BPS'),  # 每股净资产
                        'operating_cashflow_per_share': latest_data.get('MGJYXJJE'),  # 每股经营现金流
                        'total_operate_income': latest_data.get('TOTAL_OPERATE_INCOME'),  # 营业总收入
                        'parent_netprofit': latest_data.get('PARENT_NETPROFIT'),  # 归母净利润
                        'report_date': latest_data.get('REPORTDATE'),
                        'data_type': latest_data.get('DATATYPE'),  # 报告期类型
                        'security_name': latest_data.get('SECURITY_NAME_ABBR')
                    }
                    
                    # 计算一些比率
                    if ratios['parent_netprofit'] and ratios['total_operate_income']:
                        ratios['net_margin'] = (ratios['parent_netprofit'] / ratios['total_operate_income']) * 100
                    
                    # 计算ROE (净资产收益率) = 基本每股收益 / 每股净资产
                    if ratios['basic_eps'] and ratios['bps']:
                        ratios['roe'] = (ratios['basic_eps'] / ratios['bps'])
                    
                    # 过滤掉None值
                    ratios = {k: v for k, v in ratios.items() if v is not None}
                    return ratios
                    
        except Exception as e:
            print(f"获取财务指标失败: {e}")
        
        return {}
    
    async def _get_cashflow_data(self, stock_code: str) -> Dict:
        """获取现金流数据"""
        try:
            # 使用新的东方财富现金流量表API
            url = "http://datacenter-web.eastmoney.com/api/data/v1/get"
            params = {
                'sortColumns': 'NOTICE_DATE,SECURITY_CODE',
                'sortTypes': '-1,-1',
                'pageSize': '3',
                'pageNumber': '1',
                'reportName': 'RPT_DMSK_FN_CASHFLOW',
                'columns': 'SECURITY_CODE,SECURITY_NAME_ABBR,NOTICE_DATE,NETCASH_OPERATE,NETCASH_INVEST,NETCASH_FINANCE,CCE_ADD,LOAN_ADVANCE_ADD,CUSTOMER_DEPOSIT_ADD',
                'filter': f'(SECURITY_CODE="{stock_code}")'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success') and data.get('result', {}).get('data'):
                    latest_data = data['result']['data'][0]
                    
                    cashflow = {
                        'operating_cashflow': latest_data.get('NETCASH_OPERATE'),  # 经营活动现金流量净额
                        'investing_cashflow': latest_data.get('NETCASH_INVEST'),  # 投资活动现金流量净额
                        'financing_cashflow': latest_data.get('NETCASH_FINANCE'),  # 筹资活动现金流量净额
                        'cash_equivalent_add': latest_data.get('CCE_ADD'),  # 现金及现金等价物净增加额
                        'loan_advance_add': latest_data.get('LOAN_ADVANCE_ADD'),  # 客户贷款及垫款净增加额
                        'customer_deposit_add': latest_data.get('CUSTOMER_DEPOSIT_ADD'),  # 客户存款和同业存放款项净增加额
                        'notice_date': latest_data.get('NOTICE_DATE'),
                        'security_name': latest_data.get('SECURITY_NAME_ABBR')
                    }
                    
                    # 计算自由现金流 (经营现金流 - 投资现金流的绝对值)
                    if cashflow.get('operating_cashflow') and cashflow.get('investing_cashflow'):
                        cashflow['free_cashflow'] = cashflow['operating_cashflow'] - abs(cashflow['investing_cashflow'])
                    
                    # 过滤掉None值
                    cashflow = {k: v for k, v in cashflow.items() if v is not None}
                    return cashflow
                    
        except Exception as e:
            print(f"获取现金流数据失败: {e}")
        
        return {}
    
    def _extract_ratio_data(self, data: List, field_name: str) -> List:
        """提取财务比率数据"""
        for item in data:
            if item.get('REPORTNAME') == field_name:
                return item.get('DATELIST', [])
        return []
    
    def _extract_cashflow_data(self, data: List, field_name: str) -> List:
        """提取现金流数据"""
        for item in data:
            if item.get('REPORTNAME') == field_name:
                return item.get('DATELIST', [])
        return []

# 测试函数
async def test_real_financial_data():
    tool = RealFinancialDataTool()
    result = await tool.get_financial_reports("000001")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(test_real_financial_data())
