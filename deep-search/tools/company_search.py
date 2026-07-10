#!/usr/bin/env python3
"""
真实公司搜索工具 - 集成多个真实数据源API
"""
import requests
import json
import time
import re
from typing import Dict, List, Optional
import asyncio

class RealCompanySearchTool:
    def __init__(self, proxies=None):
        self.proxies = proxies or {
            'http': 'http://172.32.147.190:7890',
            'https': 'http://172.32.147.190:7890'
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'http://quote.eastmoney.com/'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.proxies.update(self.proxies)
    
    async def search_company(self, company_name: str) -> Dict:
        """真实搜索公司信息"""
        print(f"🔍 正在搜索公司: {company_name}")
        
        # 1. 尝试东方财富搜索
        eastmoney_result = await self._search_eastmoney_real(company_name)
        if eastmoney_result and eastmoney_result.get('code'):
            print(f"✅ 东方财富找到: {eastmoney_result['name']} ({eastmoney_result['code']})")
            # 获取详细信息
            detailed_info = await self._get_detailed_info(eastmoney_result['code'])
            eastmoney_result.update(detailed_info)
            return eastmoney_result
        
        # 2. 尝试腾讯财经搜索
        tencent_result = await self._search_tencent_real(company_name)
        if tencent_result and tencent_result.get('code'):
            print(f"✅ 腾讯财经找到: {tencent_result['name']} ({tencent_result['code']})")
            return tencent_result
        
        # 3. 尝试新浪财经搜索
        sina_result = await self._search_sina_real(company_name)
        if sina_result and sina_result.get('code'):
            print(f"✅ 新浪财经找到: {sina_result['name']} ({sina_result['code']})")
            return sina_result
        
        print(f"⚠️ 未找到公司信息，返回默认值")
        return {
            'name': company_name,
            'code': '',
            'market': '未知',
            'industry': '未知',
            'market_cap': '未知',
            'pe_ratio': 0,
            'pb_ratio': 0,
            'source': 'not_found'
        }
    
    async def _search_eastmoney_real(self, company_name: str) -> Optional[Dict]:
        """真实的东方财富搜索"""
        try:
            # 东方财富搜索API
            url = "http://searchapi.eastmoney.com/api/suggest/get"
            params = {
                'input': company_name,
                'type': '14',  # 股票类型
                'count': '10',
                'market': '',
                'cb': 'jQuery'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('QuotationCodeTable', {}).get('Data'):
                    # 找最匹配的结果
                    for stock_data in data['QuotationCodeTable']['Data']:
                        stock_name = stock_data.get('Name', '')
                        if company_name in stock_name or stock_name in company_name:
                            market_num = stock_data.get('MktNum', '0')
                            market_name = '深市' if market_num == '0' else '沪市' if market_num == '1' else '未知'
                            
                            return {
                                'name': stock_name,
                                'code': stock_data.get('Code', ''),
                                'market': market_name,
                                'market_num': market_num,
                                'industry': '待获取',
                                'source': 'eastmoney'
                            }
        except Exception as e:
            print(f"东方财富搜索失败: {e}")
        
        return None
    
    async def _search_tencent_real(self, company_name: str) -> Optional[Dict]:
        """真实的腾讯财经搜索"""
        try:
            # 腾讯财经搜索API
            url = "http://smartbox.gtimg.cn/s3/"
            params = {
                'q': company_name,
                't': 'gp',
                'c': '1'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                content = response.text
                # 解析腾讯返回的数据格式
                if 'v_hint=' in content:
                    # 提取股票信息
                    lines = content.split('\n')
                    for line in lines:
                        if company_name in line and '~' in line:
                            parts = line.split('~')
                            if len(parts) >= 3:
                                return {
                                    'name': parts[1] if len(parts) > 1 else company_name,
                                    'code': parts[0].split('_')[-1] if '_' in parts[0] else '',
                                    'market': '腾讯财经',
                                    'industry': '待获取',
                                    'source': 'tencent'
                                }
        except Exception as e:
            print(f"腾讯财经搜索失败: {e}")
        
        return None
    
    async def _search_sina_real(self, company_name: str) -> Optional[Dict]:
        """真实的新浪财经搜索"""
        try:
            # 新浪财经搜索API
            url = "http://suggest3.sinajs.cn/suggest/type=11,12,13,14,15"
            params = {
                'key': company_name,
                'name': 'suggestdata'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                content = response.text
                if 'suggestdata=' in content:
                    # 提取数据
                    data_str = content.split('suggestdata=')[1].strip(';')
                    if data_str and data_str != '""':
                        # 解析新浪的数据格式
                        items = data_str.strip('"').split(';')
                        for item in items:
                            if company_name in item:
                                parts = item.split(',')
                                if len(parts) >= 4:
                                    return {
                                        'name': parts[4] if len(parts) > 4 else company_name,
                                        'code': parts[3] if len(parts) > 3 else '',
                                        'market': '新浪财经',
                                        'industry': '待获取',
                                        'source': 'sina'
                                    }
        except Exception as e:
            print(f"新浪财经搜索失败: {e}")
        
        return None
    
    async def _get_detailed_info(self, stock_code: str) -> Dict:
        """获取股票详细信息"""
        try:
            # 东方财富股票详情API
            market_prefix = "1" if stock_code.startswith('6') else "0"
            url = "http://push2.eastmoney.com/api/qt/stock/get"
            params = {
                'secid': f"{market_prefix}.{stock_code}",
                'fields': 'f57,f58,f162,f163,f164,f165,f166,f167,f168,f169,f170,f46,f44,f45,f47,f260'
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
                        'industry': '银行业'  # 可以通过其他API获取
                    }
        except Exception as e:
            print(f"获取详细信息失败: {e}")
        
        return {
            'current_price': 0,
            'change_percent': 0,
            'volume': 0,
            'market_cap': 0,
            'pe_ratio': 0,
            'pb_ratio': 0,
            'industry': '未知'
        }
    
    async def get_company_profile(self, stock_code: str) -> Dict:
        """获取公司概况"""
        try:
            # 东方财富公司概况API
            market_prefix = "1" if stock_code.startswith('6') else "0"
            url = "http://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/CompanySurveyAjax"
            params = {
                'code': f"{market_prefix}{stock_code}"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('jbzl'):  # 基本资料
                    profile = data['jbzl'][0] if data['jbzl'] else {}
                    return {
                        'company_name': profile.get('SECURITY_NAME_ABBR', ''),
                        'industry': profile.get('CSRC_INDUSTRY_NAME', ''),
                        'main_business': profile.get('MAIN_BUSINESS', ''),
                        'business_scope': profile.get('BUSINESS_SCOPE', ''),
                        'established_date': profile.get('FOUND_DATE', ''),
                        'listing_date': profile.get('LISTING_DATE', ''),
                        'total_share_capital': profile.get('TOTAL_SHARE_CAPITAL', 0),
                        'source': 'eastmoney_profile'
                    }
        except Exception as e:
            print(f"获取公司概况失败: {e}")
        
        return {
            'company_name': '',
            'industry': '未知',
            'main_business': '未知',
            'business_scope': '未知',
            'established_date': '',
            'listing_date': '',
            'total_share_capital': 0,
            'source': 'default'
        }
    
    def validate_stock_code(self, code: str) -> bool:
        """验证股票代码格式"""
        if not code or len(code) != 6:
            return False
        return code.isdigit()
    
    async def batch_search_companies(self, company_names: List[str]) -> List[Dict]:
        """批量搜索公司"""
        results = []
        for name in company_names:
            result = await self.search_company(name)
            results.append(result)
            await asyncio.sleep(1)  # 避免请求过快
        return results