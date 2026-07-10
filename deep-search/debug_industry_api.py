#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试行业API调用
"""

import requests
import json

def test_industry_api():
    """测试行业API调用"""
    base_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://data.eastmoney.com/'
    }
    
    print("=== 测试1: 获取平安银行基本信息 ===")
    params1 = {
        'sortColumns': 'SECURITY_CODE',
        'sortTypes': '1',
        'pageSize': '1',
        'pageNumber': '1',
        'reportName': 'RPT_LICO_FN_CPD',
        'columns': 'SECURITY_CODE,SECURITY_NAME_ABBR,TOTAL_MARKET_CAP,INDUSTRY_NAME',
        'filter': '(SECURITY_CODE="000001")'
    }
    
    try:
        response = requests.get(base_url, params=params1, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        if data.get('success') and data.get('result', {}).get('data'):
            company_info = data['result']['data'][0]
            industry_name = company_info.get('INDUSTRY_NAME', '')
            print(f"平安银行所属行业: {industry_name}")
            
            print("\n=== 测试2: 获取同行业公司 ===")
            params2 = {
                'sortColumns': 'TOTAL_MARKET_CAP',
                'sortTypes': '-1',
                'pageSize': '10',
                'pageNumber': '1',
                'reportName': 'RPT_LICO_FN_CPD',
                'columns': 'SECURITY_CODE,SECURITY_NAME_ABBR,TOTAL_MARKET_CAP,INDUSTRY_NAME',
                'filter': f'(TRADE_MARKET_CODE in ("001001","001002"))'
            }
            
            response2 = requests.get(base_url, params=params2, headers=headers, timeout=15)
            print(f"状态码: {response2.status_code}")
            data2 = response2.json()
            
            if data2.get('success') and data2.get('result', {}).get('data'):
                print(f"总共找到 {len(data2['result']['data'])} 家公司")
                
                # 查找同行业公司
                same_industry_companies = []
                for company in data2['result']['data']:
                    company_industry = company.get('INDUSTRY_NAME', '')
                    if industry_name and (industry_name in company_industry or company_industry in industry_name):
                        same_industry_companies.append(company)
                
                print(f"同行业公司数量: {len(same_industry_companies)}")
                for company in same_industry_companies[:5]:
                    print(f"  - {company.get('SECURITY_NAME_ABBR')} ({company.get('SECURITY_CODE')}) - {company.get('INDUSTRY_NAME')}")
            else:
                print("获取公司列表失败")
                print(f"响应: {json.dumps(data2, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        print(f"API调用失败: {e}")

    print("\n=== 测试3: 直接搜索银行业公司 ===")
    params3 = {
        'sortColumns': 'TOTAL_MARKET_CAP',
        'sortTypes': '-1',
        'pageSize': '20',
        'pageNumber': '1',
        'reportName': 'RPT_LICO_FN_CPD',
        'columns': 'SECURITY_CODE,SECURITY_NAME_ABBR,TOTAL_MARKET_CAP,INDUSTRY_NAME',
        'filter': '(INDUSTRY_NAME like "%银行%")'
    }
    
    try:
        response3 = requests.get(base_url, params=params3, headers=headers, timeout=15)
        print(f"状态码: {response3.status_code}")
        data3 = response3.json()
        
        if data3.get('success') and data3.get('result', {}).get('data'):
            print(f"找到银行业公司: {len(data3['result']['data'])} 家")
            for company in data3['result']['data'][:10]:
                print(f"  - {company.get('SECURITY_NAME_ABBR')} ({company.get('SECURITY_CODE')}) - {company.get('INDUSTRY_NAME')}")
        else:
            print("搜索银行业公司失败")
            print(f"响应: {json.dumps(data3, indent=2, ensure_ascii=False)}")
            
    except Exception as e:
        print(f"搜索银行业公司失败: {e}")

if __name__ == "__main__":
    test_industry_api()