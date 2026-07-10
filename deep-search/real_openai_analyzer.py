#!/usr/bin/env python3
"""
真实OpenAI大模型财务分析工具
使用配置的Qwen3-Next-80B-A3B模型
"""
import requests
import json
import time
from datetime import datetime
import logging
import os
from typing import Dict, List, Optional
import asyncio
from dotenv import load_dotenv
from time_fix import get_current_timestamp, get_current_iso_timestamp

# 加载环境变量
load_dotenv()

class RealOpenAIAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY', 'sk-eD9oqSXvWMfQg5hUv487qw')
        self.base_url = os.getenv('OPENAI_BASE_URL', 'http://172.32.153.184:14000/v1')
        self.model = os.getenv('OPENAI_MODEL', 'Qwen3-Next-80B-A3B')
        
        self.proxies = {
            'http': os.getenv('HTTP_PROXY', 'http://172.32.147.190:7890'),
            'https': os.getenv('HTTPS_PROXY', 'http://172.32.147.190:7890')
        }
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.proxies.update(self.proxies)
        
        print(f"🤖 初始化OpenAI分析器")
        print(f"📡 API地址: {self.base_url}")
        print(f"🧠 模型: {self.model}")
    
    async def analyze_financial_data(self, company_data: Dict, financial_data: Dict) -> Dict:
        """使用真实OpenAI API分析财务数据"""
        company_name = company_data.get('name', '未知公司')
        print(f"🔍 正在使用 {self.model} 分析 {company_name} 的财务数据...")
        
        # 构建专业的财务分析提示词
        analysis_prompt = self._build_professional_prompt(company_data, financial_data)
        
        # 调用OpenAI API
        analysis_result = await self._call_openai_api(analysis_prompt)
        
        return {
            'company_name': company_name,
            'stock_code': company_data.get('code', ''),
            'analysis_result': analysis_result,
            'analysis_time': get_current_timestamp(),
            'model_used': self.model,
            'api_endpoint': self.base_url
        }
    
    def _build_professional_prompt(self, company_data: Dict, financial_data: Dict) -> str:
        """构建专业的财务分析提示词"""
        basic_data = financial_data.get('basic_data', {})
        
        prompt = f"""你是一位资深的金融分析师，请对以下公司进行专业的财务分析：

## 公司基本信息
- 公司名称：{company_data.get('name', '未知')}
- 股票代码：{company_data.get('code', '未知')}
- 所属行业：{company_data.get('industry', '未知')}
- 交易市场：{company_data.get('market', '未知')}

## 当前财务数据
- 当前股价：{basic_data.get('current_price', 0):.2f} 元
- 涨跌幅：{basic_data.get('change_percent', 0):.2f}%
- 市盈率(PE)：{basic_data.get('pe_ratio', 0):.2f}
- 市净率(PB)：{basic_data.get('pb_ratio', 0):.2f}
- 总市值：{basic_data.get('market_cap', 0):,.0f} 元
- 成交量：{basic_data.get('volume', 0):,} 手

## 分析要求
请从以下维度进行深入分析，并给出具体的数据支撑：

### 1. 估值分析
- PE、PB比率的合理性评估
- 与行业平均水平对比
- 历史估值区间分析

### 2. 财务健康度评估
- 盈利能力分析
- 偿债能力评估
- 营运能力判断

### 3. 投资风险评估
- 系统性风险（宏观经济、政策风险）
- 行业风险（竞争格局、发展趋势）
- 个股风险（经营风险、财务风险）

### 4. 投资建议
- 明确的投资评级（强烈买入/买入/持有/减持/卖出）
- 目标价位区间
- 投资逻辑和理由
- 风险提示

### 5. 关键关注点
- 需要重点关注的财务指标
- 未来业绩催化剂
- 潜在风险因素

请用专业、客观的语言进行分析，提供具体的数据支撑和逻辑推理。分析结果要有实用性，能够为投资决策提供参考。
"""
        return prompt
    
    async def _call_openai_api(self, prompt: str) -> Dict:
        """调用OpenAI API"""
        try:
            url = f"{self.base_url}/chat/completions"
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位专业的金融分析师，具有丰富的股票投资和财务分析经验。请提供专业、客观、实用的分析建议。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.9
            }
            
            print(f"📡 正在调用API: {url}")
            response = self.session.post(url, json=data, timeout=60)
            
            print(f"📊 API响应状态: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                analysis_content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                return {
                    'success': True,
                    'analysis': analysis_content,
                    'model_used': self.model,
                    'tokens_used': result.get('usage', {}).get('total_tokens', 0),
                    'api_response_time': response.elapsed.total_seconds()
                }
            else:
                error_msg = f"API调用失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'fallback_analysis': self._generate_fallback_analysis()
                }
                
        except Exception as e:
            error_msg = f"API调用异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'fallback_analysis': self._generate_fallback_analysis()
            }
    
    def _generate_fallback_analysis(self) -> str:
        """生成备用分析"""
        return """
## 备用分析报告

### 1. 估值分析
基于当前PE和PB比率，公司估值处于相对合理区间。建议结合行业平均水平进行对比分析。

### 2. 财务健康度
从基本财务指标看，公司基本面相对稳健。建议关注现金流状况和负债结构。

### 3. 投资风险
- 系统性风险：需关注宏观经济环境变化
- 行业风险：关注政策变化和竞争格局
- 个股风险：建议深入研究公司治理结构

### 4. 投资建议
评级：谨慎持有
理由：基于有限数据，建议进一步收集更多财务信息后再做决策

### 5. 风险提示
本分析基于有限数据，投资有风险，决策需谨慎。
"""

    async def batch_analyze_companies(self, companies_data: List[Dict]) -> List[Dict]:
        """批量分析多家公司"""
        results = []
        for i, company_data in enumerate(companies_data):
            print(f"\n📈 分析进度: {i+1}/{len(companies_data)}")
            
            # 这里需要获取每家公司的财务数据
            # 为演示目的，使用模拟数据
            financial_data = {
                'basic_data': {
                    'current_price': 12.5,
                    'change_percent': 1.2,
                    'pe_ratio': 8.5,
                    'pb_ratio': 0.8,
                    'market_cap': 150000000000,
                    'volume': 500000
                }
            }
            
            result = await self.analyze_financial_data(company_data, financial_data)
            results.append(result)
            
            # 避免API调用过快
            await asyncio.sleep(2)
        
        return results

# 测试函数
async def test_real_openai_analysis():
    analyzer = RealOpenAIAnalyzer()
    
    # 测试单个公司分析
    company_data = {
        'name': '平安银行',
        'code': '000001',
        'industry': '银行业',
        'market': '深市'
    }
    
    financial_data = {
        'basic_data': {
            'current_price': 12.85,
            'change_percent': 1.58,
            'pe_ratio': 4.45,
            'pb_ratio': 0.50,
            'market_cap': 221227467457,
            'volume': 753239
        }
    }
    
    print("🚀 开始财务分析测试...")
    result = await analyzer.analyze_financial_data(company_data, financial_data)
    
    print("\n" + "="*60)
    print("📊 分析结果:")
    print("="*60)
    print(f"公司: {result['company_name']} ({result['stock_code']})")
    print(f"分析时间: {result['analysis_time']}")
    print(f"使用模型: {result['model_used']}")
    
    if result['analysis_result']['success']:
        print(f"Token使用: {result['analysis_result'].get('tokens_used', 0)}")
        print(f"响应时间: {result['analysis_result'].get('api_response_time', 0):.2f}秒")
        print("\n📝 分析内容:")
        print("-" * 40)
        print(result['analysis_result']['analysis'])
    else:
        print("❌ API调用失败，使用备用分析:")
        print(result['analysis_result'].get('fallback_analysis', ''))
    
    # 保存结果
    filename = f"openai_analysis_{company_data['name']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 分析结果已保存到: {filename}")

if __name__ == "__main__":
    asyncio.run(test_real_openai_analysis())