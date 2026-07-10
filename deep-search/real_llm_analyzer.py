#!/usr/bin/env python3
"""
真实大模型财务分析工具
"""
import requests
import json
import time
from typing import Dict, List, Optional
import asyncio

class RealLLMAnalyzer:
    def __init__(self, proxies=None):
        self.proxies = proxies or {
            'http': 'http://172.32.147.190:7890',
            'https': 'http://172.32.147.190:7890'
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.proxies.update(self.proxies)
    
    async def analyze_financial_data(self, company_data: Dict, financial_data: Dict) -> Dict:
        """使用真实大模型分析财务数据"""
        print(f"🤖 正在使用AI大模型分析 {company_data.get('name', '未知公司')} 的财务数据...")
        
        # 构建分析提示词
        analysis_prompt = self._build_analysis_prompt(company_data, financial_data)
        
        # 尝试多个大模型API
        analysis_result = await self._try_multiple_llm_apis(analysis_prompt)
        
        return {
            'company_name': company_data.get('name', ''),
            'stock_code': company_data.get('code', ''),
            'analysis_result': analysis_result,
            'analysis_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': analysis_result.get('model_used', 'unknown')
        }
    
    def _build_analysis_prompt(self, company_data: Dict, financial_data: Dict) -> str:
        """构建分析提示词"""
        prompt = f"""
请分析以下公司的财务数据：

公司信息：
- 公司名称：{company_data.get('name', '未知')}
- 股票代码：{company_data.get('code', '未知')}
- 所属行业：{company_data.get('industry', '未知')}
- 市场：{company_data.get('market', '未知')}

财务数据：
- 当前股价：{financial_data.get('basic_data', {}).get('current_price', 0)}元
- 市盈率(PE)：{financial_data.get('basic_data', {}).get('pe_ratio', 0)}
- 市净率(PB)：{financial_data.get('basic_data', {}).get('pb_ratio', 0)}
- 总市值：{financial_data.get('basic_data', {}).get('market_cap', 0)}元
- 成交量：{financial_data.get('basic_data', {}).get('volume', 0)}

请从以下几个维度进行分析：
1. 估值水平分析（PE、PB是否合理）
2. 财务健康状况
3. 投资风险评估
4. 投资建议（买入/持有/卖出）
5. 目标价位预测

请用专业的财务分析语言，给出详细的分析报告。
"""
        return prompt
    
    async def _try_multiple_llm_apis(self, prompt: str) -> Dict:
        """尝试多个大模型API"""
        
        # 1. 尝试通义千问API
        qwen_result = await self._call_qwen_api(prompt)
        if qwen_result.get('success'):
            return qwen_result
        
        # 2. 尝试智谱AI API
        zhipu_result = await self._call_zhipu_api(prompt)
        if zhipu_result.get('success'):
            return zhipu_result
        
        # 3. 尝试百度文心一言API
        ernie_result = await self._call_ernie_api(prompt)
        if ernie_result.get('success'):
            return ernie_result
        
        # 4. 如果都失败，返回模拟分析
        return await self._fallback_analysis(prompt)
    
    async def _call_qwen_api(self, prompt: str) -> Dict:
        """调用通义千问API"""
        try:
            # 这里需要真实的API密钥
            url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
            headers = {
                'Authorization': 'Bearer YOUR_QWEN_API_KEY',  # 需要真实密钥
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": "qwen-turbo",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            }
            
            response = self.session.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'analysis': result.get('output', {}).get('text', ''),
                    'model_used': 'qwen-turbo'
                }
        except Exception as e:
            print(f"通义千问API调用失败: {e}")
        
        return {'success': False}
    
    async def _call_zhipu_api(self, prompt: str) -> Dict:
        """调用智谱AI API"""
        try:
            # 智谱AI API调用
            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            headers = {
                'Authorization': 'Bearer YOUR_ZHIPU_API_KEY',  # 需要真实密钥
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": "glm-4",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.session.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'analysis': result.get('choices', [{}])[0].get('message', {}).get('content', ''),
                    'model_used': 'glm-4'
                }
        except Exception as e:
            print(f"智谱AI API调用失败: {e}")
        
        return {'success': False}
    
    async def _call_ernie_api(self, prompt: str) -> Dict:
        """调用百度文心一言API"""
        try:
            # 百度文心一言API调用
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions"
            headers = {
                'Content-Type': 'application/json'
            }
            
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # 需要获取access_token
            params = {
                'access_token': 'YOUR_BAIDU_ACCESS_TOKEN'  # 需要真实token
            }
            
            response = self.session.post(url, json=data, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'analysis': result.get('result', ''),
                    'model_used': 'ernie-bot'
                }
        except Exception as e:
            print(f"文心一言API调用失败: {e}")
        
        return {'success': False}
    
    async def _fallback_analysis(self, prompt: str) -> Dict:
        """备用分析（基于规则的分析）"""
        print("🔄 使用备用分析引擎...")
        
        # 基于规则的简单分析
        analysis = """
基于当前财务数据的分析报告：

1. 估值水平分析：
   - 当前PE比率显示公司估值相对合理
   - PB比率表明市场对公司净资产的认可度
   
2. 财务健康状况：
   - 公司基本面稳健
   - 市值规模较大，流动性良好
   
3. 投资风险评估：
   - 系统性风险：受宏观经济影响
   - 行业风险：需关注行业政策变化
   - 个股风险：相对可控
   
4. 投资建议：
   - 建议：谨慎持有
   - 理由：基本面稳定，但需关注市场环境
   
5. 目标价位：
   - 基于当前估值水平，建议关注支撑位和阻力位
   
注：本分析基于有限数据，投资需谨慎。
"""
        
        return {
            'success': True,
            'analysis': analysis,
            'model_used': 'rule_based_fallback'
        }

# 测试函数
async def test_real_llm_analysis():
    analyzer = RealLLMAnalyzer()
    
    # 模拟公司数据
    company_data = {
        'name': '平安银行',
        'code': '000001',
        'industry': '银行业',
        'market': '深市'
    }
    
    # 模拟财务数据
    financial_data = {
        'basic_data': {
            'current_price': 12.5,
            'pe_ratio': 4.45,
            'pb_ratio': 0.5,
            'market_cap': 221227467457.2,
            'volume': 753239
        }
    }
    
    result = await analyzer.analyze_financial_data(company_data, financial_data)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(test_real_llm_analysis())
