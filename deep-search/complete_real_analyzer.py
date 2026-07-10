#!/usr/bin/env python3
"""
完整的真实财务分析系统
整合真实API：公司搜索 + 财报数据 + OpenAI分析 + 网络搜索
"""
from datetime import date
import requests
import json
import time
import os
import asyncio
from typing import Dict, List, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class CompleteRealAnalyzer:
    def __init__(self):
        # API配置
        self.openai_api_key = os.getenv('OPENAI_API_KEY', 'sk-eD9oqSXvWMfQg5hUv487qw')
        self.openai_base_url = os.getenv('OPENAI_BASE_URL', 'http://172.32.153.184:14000/v1')
        self.openai_model = os.getenv('OPENAI_MODEL', 'Qwen3-Next-80B-A3B')
        self.search_api_key = os.getenv('SEARCH_API_KEY', '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')
        
        # 代理配置
        self.proxies = {
            'http': os.getenv('HTTP_PROXY', 'http://172.32.147.190:7890'),
            'https': os.getenv('HTTPS_PROXY', 'http://172.32.147.190:7890')
        }
        
        # 请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*'
        }
        
        # 创建会话
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.proxies.update(self.proxies)
        
        print("🚀 初始化完整财务分析系统")
        print(f"🧠 AI模型: {self.openai_model}")
        print(f"🔍 搜索引擎: Tavily")
        print(f"📊 数据源: 东方财富")
    
    async def complete_analysis(self, company_name: str) -> Dict:
        """完整的财务分析流程"""
        print(f"\n{'='*60}")
        print(f"🎯 开始分析: {company_name}")
        print(f"{'='*60}")
        
        analysis_start_time = time.time()
        
        # 第1步：搜索公司信息
        print("\n📋 第1步：搜索公司基本信息")
        company_info = await self._search_company_info(company_name)
        
        # 第2步：获取财务数据
        print("\n📊 第2步：获取财务数据")
        financial_data = await self._get_financial_data(company_info.get('code', ''))
        
        # 第3步：搜索相关新闻
        print("\n📰 第3步：搜索相关新闻和报告")
        news_data = await self._search_financial_news(company_name)
        
        # 第4步：AI深度分析
        print("\n🤖 第4步：AI深度分析")
        ai_analysis = await self._ai_deep_analysis(company_info, financial_data, news_data)
        
        # 第5步：生成投资建议
        print("\n💡 第5步：生成投资建议")
        investment_advice = await self._generate_investment_advice(company_info, financial_data, ai_analysis)
        
        analysis_end_time = time.time()
        
        # 整合结果
        complete_result = {
            'analysis_id': f"analysis_{int(time.time())}",
            'company_info': company_info,
            'financial_data': financial_data,
            'news_data': news_data,
            'ai_analysis': ai_analysis,
            'investment_advice': investment_advice,
            'analysis_metadata': {
                'analysis_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_duration': f"{analysis_end_time - analysis_start_time:.2f}秒",
                'model_used': self.openai_model,
                'data_sources': ['eastmoney', 'tavily', 'openai']
            }
        }
        
        # 保存结果
        filename = f"complete_analysis_{company_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(complete_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 分析完成！总耗时: {analysis_end_time - analysis_start_time:.2f}秒")
        print(f"💾 结果已保存到: {filename}")
        
        return complete_result
    
    async def _search_company_info(self, company_name: str) -> Dict:
        """搜索公司基本信息"""
        try:
            url = "http://searchapi.eastmoney.com/api/suggest/get"
            params = {
                'input': company_name,
                'type': '14',
                'count': '10'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('QuotationCodeTable', {}).get('Data'):
                    for stock_data in data['QuotationCodeTable']['Data']:
                        stock_name = stock_data.get('Name', '')
                        if company_name in stock_name or stock_name in company_name:
                            # 获取真实行业信息
                            industry = await self._get_company_industry(stock_data.get('Code', ''))
                            result = {
                                'name': stock_name,
                                'code': stock_data.get('Code', ''),
                                'market': '深市' if stock_data.get('MktNum') == '0' else '沪市',
                                'industry': industry,
                                'source': 'eastmoney',
                                'search_success': True
                            }
                            print(f"✅ 找到公司: {result['name']} ({result['code']})")
                            return result
        except Exception as e:
            print(f"❌ 公司搜索失败: {e}")
        
        return {
            'name': company_name,
            'code': '',
            'market': '未知',
            'industry': '未知',
            'source': 'not_found',
            'search_success': False
        }
    
    async def _get_financial_data(self, stock_code: str) -> Dict:
        """获取财务数据"""
        if not stock_code:
            return {'error': '股票代码为空'}
        
        try:
            market_prefix = "1" if stock_code.startswith('6') else "0"
            url = "http://push2.eastmoney.com/api/qt/stock/get"
            params = {
                'secid': f"{market_prefix}.{stock_code}",
                'fields': 'f57,f58,f162,f163,f164,f165,f166,f167,f168,f169,f170,f46,f44,f45,f47,f260,f116,f43'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    stock_data = data['data']
                    result = {
                        'current_price': stock_data.get('f43', 0) / 100 if stock_data.get('f43') else 0,
                        'change_percent': stock_data.get('f170', 0) / 100 if stock_data.get('f170') else 0,
                        'volume': stock_data.get('f47', 0),
                        'market_cap': stock_data.get('f116', 0),
                        'pe_ratio': stock_data.get('f162', 0) / 100 if stock_data.get('f162') else 0,
                        'pb_ratio': stock_data.get('f167', 0) / 100 if stock_data.get('f167') else 0,
                        'data_source': 'eastmoney',
                        'fetch_success': True
                    }
                    print(f"✅ 获取财务数据成功: 股价{result['current_price']:.2f}元, PE{result['pe_ratio']:.2f}")
                    return result
        except Exception as e:
            print(f"❌ 财务数据获取失败: {e}")
        
        return {'error': '财务数据获取失败', 'fetch_success': False}
    
    async def _get_company_industry(self, stock_code: str) -> str:
        """获取公司行业信息"""
        if not stock_code:
            return '未知行业'
        
        try:
            market_prefix = "1" if stock_code.startswith('6') else "0"
            url = "http://push2.eastmoney.com/api/qt/stock/get"
            params = {
                'secid': f"{market_prefix}.{stock_code}",
                'fields': 'f127'  # f127是行业字段
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and data['data'].get('f127'):
                    return data['data']['f127']
        except Exception as e:
            print(f"❌ 获取行业信息失败: {e}")
        
        return '未知行业'
    
    async def _download_file(self, company_name: str, file_url: str, file_title: str) -> Optional[Dict]:
        """下载文件到指定目录"""
        try:
            # 创建公司目录
            company_dir = os.path.join('download', company_name)
            os.makedirs(company_dir, exist_ok=True)
            
            # 生成文件名
            import urllib.parse
            parsed_url = urllib.parse.urlparse(file_url)
            file_extension = os.path.splitext(parsed_url.path)[1] or '.pdf'
            
            # 清理文件标题作为文件名
            safe_title = "".join(c for c in file_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title[:50]  # 限制文件名长度
            
            if not safe_title:
                safe_title = f"document_{int(time.time())}"
            
            filename = f"{safe_title}{file_extension}"
            filepath = os.path.join(company_dir, filename)
            
            # 如果文件已存在，添加时间戳
            if os.path.exists(filepath):
                name, ext = os.path.splitext(filename)
                filename = f"{name}_{int(time.time())}{ext}"
                filepath = os.path.join(company_dir, filename)
            
            print(f"📥 开始下载文件: {file_title}")
            print(f"📁 保存路径: {filepath}")
            
            # 下载文件
            response = self.session.get(file_url, timeout=60, stream=True)
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size = os.path.getsize(filepath)
                print(f"✅ 文件下载成功: {filename} ({file_size:,} bytes)")
                
                return {
                    'filename': filename,
                    'filepath': filepath,
                    'file_size': file_size,
                    'download_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'source_url': file_url,
                    'title': file_title
                }
            else:
                print(f"❌ 下载失败，HTTP状态码: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 文件下载异常: {e}")
        
        return None
    
    async def _search_financial_news(self, company_name: str) -> Dict:
        """搜索财经新闻"""
        try:
            url = "https://api.tavily.com/search"
            headers = {
                'Content-Type': 'application/json'
            }

            current_time = date.today().year
            
            data = {
                "api_key": self.search_api_key,
                "query": f"{company_name} {current_time}年 财报 分析 投资",
                "search_depth": "basic",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": 5
            }
            
            response = self.session.post(url, json=data, headers=headers, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                news_items = []
                downloaded_files = []
                
                for item in result.get('results', []):
                    news_item = {
                        'title': item.get('title', ''),
                        'content': item.get('content', '')[:200] + '...',
                        'url': item.get('url', ''),
                        'score': item.get('score', 0)
                    }
                    
                    # 检查是否是PDF文件，如果是则下载
                    if item.get('url', '').lower().endswith('.pdf'):
                        download_result = await self._download_file(company_name, item.get('url', ''), item.get('title', ''))
                        if download_result:
                            news_item['downloaded_file'] = download_result
                            downloaded_files.append(download_result)
                    
                    news_items.append(news_item)
                
                news_result = {
                    'total_results': len(news_items),
                    'news_items': news_items,
                    'search_answer': result.get('answer', ''),
                    'downloaded_files': downloaded_files,
                    'search_success': True
                }
                print(f"✅ 搜索到 {len(news_items)} 条相关新闻")
                if downloaded_files:
                    print(f"📥 下载了 {len(downloaded_files)} 个文件")
                return news_result
                
        except Exception as e:
            print(f"❌ 新闻搜索失败: {e}")
        
        return {
            'total_results': 0,
            'news_items': [],
            'search_answer': '',
            'downloaded_files': [],
            'search_success': False
        }
    
    async def _ai_deep_analysis(self, company_info: Dict, financial_data: Dict, news_data: Dict) -> Dict:
        """AI深度分析"""
        try:
            # 构建分析提示词
            prompt = self._build_comprehensive_prompt(company_info, financial_data, news_data)
            
            print(f"📝 分析提示词: {prompt}")

            # 调用OpenAI API
            url = f"{self.openai_base_url}/chat/completions"
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": self.openai_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位资深的金融分析师，具有20年的股票投资和财务分析经验。请提供专业、客观、实用的分析建议。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 3000
            }
            
            response = self.session.post(url, json=data, headers=headers, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                analysis_content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                ai_result = {
                    'analysis_content': analysis_content,
                    'tokens_used': result.get('usage', {}).get('total_tokens', 0),
                    'model_used': self.openai_model,
                    'analysis_success': True
                }
                print(f"✅ AI分析完成，使用Token: {ai_result['tokens_used']}")
                return ai_result
            else:
                print(f"❌ AI分析失败: {response.status_code}")
                
        except Exception as e:
            print(f"❌ AI分析异常: {e}")
        
        return {
            'analysis_content': '分析失败，请检查API配置',
            'analysis_success': False
        }
    
    def _build_comprehensive_prompt(self, company_info: Dict, financial_data: Dict, news_data: Dict) -> str:
        """构建综合分析提示词"""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y年%m月%d日")
        current_year = datetime.now().year
        
        prompt = f"""请对以下公司进行全面的财务分析：

## ⚠️ 重要时间提醒 ⚠️
- 当前分析时间：{current_time}
- 当前年份：{current_year}年
- 请注意：现在是{current_year}年，不是2024年！
- 所有分析必须基于{current_year}年的时间框架
- 2024年的数据应该作为历史数据来参考
- 请使用正确的时间表述，如"基于{current_year}年的情况"、"展望{current_year+1}年"等
- 禁止使用"结合2024年数据"这样的表述，应该说"基于2024年历史数据"

## 公司基本信息
- 公司名称：{company_info.get('name', '未知')}
- 股票代码：{company_info.get('code', '未知')}
- 所属市场：{company_info.get('market', '未知')}
- 所属行业：{company_info.get('industry', '未知')}

## 当前财务数据
- 当前股价：{financial_data.get('current_price', 0):.2f} 元
- 涨跌幅：{financial_data.get('change_percent', 0):.2f}%
- 市盈率(PE)：{financial_data.get('pe_ratio', 0):.2f}
- 市净率(PB)：{financial_data.get('pb_ratio', 0):.2f}
- 总市值：{financial_data.get('market_cap', 0):,.0f} 元
- 成交量：{financial_data.get('volume', 0):,} 手

## 市场新闻和观点
{news_data.get('search_answer', '暂无相关新闻')}

## 分析要求
请从以下维度进行深入分析：

1. **估值分析**：PE、PB是否合理，与行业对比
2. **基本面分析**：财务健康状况，盈利能力
3. **技术面分析**：当前价位的技术特征
4. **风险评估**：主要风险因素识别
5. **投资建议**：明确的买入/持有/卖出建议
6. **目标价位**：合理的价格区间预测

请提供专业、客观的分析，包含具体的数据支撑和投资逻辑。
"""
        return prompt
    
    async def _generate_investment_advice(self, company_info: Dict, financial_data: Dict, ai_analysis: Dict) -> Dict:
        """生成投资建议"""
        # 基于数据生成量化建议
        pe_ratio = financial_data.get('pe_ratio', 0)
        pb_ratio = financial_data.get('pb_ratio', 0)
        change_percent = financial_data.get('change_percent', 0)
        
        # 简单的评分逻辑
        score = 0
        
        # PE评分
        if 0 < pe_ratio < 10:
            score += 30
        elif 10 <= pe_ratio < 20:
            score += 20
        elif pe_ratio >= 20:
            score += 10
        
        # PB评分
        if 0 < pb_ratio < 1:
            score += 25
        elif 1 <= pb_ratio < 2:
            score += 20
        elif pb_ratio >= 2:
            score += 10
        
        # 涨跌幅评分
        if change_percent > 0:
            score += 15
        else:
            score += 5
        
        # AI分析成功加分
        if ai_analysis.get('analysis_success'):
            score += 30
        
        # 生成建议
        if score >= 80:
            recommendation = "强烈买入"
            risk_level = "中等"
        elif score >= 60:
            recommendation = "买入"
            risk_level = "中等"
        elif score >= 40:
            recommendation = "持有"
            risk_level = "中高"
        else:
            recommendation = "谨慎"
            risk_level = "高"
        
        return {
            'recommendation': recommendation,
            'risk_level': risk_level,
            'confidence_score': score,
            'key_factors': [
                f"PE比率: {pe_ratio:.2f}",
                f"PB比率: {pb_ratio:.2f}",
                f"当日涨跌: {change_percent:.2f}%"
            ],
            'advice_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

# 测试函数
async def test_complete_analysis():
    analyzer = CompleteRealAnalyzer()
    
    # 测试公司列表
    test_companies = ["平安银行", "招商银行"]
    
    for company in test_companies:
        try:
            result = await analyzer.complete_analysis(company)
            
            print(f"\n{'='*60}")
            print(f"📊 {company} 分析摘要")
            print(f"{'='*60}")
            print(f"投资建议: {result['investment_advice']['recommendation']}")
            print(f"风险等级: {result['investment_advice']['risk_level']}")
            print(f"置信度: {result['investment_advice']['confidence_score']}/100")
            
        except Exception as e:
            print(f"❌ {company} 分析失败: {e}")
        
        # 避免请求过快
        await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(test_complete_analysis())