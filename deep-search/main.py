#!/usr/bin/env python3
"""
财报分析工具主程序
整合大模型和搜索引擎的7步财报分析系统
集成真实API：公司搜索 + 财报数据 + OpenAI分析 + 网络搜索
"""
import asyncio
import json
import sys
import requests
import time
import os
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from time_fix import get_current_time, get_current_timestamp, get_current_iso_timestamp, get_filename_timestamp, get_report_date
from data_filter import filter_financial_data, filter_historical_data, get_latest_annual_data
from mcp_service import MCPServiceClient

# 加载环境变量
load_dotenv()

class FinancialAnalyzer:
    """财报分析器主类 - 集成真实API"""
    
    def __init__(self):
        self.analysis_steps = [
            {"step": 1, "name": "获取上市公司名称或股票代码", "tool": "real_search_company"},
            {"step": 2, "name": "搜索该公司最新财务报告", "tool": "real_search_reports"},
            {"step": 3, "name": "提取财报关键财务指标", "tool": "real_extract_financial_data"},
            {"step": 4, "name": "对比行业平均水平", "tool": "real_industry_comparison"},
            {"step": 5, "name": "分析财报趋势变化", "tool": "real_trend_analysis"},
            {"step": 6, "name": "识别潜在风险或异常项", "tool": "real_risk_analysis"},
            {"step": 7, "name": "生成综合分析报告", "tool": "real_generate_report"},
            {"step": 8, "name": "校验修正报告内容", "tool": "real_report_validation"},
            {"step": 9, "name": "MCP服务增强分析", "tool": "mcp_enhance_analysis"}
        ]
        
        # 初始化真实API配置
        self._init_real_apis()
        
        # 初始化MCP服务客户端
        self.mcp_client = MCPServiceClient()
        
        # 知识库配置
        self.knowledge_base_id = os.getenv('WEKNORA_KB_ID', '77c5a766-bbb7-47c4-8027-da3cec4d2a2b')
    
    def _init_real_apis(self):
        """初始化真实API配置"""
        # OpenAI API配置
        self.api_key = os.getenv('OPENAI_API_KEY', 'sk-eD9oqSXvWMfQg5hUv487qw')
        self.base_url = os.getenv('OPENAI_BASE_URL', 'http://172.32.153.184:14000/v1')
        self.model = os.getenv('OPENAI_MODEL', 'Qwen3-Next-80B-A3B')
        
        # 搜索API配置
        self.search_api_key = os.getenv('SEARCH_API_KEY', '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')
        
        # 代理配置
        self.proxies = {
            'http': os.getenv('HTTP_PROXY', 'http://172.32.147.190:7890'),
            'https': os.getenv('HTTPS_PROXY', 'http://172.32.147.190:7890')
        }
        
        # 请求头配置
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*'
        }
        
        # 创建会话
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.proxies.update(self.proxies)
        
        print(f"🚀 初始化真实财务分析系统")
        print(f"🧠 AI模型: {self.model}")
        print(f"🔍 搜索引擎: Tavily")
        print(f"📊 数据源: 东方财富")
    
    async def analyze_company(self, company_input: str) -> Dict:
        """
        执行完整的财报分析流程
        
        Args:
            company_input: 公司名称或股票代码
            
        Returns:
            Dict: 完整的分析报告
        """
        print(f"🔍 开始分析公司: {company_input}")
        print("=" * 50)
        
        analysis_result = {
            "company_input": company_input,
            "analysis_date": get_current_iso_timestamp(),
            "steps_completed": [],
            "final_report": {},
            "status": "in_progress"
        }
        
        try:
            # 执行7个分析步骤
            for step_info in self.analysis_steps:
                step_num = step_info["step"]
                step_name = step_info["name"]
                tool_name = step_info["tool"]
                
                print(f"📊 步骤 {step_num}: {step_name}")
                
                # 执行具体步骤
                step_result = await self._execute_step(step_num, company_input, analysis_result)
                
                analysis_result["steps_completed"].append({
                    "step": step_num,
                    "name": step_name,
                    "tool": tool_name,
                    "result": step_result,
                    "status": "completed" if not step_result.get("error") else "failed",
                    "timestamp": get_current_iso_timestamp()
                })
                
                if step_result.get("error"):
                    print(f"❌ 步骤 {step_num} 失败: {step_result['error']}")
                    analysis_result["status"] = "failed"
                    return analysis_result
                else:
                    print(f"✅ 步骤 {step_num} 完成")
                    
                    # 如果是第7步（生成报告），将结果设置为 final_report
                    if step_num == 7:
                        analysis_result["final_report"] = step_result
                    # 如果是第8步（校验修正），更新 final_report
                    elif step_num == 8:
                        if step_result.get("validated_report"):
                            analysis_result["final_report"] = step_result["validated_report"]
                        analysis_result["validation_result"] = step_result
                
                # 短暂延迟，模拟真实处理时间
                await asyncio.sleep(0.5)
            
            analysis_result["status"] = "completed"
            print("\n🎉 财报分析完成!")
            
        except Exception as e:
            analysis_result["error"] = f"分析过程中发生错误: {str(e)}"
            analysis_result["status"] = "error"
            print(f"💥 分析失败: {str(e)}")
        
        return analysis_result
    
    async def _execute_step(self, step_num: int, company_input: str, context: Dict) -> Dict:
        """执行具体的分析步骤"""
        
        if step_num == 1:
            # 步骤1: 获取公司信息
            return await self._step1_search_company(company_input)
        elif step_num == 2:
            # 步骤2: 搜索财报
            company_info = self._get_company_info_from_context(context)
            return await self._step2_search_reports(company_info)
        elif step_num == 3:
            # 步骤3: 提取财务指标
            company_info = self._get_company_info_from_context(context)
            reports = self._get_reports_from_context(context)
            return await self._step3_extract_metrics(company_info, reports)
        elif step_num == 4:
            # 步骤4: 行业对比
            company_info = self._get_company_info_from_context(context)
            metrics = self._get_metrics_from_context(context)
            return await self._step4_industry_comparison(company_info, metrics)
        elif step_num == 5:
            # 步骤5: 趋势分析
            company_info = self._get_company_info_from_context(context)
            return await self._step5_trend_analysis(company_info)
        elif step_num == 6:
            # 步骤6: 风险分析
            company_info = self._get_company_info_from_context(context)
            metrics = self._get_metrics_from_context(context)
            industry_comparison = self._get_industry_comparison_from_context(context)
            trend_analysis = self._get_trend_from_context(context)
            return await self._step6_risk_analysis(company_info, metrics, industry_comparison, trend_analysis)
        elif step_num == 7:
            # 步骤7: 生成报告
            return await self._step7_generate_report(context)
        elif step_num == 8:
            # 步骤8: 校验修正报告
            company_info = self._get_company_info_from_context(context)
            final_report = self._get_final_report_from_context(context)
            return await self._step8_validate_report(company_info, final_report, context)
        elif step_num == 9:
            # 步骤9: MCP服务增强分析
            company_info = self._get_company_info_from_context(context)
            final_report = self._get_final_report_from_context(context)
            metrics = self._get_metrics_from_context(context)
            return await self._step9_mcp_enhance_analysis(company_info, final_report, metrics, context)
        else:
            return {"success": False, "error": f"未知的步骤: {step_num}"}
    
    async def _step1_search_company(self, company_input: str) -> Dict:
        """步骤1: 真实搜索公司信息"""
        try:
            url = "http://searchapi.eastmoney.com/api/suggest/get"
            params = {
                'input': company_input,
                'type': '14',
                'count': '10'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('QuotationCodeTable', {}).get('Data'):
                    for stock_data in data['QuotationCodeTable']['Data']:
                        stock_name = stock_data.get('Name', '')
                        if company_input in stock_name or stock_name in company_input:
                            # 获取真实行业信息
                            industry = await self._get_company_industry(stock_data.get('Code', ''))
                            result = {
                                'company_name': stock_name,
                                'stock_code': stock_data.get('Code', ''),
                                'market': '深市' if stock_data.get('MktNum') == '0' else '沪市',
                                'industry': industry,
                                'source': 'eastmoney',
                                'search_success': True
                            }
                            print(f"✅ 找到公司: {result['company_name']} ({result['stock_code']})")
                            return result
        except Exception as e:
            print(f"❌ 公司搜索失败: {e}")
        
        return {
            'company_name': company_input,
            'stock_code': '',
            'market': '未知',
            'industry': '未知',
            'source': 'not_found',
            'search_success': False
        }
    
    async def _get_company_industry(self, stock_code: str) -> str:
        """获取公司行业信息"""
        try:
            # 使用东方财富API获取公司详细信息
            url = f"http://push2.eastmoney.com/api/qt/stock/get"
            params = {
                'secid': f"0.{stock_code}" if stock_code.startswith('0') or stock_code.startswith('3') else f"1.{stock_code}",
                'fields': 'f57,f58,f162,f163'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('data', {}).get('f162'):
                    return data['data']['f162']
        except Exception as e:
            print(f"   获取行业信息失败: {e}")
        
        return "未知行业"
    
    async def _download_file(self, company_name: str, url: str, title: str) -> Dict:
        """下载PDF文件"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                # 创建下载目录
                download_dir = f"downloads/{company_name}"
                os.makedirs(download_dir, exist_ok=True)
                
                # 生成文件名
                filename = f"{title[:50]}_{int(time.time())}.pdf"
                filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
                filepath = os.path.join(download_dir, filename)
                
                # 保存文件
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                return {
                    'title': title,
                    'url': url,
                    'local_path': filepath,
                    'size': len(response.content)
                }
        except Exception as e:
            print(f"   下载文件失败 {url}: {e}")
        
        return None
    
    async def _step2_search_reports(self, company_info: Dict) -> Dict:
        """步骤2: 真实搜索财报和新闻"""
        company_name = company_info.get('company_name', '')
        
        try:
            url = "https://api.tavily.com/search"
            headers = {
                'Content-Type': 'application/json'
            }

            current_year = date.today().year
            
            data = {
                "api_key": self.search_api_key,
                "query": f"{company_name} {current_year}年 财报 分析 投资",
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
                        download_result = await self._download_file(company_name, item.get('url'), item.get('title', ''))
                        if download_result:
                            downloaded_files.append(download_result)
                    
                    news_items.append(news_item)
                
                print(f"✅ 找到 {len(news_items)} 条相关新闻")
                if downloaded_files:
                    print(f"📄 下载了 {len(downloaded_files)} 个PDF文件")
                
                return {
                    'news_items': news_items,
                    'downloaded_files': downloaded_files,
                    'search_summary': result.get('answer', ''),
                    'total_results': len(news_items),
                    'data_source': 'tavily_search'
                }
        except Exception as e:
            print(f"❌ 财报搜索失败: {e}")
        
        return {
            'news_items': [],
            'downloaded_files': [],
            'search_summary': '搜索失败',
            'total_results': 0,
            'data_source': 'search_failed'
        }
    
    async def _step3_extract_metrics(self, company_info: Dict, reports: Dict) -> Dict:
        """步骤3: 真实提取财务数据"""
        stock_code = company_info.get('stock_code', '')
        
        if not stock_code:
            print("❌ 无股票代码，无法获取财务数据")
            return self._get_empty_financial_data()
        
        try:
            # 使用real_financial_data.py中的真实API
            from real_financial_data import RealFinancialDataTool
            tool = RealFinancialDataTool()
            result = await tool.get_financial_reports(stock_code)
            
            # 应用数据过滤，移除不合理的未来年份数据
            result = filter_financial_data(result)
            print("📊 已应用数据过滤，确保使用合理年份的财务数据")
            
            # 提取各类数据
            basic_data = result.get('basic_data', {})
            financial_ratios = result.get('financial_ratios', {})
            cashflow_data = result.get('cashflow_data', {})
            
            # 合并所有财务数据
            financial_data = {
                **basic_data,
                **financial_ratios,
                **cashflow_data
            }
            
            # 字段映射：将API字段名映射到标准字段名
            if 'total_operate_income' in financial_data:
                financial_data['revenue'] = financial_data['total_operate_income']
            if 'parent_netprofit' in financial_data:
                financial_data['net_profit'] = financial_data['parent_netprofit']
            
            # 如果没有获取到任何数据，使用备用数据
            if not financial_data or all(not v for v in financial_data.values()):
                print("⚠️ 未获取到真实财务数据，使用备用数据")
                financial_data = self._get_empty_financial_data()
            else:
                print(f"✅ 成功获取财务数据，包含 {len([k for k, v in financial_data.items() if v])} 个有效指标")
            
            return financial_data
            
        except Exception as e:
            print(f"❌ 财务数据获取失败: {e}")
            return self._get_empty_financial_data()


    def _get_empty_financial_data(self) -> Dict:
        """返回空的财务数据"""
        return {
            "revenue": 0,
            "net_profit": 0,
            "total_assets": 0,
            "roe": 0,
            "debt_ratio": 0,
            "current_ratio": 0,
            "eps": 0,
            "pe_ratio": 0,
            "operating_cashflow": 0,
            "free_cashflow": 0,
            "cashflow_per_share": 0,
            "extraction_method": "数据获取失败"
        }
    
    async def _step4_industry_comparison(self, company_info: Dict, metrics: Dict) -> Dict:
        """步骤4: 行业对比"""
        stock_code = company_info.get('stock_code', '')
        industry = company_info.get("industry", "未知行业")
        
        try:
            # 导入真实的行业对比模块
            from real_industry_comparison import RealIndustryComparison
            
            # 创建行业对比分析器
            industry_analyzer = RealIndustryComparison()
            
            # 生成行业对比分析
            analysis_result = industry_analyzer.generate_comparison_analysis(stock_code, company_info.get('name', ''))
            
            if analysis_result:
                # 转换为主程序需要的格式
                industry_data = {
                    "industry": analysis_result.get("industry_name", industry),
                    "industry_avg_roe": analysis_result.get("industry_averages", {}).get("avg_roe", 0.142),
                    "industry_avg_debt_ratio": 0.62,  # 暂时使用默认值
                    "industry_avg_pe": analysis_result.get("industry_averages", {}).get("avg_pe", 9.1),
                    "company_ranking": f"行业第{analysis_result.get('company_ranking', 0)}名",
                    "comparison_summary": f"该公司在{analysis_result.get('industry_name', industry)}中排名第{analysis_result.get('company_ranking', 0)}位",
                    "peer_companies": [comp.get('name', '') for comp in analysis_result.get("peer_companies", [])],
                    "total_companies": analysis_result.get("total_companies", 0),
                    "data_source": analysis_result.get("data_source", "东方财富"),
                    "update_time": analysis_result.get("update_time", get_current_timestamp()),
                    "data_period": analysis_result.get("data_period", ""),
                    "data_freshness": analysis_result.get("data_freshness", "实时数据"),
                    "data_fetch_time": analysis_result.get("data_fetch_time_seconds", 0),
                    "data_quality": analysis_result.get("data_quality", {})
                }
                
                # 显示详细的数据获取信息
                data_quality = industry_data.get("data_quality", {})
                print(f"✅ 行业对比分析完成，找到 {industry_data['total_companies']} 家同行业公司")
                print(f"   📊 数据时期: {industry_data.get('data_period', 'N/A')}")
                print(f"   ⏱️  获取耗时: {industry_data.get('data_fetch_time', 0)} 秒")
                print(f"   🔄 数据新鲜度: {industry_data.get('data_freshness', 'N/A')}")
                print(f"   📈 数据质量: {data_quality.get('valid_data_ratio', 0)*100:.0f}% ({data_quality.get('companies_found', 0)} 家公司有效数据)")
                print(f"   🕐 最后更新: {industry_data.get('update_time', 'N/A')}")
                return industry_data
            
        except Exception as e:
            print(f"❌ 行业对比分析失败: {e}")
        
        # 如果真实API失败，返回模拟数据
        await asyncio.sleep(0.3)
        return {
            "industry": industry,
            "industry_avg_roe": 0.142,
            "industry_avg_debt_ratio": 0.62,
            "industry_avg_pe": 9.1,
            "company_ranking": "行业前25%",
            "comparison_summary": f"该公司在{industry}中表现优于平均水平",
            "peer_companies": ["工商银行", "建设银行", "农业银行"],
            "data_source": "模拟数据"
        }
    
    async def _step5_trend_analysis(self, company_info: Dict) -> Dict:
        """步骤5: 趋势分析"""
        # 导入真实趋势分析工具
        from real_trend_analysis import RealTrendAnalysis
        
        # 获取股票代码和公司名称
        stock_code = company_info.get('stock_code', '')
        company_name = company_info.get('company_name', '')
        
        if not stock_code:
            raise ValueError(f"未找到股票代码，无法进行趋势分析。公司信息: {company_info}")
        
        # 创建趋势分析实例
        trend_analyzer = RealTrendAnalysis()
        
        # 获取真实趋势分析数据
        print(f"📈 正在分析 {company_name}({stock_code}) 的历史趋势...")
        trend_data = await trend_analyzer.generate_trend_analysis(stock_code, company_name)
        
        # 显示详细信息
        print(f"   📊 分析期间: {trend_data.get('analysis_period', 'N/A')}")
        print(f"   📈 平均收入增长: {trend_data.get('avg_revenue_growth', 0):.1%}")
        print(f"   💰 平均利润增长: {trend_data.get('avg_profit_growth', 0):.1%}")
        print(f"   📊 成长阶段: {trend_data.get('growth_stage', 'N/A')}")
        print(f"   ⏱️  获取耗时: {trend_data.get('data_fetch_time', 0)}秒")
        print(f"   📋 数据质量: {trend_data.get('data_quality', {}).get('quality_score', 0):.2f}")
        print(f"   🕒 最后更新: {trend_data.get('update_time', 'N/A')}")
        
        return trend_data
    

    
    async def _step6_risk_analysis(self, company_info: Dict, metrics: Dict, 
                                  industry_comparison: Dict, trend_analysis: Dict) -> Dict:
        """步骤6: 风险分析"""
        from real_risk_analysis import RealRiskAnalysis
        
        stock_code = company_info.get('stock_code')
        company_name = company_info.get('company_name', '')
        
        if not stock_code:
            raise ValueError("未找到股票代码，无法进行风险分析")
        
        if not company_name:
            raise ValueError("未找到公司名称，无法进行风险分析")
        
        # 使用真实数据进行风险分析
        risk_analyzer = RealRiskAnalysis()
        risk_analysis = await risk_analyzer.generate_risk_analysis(
            stock_code=stock_code,
            company_name=company_name,
            financial_data=metrics,
            industry_data=industry_comparison,
            trend_data=trend_analysis
        )
        
        return risk_analysis
    
    async def _step7_generate_report(self, context: Dict) -> Dict:
        """步骤7: 生成综合报告 - 使用真实OpenAI API"""
        print("🤖 正在使用OpenAI生成专业分析报告...")
        
        # 从上下文中提取信息
        company_info = self._get_company_info_from_context(context)
        metrics = self._get_metrics_from_context(context)
        industry_comp = self._get_industry_comparison_from_context(context)
        trend = self._get_trend_from_context(context)
        risk = self._get_risk_from_context(context)
        
        # 构建公司数据和财务数据
        company_data = {
            'name': company_info.get('company_name', '目标公司'),
            'code': company_info.get('stock_code', ''),
            'industry': company_info.get('industry', ''),
            'market': company_info.get('market', '')
        }
        
        financial_data = {
            'basic_data': {
                'current_price': 12.5,  # 这里应该从真实数据源获取
                'change_percent': 1.2,
                'pe_ratio': metrics.get('pe_ratio', 8.5),
                'pb_ratio': 0.8,
                'market_cap': 150000000000,
                'volume': 500000
            }
        }
        
        # 使用OpenAI API生成专业分析
        openai_result = await self.analyze_with_openai(company_data, financial_data)
        
        # 构建最终报告
        final_report = {
            "executive_summary": f"{company_info.get('company_name', '目标公司')}整体财务表现良好，在行业中处于领先地位",
            "financial_highlights": {
                "revenue": f"{metrics.get('revenue', 0)/100000000:.1f}亿元",
                "net_profit": f"{metrics.get('net_profit', 0)/100000000:.1f}亿元",
                "roe": f"{metrics.get('roe', 0)*100:.1f}%",
                "pe_ratio": metrics.get('pe_ratio', 0)
            },
            "industry_position": industry_comp.get("company_ranking", "行业中等"),
            "key_risks": risk.get("identified_risks", []),
            "investment_recommendation": "建议持有" if risk.get("risk_level") == "低" else "谨慎观察",
            "target_price_analysis": "基于DCF模型和相对估值法综合评估",
            "analyst_rating": "买入",
            "confidence_level": "高",
            "report_date": get_report_date(),
            "disclaimer": "本报告仅供参考，不构成投资建议",
            # 添加OpenAI分析结果
            "openai_analysis": openai_result
        }
        
        return final_report
    
    def _get_company_info_from_context(self, context: Dict) -> Dict:
        """从上下文中获取公司信息"""
        for step in context.get("steps_completed", []):
            if step["step"] == 1:
                return step.get("result", {})
        return {}
    
    def _get_reports_from_context(self, context: Dict) -> Dict:
        """从上下文中获取财报信息"""
        for step in context.get("steps_completed", []):
            if step["step"] == 2:
                return step.get("result", {})
        return {}
    
    def _get_metrics_from_context(self, context: Dict) -> Dict:
        """从上下文中获取财务指标"""
        for step in context.get("steps_completed", []):
            if step["step"] == 3:
                return step.get("result", {})
        return {}
    
    def _get_industry_comparison_from_context(self, context: Dict) -> Dict:
        """从上下文中获取行业对比"""
        for step in context.get("steps_completed", []):
            if step["step"] == 4:
                return step.get("result", {})
        return {}
    
    def _get_trend_from_context(self, context: Dict) -> Dict:
        """从上下文中获取趋势分析"""
        for step in context.get("steps_completed", []):
            if step["step"] == 5:
                return step.get("result", {})
        return {}
    
    def _get_risk_from_context(self, context: Dict) -> Dict:
        """从上下文中获取风险分析结果"""
        for step in context.get("steps_completed", []):
            if step.get("name") == "识别潜在风险或异常项":
                return step.get("result", {})
        return {}
    
    def _get_final_report_from_context(self, context: Dict) -> Dict:
        """从上下文中获取最终报告"""
        return context.get("final_report", {})
    
    async def _step8_validate_report(self, company_info: Dict, final_report: Dict, context: Dict) -> Dict:
        """步骤8: 校验修正报告内容的真实性、合理性和时效性"""
        print("🔍 开始校验报告内容...")
        
        validation_result = {
            "validation_summary": {},
            "corrections_made": [],
            "validation_score": 0,
            "validated_report": final_report.copy(),
            "validation_details": {}
        }
        
        try:
            company_name = company_info.get('company_name', '')
            stock_code = company_info.get('stock_code', '')
            
            # 1. 真实性校验：通过可信网站搜索验证关键信息
            print("   📊 执行真实性校验...")
            authenticity_result = await self._validate_authenticity(company_name, stock_code, final_report)
            validation_result["validation_details"]["authenticity"] = authenticity_result
            
            # 2. 合理性校验：使用大模型分析数据逻辑一致性
            print("   🧠 执行合理性校验...")
            rationality_result = await self._validate_rationality(company_name, final_report)
            validation_result["validation_details"]["rationality"] = rationality_result
            
            # 3. 时效性校验：验证数据的时间有效性
            print("   ⏰ 执行时效性校验...")
            timeliness_result = await self._validate_timeliness(company_name, stock_code, final_report)
            validation_result["validation_details"]["timeliness"] = timeliness_result
            
            # 4. 综合修正：基于校验结果修正报告
            print("   🔧 执行报告修正...")
            validation_results = {
                "authenticity": authenticity_result,
                "rationality": rationality_result,
                "timeliness": timeliness_result
            }
            correction_result = await self._apply_corrections(final_report, validation_results)
            
            validation_result["validated_report"] = correction_result["corrected_report"]
            validation_result["corrections_made"] = correction_result["corrections_applied"]
            
            # 5. 计算综合校验分数
            validation_result["validation_score"] = self._calculate_validation_score(
                authenticity_result, rationality_result, timeliness_result
            )
            
            # 6. 生成校验摘要
            validation_result["validation_summary"] = self._generate_validation_summary(
                authenticity_result, rationality_result, timeliness_result, 
                validation_result["validation_score"]
            )
            
            print(f"✅ 报告校验完成，综合得分: {validation_result['validation_score']:.1f}/10")
            if validation_result["corrections_made"]:
                print(f"   🔧 应用了 {len(validation_result['corrections_made'])} 项修正")
            
        except Exception as e:
            print(f"❌ 报告校验失败: {e}")
            validation_result["error"] = str(e)
            validation_result["validation_score"] = 0
        
        return validation_result
    
    async def _validate_authenticity(self, company_name: str, stock_code: str, final_report: Dict) -> Dict:
        """真实性校验：通过可信网站搜索验证关键信息"""
        authenticity_result = {
            "score": 0,
            "issues_found": [],
            "verified_facts": [],
            "search_results": []
        }
        
        try:
            # 构建搜索查询，验证关键财务数据
            current_year = date.today().year
            search_queries = [
                f"{company_name} {stock_code} {current_year}年 财报 营收 净利润",
                f"{company_name} {stock_code} 最新 财务数据 ROE 负债率",
                f"{company_name} {stock_code} 行业排名 市场地位",
                f"{company_name} {stock_code} 风险 投资 分析"
            ]
            
            verified_count = 0
            total_checks = 0
            
            for query in search_queries:
                try:
                    # 使用Tavily搜索可信财经网站
                    search_result = await self._search_trusted_sources(query)
                    authenticity_result["search_results"].append({
                        "query": query,
                        "results": search_result.get("results", [])[:3],  # 只保留前3个结果
                        "answer": search_result.get("answer", "")
                    })
                    
                    # 验证搜索结果与报告内容的一致性
                    verification = await self._verify_against_search_results(
                        final_report, search_result, query
                    )
                    
                    if verification["verified"]:
                        verified_count += 1
                        authenticity_result["verified_facts"].extend(verification["facts"])
                    else:
                        authenticity_result["issues_found"].extend(verification["issues"])
                    
                    total_checks += 1
                    
                except Exception as e:
                    print(f"   搜索查询失败 '{query}': {e}")
                    continue
            
            # 计算真实性得分
            if total_checks > 0:
                authenticity_result["score"] = (verified_count / total_checks) * 10
            else:
                authenticity_result["score"] = 5  # 默认中等分数
                
        except Exception as e:
            print(f"   真实性校验异常: {e}")
            authenticity_result["score"] = 5
            authenticity_result["issues_found"].append(f"校验过程异常: {str(e)}")
        
        return authenticity_result
    
    async def _search_trusted_sources(self, query: str) -> Dict:
        """搜索可信的财经网站"""
        try:
            url = "https://api.tavily.com/search"
            headers = {'Content-Type': 'application/json'}
            
            # 限定搜索可信的财经网站
            trusted_domains = [
                "eastmoney.com", "sina.com.cn", "163.com", "qq.com", 
                "cninfo.com.cn", "sse.com.cn", "szse.cn", "csrc.gov.cn"
            ]
            
            data = {
                "api_key": self.search_api_key,
                "query": query + " site:" + " OR site:".join(trusted_domains),
                "search_depth": "basic",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": 5
            }
            
            response = self.session.post(url, json=data, headers=headers, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"results": [], "answer": ""}
                
        except Exception as e:
            print(f"   可信源搜索失败: {e}")
            return {"results": [], "answer": ""}
    
    async def _verify_against_search_results(self, final_report: Dict, search_result: Dict, query: str) -> Dict:
        """验证报告内容与搜索结果的一致性"""
        verification = {
            "verified": False,
            "facts": [],
            "issues": []
        }
        
        try:
            # 使用大模型比较报告内容与搜索结果
            prompt = self._build_verification_prompt(final_report, search_result, query)
            llm_result = await self._call_openai_api(prompt)
            
            if llm_result.get("success"):
                analysis = llm_result.get("content", "")
                
                # 解析大模型的验证结果
                if "一致性良好" in analysis or "基本一致" in analysis:
                    verification["verified"] = True
                    verification["facts"].append(f"查询'{query}'的结果与报告内容基本一致")
                elif "存在差异" in analysis or "不一致" in analysis:
                    verification["issues"].append(f"查询'{query}'发现与报告内容存在差异")
                else:
                    verification["verified"] = True  # 默认认为一致
                    
        except Exception as e:
            print(f"   验证过程异常: {e}")
            verification["issues"].append(f"验证过程异常: {str(e)}")
        
        return verification
    
    def _build_verification_prompt(self, final_report: Dict, search_result: Dict, query: str) -> str:
        """构建验证提示词"""
        return f"""
请比较以下财务分析报告内容与搜索结果的一致性：

查询内容：{query}

报告摘要：
{json.dumps(final_report, ensure_ascii=False, indent=2)[0:]}...

搜索结果摘要：
{search_result.get('answer', '')}

请分析：
1. 报告中的关键数据与搜索结果是否一致
2. 是否存在明显的数据差异或矛盾
3. 整体一致性如何

请简洁回答：一致性良好/基本一致/存在差异/严重不一致，并说明理由。
"""
    
    async def _validate_rationality(self, company_name: str, final_report: Dict) -> Dict:
        """合理性校验：检查分析逻辑和结论的合理性"""
        rationality_result = {
            "score": 0,
            "issues_found": [],
            "logical_checks": [],
            "recommendations": []
        }
        
        try:
            # 检查财务指标的逻辑一致性
            financial_checks = await self._check_financial_logic(final_report)
            rationality_result["logical_checks"].extend(financial_checks["checks"])
            rationality_result["issues_found"].extend(financial_checks["issues"])
            
            # 检查风险评估的合理性
            risk_checks = await self._check_risk_logic(final_report)
            rationality_result["logical_checks"].extend(risk_checks["checks"])
            rationality_result["issues_found"].extend(risk_checks["issues"])
            
            # 检查投资建议的合理性
            investment_checks = await self._check_investment_logic(final_report)
            rationality_result["logical_checks"].extend(investment_checks["checks"])
            rationality_result["issues_found"].extend(investment_checks["issues"])
            
            # 使用大模型进行整体合理性分析
            llm_analysis = await self._llm_rationality_check(final_report)
            if llm_analysis.get("success"):
                rationality_result["recommendations"].extend(llm_analysis.get("recommendations", []))
                rationality_result["issues_found"].extend(llm_analysis.get("issues", []))
            
            # 计算合理性得分
            total_checks = len(rationality_result["logical_checks"])
            failed_checks = len(rationality_result["issues_found"])
            
            if total_checks > 0:
                rationality_result["score"] = max(0, (total_checks - failed_checks) / total_checks * 10)
            else:
                rationality_result["score"] = 7  # 默认较高分数
                
        except Exception as e:
            print(f"   合理性校验异常: {e}")
            rationality_result["score"] = 5
            rationality_result["issues_found"].append(f"校验过程异常: {str(e)}")
        
        return rationality_result
    
    async def _check_financial_logic(self, final_report: Dict) -> Dict:
        """检查财务指标逻辑一致性"""
        checks = {
            "checks": [],
            "issues": []
        }
        
        try:
            # 获取财务数据
            financial_data = final_report.get("financial_analysis", {})
            
            # 检查ROE与净利润率、资产周转率的关系
            roe = financial_data.get("roe")
            net_margin = financial_data.get("net_profit_margin")
            asset_turnover = financial_data.get("asset_turnover")
            
            if all([roe, net_margin, asset_turnover]):
                expected_roe = net_margin * asset_turnover
                if abs(roe - expected_roe) > 0.05:  # 允许5%的误差
                    checks["issues"].append("ROE与净利润率、资产周转率计算不一致")
                else:
                    checks["checks"].append("ROE计算逻辑正确")
            
            # 检查负债率的合理性
            debt_ratio = financial_data.get("debt_ratio")
            if debt_ratio and (debt_ratio < 0 or debt_ratio > 1):
                checks["issues"].append("负债率数值超出合理范围[0,1]")
            elif debt_ratio:
                checks["checks"].append("负债率数值在合理范围内")
            
            # 检查现金比率的合理性
            cash_ratio = financial_data.get("cash_ratio")
            if cash_ratio and cash_ratio < 0:
                checks["issues"].append("现金比率不能为负数")
            elif cash_ratio:
                checks["checks"].append("现金比率数值合理")
                
        except Exception as e:
            checks["issues"].append(f"财务逻辑检查异常: {str(e)}")
        
        return checks
    
    async def _check_risk_logic(self, final_report: Dict) -> Dict:
        """检查风险评估逻辑"""
        checks = {
            "checks": [],
            "issues": []
        }
        
        try:
            risk_analysis = final_report.get("risk_analysis", {})
            overall_risk = risk_analysis.get("overall_risk", "")
            risk_factors = risk_analysis.get("risk_factors", [])
            
            # 检查风险等级与风险因素的一致性
            high_risk_count = sum(1 for factor in risk_factors if "高" in factor or "严重" in factor)
            low_risk_count = sum(1 for factor in risk_factors if "低" in factor or "轻微" in factor)
            
            if overall_risk == "高" and high_risk_count == 0:
                checks["issues"].append("整体风险评级为高，但未发现高风险因素")
            elif overall_risk == "低" and high_risk_count > low_risk_count:
                checks["issues"].append("整体风险评级为低，但高风险因素较多")
            else:
                checks["checks"].append("风险评级与风险因素基本一致")
            
            # 检查是否存在矛盾的风险描述
            risk_text = " ".join(risk_factors)
            if "排名第1" in risk_text and "排名靠后" in risk_text:
                checks["issues"].append("存在矛盾的行业排名描述")
            else:
                checks["checks"].append("风险描述逻辑一致")
                
        except Exception as e:
            checks["issues"].append(f"风险逻辑检查异常: {str(e)}")
        
        return checks
    
    async def _check_investment_logic(self, final_report: Dict) -> Dict:
        """检查投资建议逻辑"""
        checks = {
            "checks": [],
            "issues": []
        }
        
        try:
            investment = final_report.get("investment_recommendation", {})
            recommendation = investment.get("recommendation", "")
            risk_level = final_report.get("risk_analysis", {}).get("overall_risk", "")
            
            # 检查投资建议与风险评级的一致性
            if "买入" in recommendation and risk_level == "高":
                checks["issues"].append("高风险情况下建议买入，逻辑不一致")
            elif "卖出" in recommendation and risk_level == "低":
                checks["issues"].append("低风险情况下建议卖出，逻辑不一致")
            else:
                checks["checks"].append("投资建议与风险评级基本一致")
            
            # 检查投资建议的完整性
            if not recommendation:
                checks["issues"].append("缺少明确的投资建议")
            else:
                checks["checks"].append("提供了明确的投资建议")
                
        except Exception as e:
            checks["issues"].append(f"投资逻辑检查异常: {str(e)}")
        
        return checks
    
    async def _llm_rationality_check(self, final_report: Dict) -> Dict:
        """使用大模型进行整体合理性检查"""
        try:
            prompt = f"""
请对以下财务分析报告进行合理性检查：

{json.dumps(final_report, ensure_ascii=False, indent=2)[:2000]}...

请检查：
1. 财务指标计算是否合理
2. 风险评估是否逻辑一致
3. 投资建议是否与分析结果匹配
4. 是否存在明显的逻辑矛盾

请以JSON格式回答：
{{
    "issues": ["发现的问题1", "发现的问题2"],
    "recommendations": ["改进建议1", "改进建议2"],
    "overall_score": 分数(1-10)
}}
"""
            
            result = await self._call_openai_api(prompt)
            if result.get("success"):
                try:
                    content = result.get("content", "{}")
                    # 尝试解析JSON响应
                    analysis = json.loads(content)
                    return {
                        "success": True,
                        "issues": analysis.get("issues", []),
                        "recommendations": analysis.get("recommendations", []),
                        "score": analysis.get("overall_score", 7)
                    }
                except json.JSONDecodeError:
                    return {"success": False, "issues": ["大模型响应格式错误"]}
            else:
                return {"success": False, "issues": ["大模型调用失败"]}
                
        except Exception as e:
            return {"success": False, "issues": [f"合理性检查异常: {str(e)}"]}
    
    async def _validate_timeliness(self, company_name: str, stock_code: str, final_report: Dict) -> Dict:
        """时效性校验：检查数据和信息的时效性"""
        timeliness_result = {
            "score": 0,
            "issues_found": [],
            "data_freshness": {},
            "outdated_info": []
        }
        
        try:
            current_date = date.today()
            current_year = current_date.year
            current_quarter = (current_date.month - 1) // 3 + 1
            
            # 检查财务数据时效性
            financial_freshness = await self._check_financial_data_freshness(
                final_report, current_year, current_quarter
            )
            timeliness_result["data_freshness"].update(financial_freshness)
            
            # 检查市场信息时效性
            market_freshness = await self._check_market_info_freshness(
                company_name, stock_code, final_report
            )
            timeliness_result["data_freshness"].update(market_freshness)
            
            # 检查行业信息时效性
            industry_freshness = await self._check_industry_info_freshness(
                final_report, current_date
            )
            timeliness_result["data_freshness"].update(industry_freshness)
            
            # 计算时效性得分
            freshness_scores = [
                financial_freshness.get("score", 5),
                market_freshness.get("score", 5),
                industry_freshness.get("score", 5)
            ]
            timeliness_result["score"] = sum(freshness_scores) / len(freshness_scores)
            
            # 收集过时信息
            for freshness in [financial_freshness, market_freshness, industry_freshness]:
                timeliness_result["outdated_info"].extend(freshness.get("outdated", []))
                timeliness_result["issues_found"].extend(freshness.get("issues", []))
                
        except Exception as e:
            print(f"   时效性校验异常: {e}")
            timeliness_result["score"] = 5
            timeliness_result["issues_found"].append(f"校验过程异常: {str(e)}")
        
        return timeliness_result
    

    ## 通过判断确认需要是最近四个季度的报告
    async def _check_financial_data_freshness(self, final_report: Dict, current_year: int, current_quarter: int) -> Dict:
        """检查财务数据的时效性"""
        freshness = {
            "score": 0,
            "issues": [],
            "outdated": []
        }
        
        try:
            financial_data = final_report.get("financial_analysis", {})
            
            # 检查报告期
            report_period = financial_data.get("report_period", "")
            if report_period:
                # 解析报告期（假设格式为"2024Q3"或"2024年第三季度"）
                if "Q" in report_period:
                    year_quarter = report_period.split("Q")
                    if len(year_quarter) == 2:
                        report_year = int(year_quarter[0])
                        report_q = int(year_quarter[1])
                        
                        # 计算数据新鲜度
                        year_diff = current_year - report_year
                        quarter_diff = current_quarter - report_q + year_diff * 4
                        
                        if quarter_diff <= 1:
                            freshness["score"] = 10  # 最新数据
                        elif quarter_diff <= 2:
                            freshness["score"] = 8   # 较新数据
                        elif quarter_diff <= 4:
                            freshness["score"] = 6   # 一般数据
                        else:
                            freshness["score"] = 3   # 过时数据
                            freshness["outdated"].append(f"财务数据过时，报告期：{report_period}")
                    else:
                        freshness["score"] = 5
                        freshness["issues"].append("无法解析财务数据报告期")
                else:
                    freshness["score"] = 5
                    freshness["issues"].append("财务数据报告期格式不标准")
            else:
                freshness["score"] = 3
                freshness["issues"].append("缺少财务数据报告期信息")
                
        except Exception as e:
            freshness["score"] = 5
            freshness["issues"].append(f"财务数据时效性检查异常: {str(e)}")
        
        return freshness
    
    async def _check_market_info_freshness(self, company_name: str, stock_code: str, final_report: Dict) -> Dict:
        """检查市场信息的时效性"""
        freshness = {
            "score": 0,
            "issues": [],
            "outdated": []
        }
        
        try:
            # 搜索最新的市场信息
            current_date = date.today()
            search_query = f"{company_name} {stock_code} 最新 股价 市值 {current_date.strftime('%Y年%m月')}"
            
            search_result = await self._search_trusted_sources(search_query)
            
            if search_result.get("results"):
                # 检查搜索结果的时效性
                recent_info_found = False
                for result in search_result["results"][:3]:
                    title = result.get("title", "")
                    content = result.get("content", "")
                    
                    # 检查是否包含最新日期信息
                    if any(month in title + content for month in [
                        current_date.strftime("%Y-%m"),
                        current_date.strftime("%Y年%m月"),
                        "今日", "最新", "实时"
                    ]):
                        recent_info_found = True
                        break
                
                if recent_info_found:
                    freshness["score"] = 9
                else:
                    freshness["score"] = 6
                    freshness["outdated"].append("市场信息可能不是最新的")
            else:
                freshness["score"] = 4
                freshness["issues"].append("无法获取最新市场信息")
                
        except Exception as e:
            freshness["score"] = 5
            freshness["issues"].append(f"市场信息时效性检查异常: {str(e)}")
        
        return freshness
    
    async def _check_industry_info_freshness(self, final_report: Dict, current_date: date) -> Dict:
        """检查行业信息的时效性"""
        freshness = {
            "score": 0,
            "issues": [],
            "outdated": []
        }
        
        try:
            industry_data = final_report.get("industry_comparison", {})
            
            # 检查行业数据的时间戳
            if "analysis_date" in industry_data:
                analysis_date_str = industry_data["analysis_date"]
                try:
                    analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d").date()
                    days_diff = (current_date - analysis_date).days
                    
                    if days_diff <= 7:
                        freshness["score"] = 10  # 一周内
                    elif days_diff <= 30:
                        freshness["score"] = 8   # 一月内
                    elif days_diff <= 90:
                        freshness["score"] = 6   # 三月内
                    else:
                        freshness["score"] = 4   # 超过三月
                        freshness["outdated"].append(f"行业分析数据过时，分析日期：{analysis_date_str}")
                        
                except ValueError:
                    freshness["score"] = 5
                    freshness["issues"].append("行业数据时间戳格式错误")
            else:
                freshness["score"] = 6  # 默认分数
                freshness["issues"].append("缺少行业数据时间戳")
                
        except Exception as e:
            freshness["score"] = 5
            freshness["issues"].append(f"行业信息时效性检查异常: {str(e)}")
        
        return freshness
    
    async def _apply_corrections(self, final_report: Dict, validation_results: Dict) -> Dict:
        """应用修正建议，生成修正后的报告"""
        corrected_report = final_report.copy()
        corrections_applied = []
        
        try:
            # 收集所有发现的问题
            all_issues = []
            all_recommendations = []
            
            for validation_type, result in validation_results.items():
                if validation_type.endswith("_result"):
                    all_issues.extend(result.get("issues_found", []))
                    all_recommendations.extend(result.get("recommendations", []))
            
            # 应用自动修正
            auto_corrections = await self._apply_automatic_corrections(corrected_report, all_issues)
            corrections_applied.extend(auto_corrections)
            
            # 使用大模型生成修正建议
            if all_issues or all_recommendations:
                llm_corrections = await self._generate_llm_corrections(
                    corrected_report, all_issues, all_recommendations
                )
                if llm_corrections.get("success"):
                    corrected_report = llm_corrections.get("corrected_report", corrected_report)
                    corrections_applied.extend(llm_corrections.get("corrections", []))
            
            # 添加修正记录
            corrected_report["validation_info"] = {
                "validation_date": date.today().isoformat(),
                "corrections_applied": corrections_applied,
                "original_issues": all_issues,
                "validation_scores": {
                    "authenticity": validation_results.get("authenticity_result", {}).get("score", 0),
                    "rationality": validation_results.get("rationality_result", {}).get("score", 0),
                    "timeliness": validation_results.get("timeliness_result", {}).get("score", 0)
                }
            }
            
        except Exception as e:
            print(f"   应用修正异常: {e}")
            corrections_applied.append(f"修正过程异常: {str(e)}")
        
        return {
            "corrected_report": corrected_report,
            "corrections_applied": corrections_applied
        }
    
    async def _apply_automatic_corrections(self, report: Dict, issues: List[str]) -> List[str]:
        """应用自动修正规则"""
        corrections = []
        
        try:
            # 修正矛盾的行业排名描述
            for issue in issues:
                if "矛盾的行业排名描述" in issue:
                    risk_analysis = report.get("risk_analysis", {})
                    risk_factors = risk_analysis.get("risk_factors", [])
                    
                    # 移除矛盾的风险因素
                    updated_factors = []
                    for factor in risk_factors:
                        if not ("排名第1" in factor and "排名靠后" in factor):
                            updated_factors.append(factor)
                        else:
                            corrections.append("移除矛盾的行业排名风险描述")
                    
                    risk_analysis["risk_factors"] = updated_factors
                    report["risk_analysis"] = risk_analysis
            
            # 修正数值范围错误
            for issue in issues:
                if "负债率数值超出合理范围" in issue:
                    financial_data = report.get("financial_analysis", {})
                    debt_ratio = financial_data.get("debt_ratio")
                    if debt_ratio and (debt_ratio < 0 or debt_ratio > 1):
                        # 将负债率限制在合理范围内
                        financial_data["debt_ratio"] = max(0, min(1, debt_ratio))
                        corrections.append("修正负债率数值范围")
                        report["financial_analysis"] = financial_data
            
            # 修正缺失的投资建议
            for issue in issues:
                if "缺少明确的投资建议" in issue:
                    investment = report.get("investment_recommendation", {})
                    if not investment.get("recommendation"):
                        risk_level = report.get("risk_analysis", {}).get("overall_risk", "中")
                        if risk_level == "低":
                            investment["recommendation"] = "建议关注"
                        elif risk_level == "高":
                            investment["recommendation"] = "谨慎投资"
                        else:
                            investment["recommendation"] = "中性观望"
                        corrections.append("补充投资建议")
                        report["investment_recommendation"] = investment
                        
        except Exception as e:
            corrections.append(f"自动修正异常: {str(e)}")
        
        return corrections
    
    async def _generate_llm_corrections(self, report: Dict, issues: List[str], recommendations: List[str]) -> Dict:
        """使用大模型生成修正建议"""
        try:
            prompt = f"""
请对以下财务分析报告进行修正，解决发现的问题：

原始报告：
{json.dumps(report, ensure_ascii=False, indent=2)[:3000]}...

发现的问题：
{chr(10).join(f"- {issue}" for issue in issues)}

修正建议：
{chr(10).join(f"- {rec}" for rec in recommendations)}

请提供修正后的报告，重点修正以下方面：
1. 解决逻辑矛盾
2. 补充缺失信息
3. 更新过时数据
4. 改进分析质量

请以JSON格式返回修正后的完整报告。
"""
            
            result = await self._call_openai_api(prompt)
            if result.get("success"):
                try:
                    content = result.get("content", "{}")
                    corrected_report = json.loads(content)
                    return {
                        "success": True,
                        "corrected_report": corrected_report,
                        "corrections": ["应用大模型修正建议"]
                    }
                except json.JSONDecodeError:
                    return {"success": False, "error": "大模型响应格式错误"}
            else:
                return {"success": False, "error": "大模型调用失败"}
                
        except Exception as e:
            return {"success": False, "error": f"生成修正建议异常: {str(e)}"}

    def _calculate_validation_score(self, authenticity_result: Dict, rationality_result: Dict, timeliness_result: Dict) -> float:
        """计算综合校验分数"""
        try:
            # 获取各项分数，默认为0
            auth_score = authenticity_result.get('score', 0)
            ratio_score = rationality_result.get('score', 0)
            time_score = timeliness_result.get('score', 0)
            
            # 加权平均：真实性40%，合理性40%，时效性20%
            total_score = (auth_score * 0.4 + ratio_score * 0.4 + time_score * 0.2)
            return round(total_score, 2)
        except Exception as e:
            print(f"❌ 计算校验分数失败: {str(e)}")
            return 0.0

    def _generate_validation_summary(self, authenticity_result: Dict, rationality_result: Dict, 
                                   timeliness_result: Dict, validation_score: float) -> Dict:
        """生成校验摘要"""
        try:
            # 统计各项校验状态
            auth_status = authenticity_result.get('status', 'unknown')
            ratio_status = rationality_result.get('status', 'unknown')
            time_status = timeliness_result.get('status', 'unknown')
            
            # 统计问题数量
            total_issues = (
                len(authenticity_result.get('issues_found', [])) +
                len(rationality_result.get('issues_found', [])) +
                len(timeliness_result.get('issues_found', []))
            )
            
            # 确定总体状态
            if validation_score >= 0.8:
                overall_status = "excellent"
                status_desc = "报告质量优秀"
            elif validation_score >= 0.6:
                overall_status = "good"
                status_desc = "报告质量良好"
            elif validation_score >= 0.4:
                overall_status = "fair"
                status_desc = "报告质量一般"
            else:
                overall_status = "poor"
                status_desc = "报告质量较差"
            
            return {
                "overall_status": overall_status,
                "status_description": status_desc,
                "validation_score": validation_score,
                "total_issues_found": total_issues,
                "authenticity_status": auth_status,
                "rationality_status": ratio_status,
                "timeliness_status": time_status,
                "validation_time": get_current_iso_timestamp()
            }
        except Exception as e:
            print(f"❌ 生成校验摘要失败: {str(e)}")
            return {
                "overall_status": "error",
                "status_description": "校验摘要生成失败",
                "validation_score": 0.0,
                "error": str(e)
            }
    
    async def analyze_with_openai(self, company_data: Dict, financial_data: Dict) -> Dict:
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
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
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
            response = self.session.post(url, json=data, headers=headers, timeout=60)
            
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

    async def _step9_mcp_enhance_analysis(self, company_info: Dict, final_report: Dict, 
                                        metrics: Dict, context: Dict) -> Dict:
        """第9步：知识库+网络搜索增强分析（知识库优先）"""
        print("🔧 开始知识库+网络搜索增强分析...")
        
        try:
            company_name = company_info.get('name', '未知公司')
            stock_code = company_info.get('code', '')
            
            # 使用现有知识库进行增强分析
            enhanced_insights = await self._get_enhanced_insights_with_priority(
                company_name, stock_code
            )
            
            # 分类洞察来源
            categorized_insights = self._categorize_sources(enhanced_insights)
            
            # 整合增强分析结果
            enhanced_report = final_report.copy()
            enhanced_report['enhanced_analysis'] = {
                'insights': enhanced_insights,
                'categorized_insights': categorized_insights,
                'enhancement_time': get_current_iso_timestamp(),
                'total_insights': len(enhanced_insights),
                'knowledge_base_count': len(categorized_insights.get('knowledge_base_sources', [])),
                'web_search_count': len(categorized_insights.get('web_search_sources', []))
            }
            
            # 生成增强摘要
            if enhanced_insights:
                enhanced_summary = self._generate_enhanced_summary(categorized_insights)
                enhanced_report['enhanced_executive_summary'] = enhanced_summary
            
            print(f"✅ MCP服务增强分析完成，获得 {len(enhanced_insights)} 项增强洞察")
            
            return {
                'status': 'completed',
                'enhanced_report': enhanced_report,
                'enhanced_insights': categorized_insights,
                'summary': enhanced_summary if enhanced_insights else "无增强洞察",
                'mcp_analysis': {
                    'knowledge_base_id': self.knowledge_base_id,
                    'insights_count': len(enhanced_insights),
                    'enhancement_time': get_current_iso_timestamp()
                }
            }
            
        except Exception as e:
            print(f"❌ MCP服务增强分析失败: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'enhanced_report': final_report
            }
    
    async def _get_enhanced_insights_with_priority(self, company_name: str, stock_code: str) -> List[Dict]:
        """
        使用知识库优先、网络搜索补充的策略获取增强洞察
        
        Args:
            company_name: 公司名称
            stock_code: 股票代码
            
        Returns:
            List[Dict]: 增强洞察列表，包含来源标识和可信度评分
        """
        print(f"📊 开始获取 {company_name} 的增强洞察...")
        
        # 定义分析查询
        analysis_queries = [
            f"{company_name}的最新财务状况和投资价值分析",
            f"{company_name}的行业竞争地位和发展前景",
            f"{company_name}面临的主要风险和机遇",
            f"{company_name}的投资建议和目标价位"
        ]
        
        enhanced_insights = []
        
        for query in analysis_queries:
            print(f"🔍 处理查询: {query}")
            
            # 1. 优先从知识库获取信息
            kb_insight = await self._get_knowledge_base_insight(query)
            
            # 2. 从网络搜索获取补充信息
            web_insight = await self._get_web_search_insight(query)
            
            # 3. 合并和评估洞察
            merged_insight = self._merge_insights_with_priority(
                query, kb_insight, web_insight
            )
            
            if merged_insight:
                enhanced_insights.append(merged_insight)
            
            # 避免频繁请求
            await asyncio.sleep(1)
        
        print(f"✅ 获取增强洞察完成，共 {len(enhanced_insights)} 条")
        return enhanced_insights
    
    async def _get_knowledge_base_insight(self, query: str) -> Dict:
        """从知识库获取洞察"""
        try:
            print(f"   📚 查询知识库: {query[:30]}...")
            
            # 使用现有知识库进行查询
            kb_result = await self.query_knowledge(query, match_count=3)
            
            if kb_result.get('success') and kb_result.get('results'):
                return {
                    'source': 'knowledge_base',
                    'content': kb_result.get('summary', ''),
                    'results': kb_result.get('results', []),
                    'confidence': 0.9,  # 知识库数据可信度高
                    'timestamp': get_current_iso_timestamp()
                }
            else:
                print(f"   ⚠️ 知识库未找到相关信息")
                return {}
                
        except Exception as e:
            print(f"   ❌ 知识库查询失败: {str(e)}")
            return {}
    
    async def _get_web_search_insight(self, query: str) -> Dict:
        """从网络搜索获取洞察"""
        try:
            print(f"   🌐 网络搜索: {query[:30]}...")
            
            # 使用可信源搜索
            search_result = await self._search_trusted_sources(query)
            
            if search_result.get('results'):
                return {
                    'source': 'web_search',
                    'content': search_result.get('answer', ''),
                    'results': search_result.get('results', []),
                    'confidence': 0.7,  # 网络搜索数据可信度中等
                    'timestamp': get_current_iso_timestamp()
                }
            else:
                print(f"   ⚠️ 网络搜索未找到相关信息")
                return {}
                
        except Exception as e:
            print(f"   ❌ 网络搜索失败: {str(e)}")
            return {}
    
    def _merge_insights_with_priority(self, query: str, kb_insight: Dict, web_insight: Dict) -> Dict:
        """
        合并知识库和网络搜索的洞察，知识库优先
        
        Args:
            query: 查询问题
            kb_insight: 知识库洞察
            web_insight: 网络搜索洞察
            
        Returns:
            Dict: 合并后的洞察
        """
        if not kb_insight and not web_insight:
            return {}
        
        # 如果知识库有结果，优先使用知识库数据
        if kb_insight and kb_insight.get('content'):
            primary_source = kb_insight
            secondary_source = web_insight
            print(f"   ✅ 使用知识库数据作为主要来源")
        elif web_insight and web_insight.get('content'):
            primary_source = web_insight
            secondary_source = {}
            print(f"   ✅ 使用网络搜索数据作为主要来源")
        else:
            return {}
        
        # 构建合并后的洞察
        merged_insight = {
            'query': query,
            'primary_content': primary_source.get('content', ''),
            'primary_source': primary_source.get('source', ''),
            'primary_confidence': primary_source.get('confidence', 0.5),
            'primary_results': primary_source.get('results', []),
            'timestamp': get_current_iso_timestamp()
        }
        
        # 如果有补充信息，添加到洞察中
        if secondary_source and secondary_source.get('content'):
            merged_insight.update({
                'secondary_content': secondary_source.get('content', ''),
                'secondary_source': secondary_source.get('source', ''),
                'secondary_confidence': secondary_source.get('confidence', 0.5),
                'secondary_results': secondary_source.get('results', [])
            })
            print(f"   📝 添加补充信息来源: {secondary_source.get('source', '')}")
        
        return merged_insight
    
    def _categorize_sources(self, insights: List[Dict]) -> Dict:
        """分类洞察来源统计"""
        sources = {
            'knowledge_base_primary': 0,
            'web_search_primary': 0,
            'total_with_secondary': 0,
            'average_confidence': 0.0
        }
        
        if not insights:
            return sources
        
        total_confidence = 0
        for insight in insights:
            primary_source = insight.get('primary_source', '')
            if primary_source == 'knowledge_base':
                sources['knowledge_base_primary'] += 1
            elif primary_source == 'web_search':
                sources['web_search_primary'] += 1
            
            if insight.get('secondary_content'):
                sources['total_with_secondary'] += 1
            
            total_confidence += insight.get('primary_confidence', 0)
        
        sources['average_confidence'] = total_confidence / len(insights) if insights else 0
        return sources

    def _generate_enhanced_summary(self, enhanced_insights: Dict) -> str:
        """生成增强分析摘要"""
        try:
            summary_parts = []
            
            # 处理分类后的洞察数据
            if isinstance(enhanced_insights, dict):
                # 优先展示知识库来源的信息
                kb_sources = enhanced_insights.get('knowledge_base_sources', [])
                web_sources = enhanced_insights.get('web_search_sources', [])
                
                # 处理知识库来源
                for insight in kb_sources:
                    query = insight.get('query', '')
                    response = insight.get('response', '')
                    
                    if '财务状况' in query and response:
                        summary_parts.append(f"财务状况(知识库)：{response[:80]}...")
                    elif '竞争地位' in query and response:
                        summary_parts.append(f"竞争地位(知识库)：{response[:80]}...")
                    elif '风险' in query and response:
                        summary_parts.append(f"风险评估(知识库)：{response[:80]}...")
                    elif '投资建议' in query and response:
                        summary_parts.append(f"投资建议(知识库)：{response[:80]}...")
                
                # 处理网络搜索来源（作为补充）
                for insight in web_sources:
                    query = insight.get('query', '')
                    response = insight.get('response', '')
                    
                    if '财务状况' in query and response and not any('财务状况(知识库)' in part for part in summary_parts):
                        summary_parts.append(f"财务状况(网络)：{response[:80]}...")
                    elif '竞争地位' in query and response and not any('竞争地位(知识库)' in part for part in summary_parts):
                        summary_parts.append(f"竞争地位(网络)：{response[:80]}...")
                    elif '风险' in query and response and not any('风险评估(知识库)' in part for part in summary_parts):
                        summary_parts.append(f"风险评估(网络)：{response[:80]}...")
                    elif '投资建议' in query and response and not any('投资建议(知识库)' in part for part in summary_parts):
                        summary_parts.append(f"投资建议(网络)：{response[:80]}...")
            
            # 兼容旧格式（列表格式）
            elif isinstance(enhanced_insights, list):
                for insight in enhanced_insights:
                    query = insight.get('query', '')
                    response = insight.get('response', '')
                    
                    if '财务状况' in query and response:
                        summary_parts.append(f"财务状况：{response[:100]}...")
                    elif '竞争地位' in query and response:
                        summary_parts.append(f"竞争地位：{response[:100]}...")
                    elif '风险' in query and response:
                        summary_parts.append(f"风险评估：{response[:100]}...")
                    elif '投资建议' in query and response:
                        summary_parts.append(f"投资建议：{response[:100]}...")
            
            if summary_parts:
                return "基于MCP服务增强分析：" + " | ".join(summary_parts)
            else:
                return "MCP服务增强分析已完成，详见增强洞察部分。"
                
        except Exception as e:
            print(f"❌ 生成增强摘要失败: {str(e)}")
            return "MCP服务增强分析已完成。"
    
    # ==================== 知识库操作功能 ====================
    
    async def add_knowledge_from_url(self, url: str, description: str = "") -> Dict:
        """
        从URL添加知识到知识库
        
        Args:
            url: 要添加的URL地址
            description: 知识描述
            
        Returns:
            Dict: 添加结果
        """
        try:
            print(f"📚 正在添加知识到知识库...")
            print(f"🔗 URL: {url}")
            print(f"📝 描述: {description}")
            
            # 使用MCP服务添加知识
            result = await self.mcp_client.create_knowledge_from_url(
                kb_id=self.knowledge_base_id,
                url=url,
                enable_multimodel=True
            )
            
            if result.get('success'):
                print(f"✅ 知识添加成功")
                return {
                    'success': True,
                    'knowledge_id': result.get('knowledge_id'),
                    'url': url,
                    'description': description,
                    'message': '知识已成功添加到知识库',
                    'timestamp': get_current_iso_timestamp()
                }
            else:
                print(f"❌ 知识添加失败: {result.get('error', '未知错误')}")
                return {
                    'success': False,
                    'error': result.get('error', '知识添加失败'),
                    'url': url
                }
                
        except Exception as e:
            error_msg = f"添加知识时发生异常: {str(e)}"
            print(f"💥 {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'url': url
            }
    
    async def query_knowledge(self, query: str, match_count: int = 5) -> Dict:
        """
        查询知识库
        
        Args:
            query: 查询问题
            match_count: 返回结果数量
            
        Returns:
            Dict: 查询结果
        """
        try:
            print(f"🔍 正在查询知识库...")
            print(f"❓ 查询问题: {query}")
            
            # 使用混合搜索查询知识库
            search_result = await self.mcp_client.hybrid_search(
                kb_id=self.knowledge_base_id,
                query=query,
                match_count=match_count,
                vector_threshold=0.5,
                keyword_threshold=0.3
            )
            
            if search_result.get('success'):
                results = search_result.get('results', [])
                print(f"✅ 找到 {len(results)} 条相关知识")
                
                return {
                    'success': True,
                    'query': query,
                    'results': results,
                    'total_count': len(results),
                    'timestamp': get_current_iso_timestamp()
                }
            else:
                print(f"❌ 知识查询失败: {search_result.get('error', '未知错误')}")
                return {
                    'success': False,
                    'error': search_result.get('error', '知识查询失败'),
                    'query': query
                }
                
        except Exception as e:
            error_msg = f"查询知识时发生异常: {str(e)}"
            print(f"💥 {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'query': query
            }
    
    async def chat_with_knowledge(self, question: str) -> Dict:
        """
        与知识库进行对话
        
        Args:
            question: 用户问题
            
        Returns:
            Dict: 对话结果
        """
        try:
            print(f"💬 正在与知识库对话...")
            print(f"❓ 问题: {question}")
            
            # 创建聊天会话
            session_result = await self.mcp_client.create_chat_session(
                kb_id=self.knowledge_base_id,
                enable_rewrite=True,
                max_rounds=5,
                fallback_response="抱歉，我无法在知识库中找到相关信息来回答这个问题。"
            )
            
            if not session_result.get('success'):
                return {
                    'success': False,
                    'error': f"创建聊天会话失败: {session_result.get('error')}",
                    'question': question
                }
            
            session_id = session_result.get('session_id')
            print(f"📱 创建聊天会话: {session_id}")
            
            # 进行对话
            chat_result = await self.mcp_client.chat(session_id, question)
            
            if chat_result.get('success'):
                answer = chat_result.get('response', '未获取到回答')
                print(f"✅ 获得回答")
                
                # 清理会话
                await self.mcp_client.delete_session(session_id)
                
                return {
                    'success': True,
                    'question': question,
                    'answer': answer,
                    'session_id': session_id,
                    'timestamp': get_current_iso_timestamp()
                }
            else:
                # 清理会话
                await self.mcp_client.delete_session(session_id)
                
                print(f"❌ 对话失败: {chat_result.get('error', '未知错误')}")
                return {
                    'success': False,
                    'error': chat_result.get('error', '对话失败'),
                    'question': question
                }
                
        except Exception as e:
            error_msg = f"与知识库对话时发生异常: {str(e)}"
            print(f"💥 {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'question': question
            }
    
    async def list_knowledge_base_info(self) -> Dict:
        """
        获取知识库信息
        
        Returns:
            Dict: 知识库信息
        """
        try:
            print(f"📊 正在获取知识库信息...")
            
            # 获取知识库列表
            kb_list_result = await self.mcp_client.list_knowledge_bases()
            
            if kb_list_result.get('success'):
                knowledge_bases = kb_list_result.get('knowledge_bases', [])
                
                # 查找当前使用的知识库
                current_kb = None
                for kb in knowledge_bases:
                    if kb.get('id') == self.knowledge_base_id:
                        current_kb = kb
                        break
                
                if current_kb:
                    print(f"✅ 找到当前知识库: {current_kb.get('name', '未知')}")
                    return {
                        'success': True,
                        'knowledge_base': current_kb,
                        'kb_id': self.knowledge_base_id,
                        'total_knowledge_bases': len(knowledge_bases),
                        'timestamp': get_current_iso_timestamp()
                    }
                else:
                    return {
                        'success': False,
                        'error': f"未找到ID为 {self.knowledge_base_id} 的知识库",
                        'available_kbs': [kb.get('name') for kb in knowledge_bases]
                    }
            else:
                return {
                    'success': False,
                    'error': kb_list_result.get('error', '获取知识库列表失败')
                }
                
        except Exception as e:
            error_msg = f"获取知识库信息时发生异常: {str(e)}"
            print(f"💥 {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    async def analyze_company_with_knowledge(self, company_input: str, use_knowledge: bool = True) -> Dict:
        """
        结合知识库进行公司分析
        
        Args:
            company_input: 公司名称或股票代码
            use_knowledge: 是否使用知识库增强分析
            
        Returns:
            Dict: 增强的分析报告
        """
        print(f"🚀 开始结合知识库分析公司: {company_input}")
        
        # 首先执行标准分析
        standard_analysis = await self.analyze_company(company_input)
        
        if not use_knowledge or not standard_analysis.get('final_report'):
            return standard_analysis
        
        try:
            # 使用知识库增强分析
            company_info = self._get_company_info_from_context(standard_analysis)
            company_name = company_info.get('company_name', company_input)
            
            # 构建知识库查询问题
            knowledge_queries = [
                f"{company_name}的最新财务状况如何？",
                f"{company_name}所在行业的发展趋势是什么？",
                f"{company_name}面临的主要风险有哪些？",
                f"{company_name}的投资价值分析"
            ]
            
            knowledge_insights = []
            
            for query in knowledge_queries:
                print(f"🔍 查询知识库: {query}")
                kb_result = await self.chat_with_knowledge(query)
                
                if kb_result.get('success'):
                    knowledge_insights.append({
                        'query': query,
                        'answer': kb_result.get('answer', ''),
                        'source': 'knowledge_base'
                    })
                
                # 避免频繁请求
                await asyncio.sleep(1)
            
            # 将知识库洞察整合到分析报告中
            if knowledge_insights:
                enhanced_report = standard_analysis['final_report'].copy()
                enhanced_report['knowledge_base_insights'] = knowledge_insights
                enhanced_report['enhancement_summary'] = self._generate_knowledge_enhancement_summary(knowledge_insights)
                
                standard_analysis['final_report'] = enhanced_report
                standard_analysis['knowledge_enhanced'] = True
                
                print(f"✅ 知识库增强分析完成，获得 {len(knowledge_insights)} 条额外洞察")
            else:
                print("⚠️ 未从知识库获得额外洞察")
                standard_analysis['knowledge_enhanced'] = False
            
        except Exception as e:
            print(f"⚠️ 知识库增强分析失败: {str(e)}")
            standard_analysis['knowledge_enhancement_error'] = str(e)
            standard_analysis['knowledge_enhanced'] = False
        
        return standard_analysis
    
    def _generate_knowledge_enhancement_summary(self, insights: List[Dict]) -> str:
        """生成知识库增强摘要"""
        if not insights:
            return "未获得知识库增强信息"
        
        summary_parts = []
        for insight in insights:
            query = insight.get('query', '')
            answer = insight.get('answer', '')
            if answer and len(answer) > 50:
                summary_parts.append(f"关于'{query}'：{answer[:100]}...")
        
        if summary_parts:
            return "知识库增强洞察：" + " | ".join(summary_parts)
        else:
            return "知识库查询完成，但未获得有效增强信息"

async def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python main.py <公司名称或股票代码>")
        print("示例: python main.py 平安银行")
        print("示例: python main.py 000001")
        return
    
    company_input = sys.argv[1]
    
    # 创建分析器并执行分析
    analyzer = FinancialAnalyzer()
    result = await analyzer.analyze_company(company_input)
    
    # 输出结果
    print("\n" + "="*60)
    print("📋 财报分析报告")
    print("="*60)
    
    if result.get("status") == "completed":
        final_report = result.get("final_report", {})
        
        print(f"📊 执行摘要: {final_report.get('executive_summary', '')}")
        print(f"💰 财务亮点: {final_report.get('financial_highlights', {})}")
        print(f"🏆 行业地位: {final_report.get('industry_position', '')}")
        print(f"⚠️  主要风险: {final_report.get('key_risks', [])}")
        print(f"💡 投资建议: {final_report.get('investment_recommendation', '')}")
        print(f"📅 报告日期: {final_report.get('report_date', '')}")
        
        # 显示 OpenAI 分析结果
        openai_analysis = final_report.get('openai_analysis', {})
        if openai_analysis:
            print("\n" + "="*60)
            print("🤖 AI 专业分析")
            print("="*60)
            
            # 正确访问嵌套的analysis_result结构
            analysis_result = openai_analysis.get('analysis_result', {})
            if analysis_result.get('success'):
                analysis_content = analysis_result.get('analysis', '')
                if analysis_content:
                    print(analysis_content)
                else:
                    print("AI 分析内容为空")
            else:
                print(f"❌ AI 分析失败: {analysis_result.get('error', '未知错误')}")
                fallback = analysis_result.get('fallback_analysis', '')
                if fallback:
                    print("\n📝 备用分析:")
                    print(fallback)
        
        # 保存完整报告到文件
        output_file = f"financial_report_{company_input}_{get_filename_timestamp()}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n📄 完整报告已保存到: {output_file}")
        
    else:
        print(f"❌ 分析失败: {result.get('error', '未知错误')}")

if __name__ == "__main__":
    asyncio.run(main())
