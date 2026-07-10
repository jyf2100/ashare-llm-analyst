#!/usr/bin/env python3
"""
AI分析器模块
负责AI模型调用和智能分析
"""
import json
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI

from config_manager import config
from logger_manager import get_logger
from exception_handler import api_retry, safe_data_processing, APIError
from cache_manager import cache_manager

logger = get_logger('ai_analyzer')

class AIAnalyzer:
    """AI分析器"""
    
    def __init__(self) -> None:
        openai_config: Dict[str, Any] = config.get_openai_config()
        self.client: OpenAI = OpenAI(
            api_key=openai_config['api_key'],
            base_url=openai_config['base_url']
        )
        self.model: str = openai_config['model']
        self.timeout: int = openai_config['timeout']
        self.max_retries: int = openai_config['max_retries']
    
    @api_retry(max_retries=3, delay=2.0)
    @safe_data_processing(default_value="")
    def generate_analysis(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """生成AI分析"""
        cache_key = f"ai_analysis_{hash(prompt)}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 构建完整的提示词
            full_prompt = self._build_prompt(prompt, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的财务分析师，具有丰富的A股市场分析经验。"},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                timeout=self.timeout
            )
            
            result = response.choices[0].message.content.strip()
            
            cache_manager.set(cache_key, result)
            logger.info(f"AI分析生成成功，长度: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"AI分析生成失败: {e}")
            raise APIError(f"AI分析失败: {e}", "openai")
    
    def _build_prompt(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """构建完整的提示词"""
        if not context:
            return prompt
        
        context_str = ""
        if context:
            context_str = "\n\n参考信息：\n"
            for key, value in context.items():
                if isinstance(value, (dict, list)):
                    context_str += f"{key}: {json.dumps(value, ensure_ascii=False, indent=2)}\n"
                else:
                    context_str += f"{key}: {value}\n"
        
        return f"{prompt}{context_str}"
    
    def analyze_financial_data(self, company_name: str, financial_data: Dict[str, Any]) -> str:
        """分析财务数据"""
        prompt = f"""
请对{company_name}的财务数据进行专业分析，包括：

1. 盈利能力分析
2. 偿债能力分析  
3. 营运能力分析
4. 成长能力分析
5. 财务风险评估
6. 投资建议

请提供详细的分析结论和具体的数据支撑。
"""
        
        context = {
            "公司名称": company_name,
            "财务数据": financial_data
        }
        
        return self.generate_analysis(prompt, context)
    
    def analyze_market_trend(self, company_name: str, market_data: Dict[str, Any], 
                           industry_data: Dict[str, Any] = None) -> str:
        """分析市场趋势"""
        prompt = f"""
请分析{company_name}的市场表现和趋势，包括：

1. 股价走势分析
2. 市场表现评估
3. 行业对比分析
4. 技术指标分析
5. 市场情绪分析
6. 未来趋势预测

请结合具体数据给出专业判断。
"""
        
        context = {
            "公司名称": company_name,
            "市场数据": market_data
        }
        
        if industry_data:
            context["行业数据"] = industry_data
        
        return self.generate_analysis(prompt, context)
    
    def analyze_company_fundamentals(self, company_name: str, basic_info: Dict[str, Any],
                                   business_data: Dict[str, Any] = None) -> str:
        """分析公司基本面"""
        prompt = f"""
请对{company_name}进行基本面分析，包括：

1. 公司概况和主营业务
2. 竞争优势分析
3. 商业模式评估
4. 管理层评价
5. 发展战略分析
6. 行业地位评估

请提供客观、专业的分析意见。
"""
        
        context = {
            "公司名称": company_name,
            "基本信息": basic_info
        }
        
        if business_data:
            context["业务数据"] = business_data
        
        return self.generate_analysis(prompt, context)
    
    def generate_investment_advice(self, company_name: str, analysis_data: Dict[str, Any]) -> str:
        """生成投资建议"""
        prompt = f"""
基于对{company_name}的综合分析，请提供投资建议：

1. 投资评级（买入/持有/卖出）
2. 目标价位预测
3. 投资亮点
4. 主要风险
5. 适合的投资者类型
6. 投资时间建议

请给出明确的投资建议和风险提示。
"""
        
        context = {
            "公司名称": company_name,
            "分析数据": analysis_data
        }
        
        return self.generate_analysis(prompt, context)
    
    def summarize_research_report(self, company_name: str, all_analysis: Dict[str, Any]) -> str:
        """生成研究报告摘要"""
        prompt = f"""
请为{company_name}生成一份专业的投资研究报告摘要，包括：

1. 执行摘要
2. 核心观点
3. 关键数据
4. 主要结论
5. 投资建议
6. 风险提示

报告应该简洁明了，突出重点，适合投资者快速了解。
"""
        
        context = {
            "公司名称": company_name,
            "完整分析": all_analysis
        }
        
        return self.generate_analysis(prompt, context)
    
    def validate_analysis_quality(self, analysis_text: str) -> Dict[str, Any]:
        """验证分析质量"""
        prompt = f"""
请评估以下财务分析报告的质量，从以下维度打分（1-10分）：

1. 数据准确性
2. 分析深度
3. 逻辑性
4. 专业性
5. 可读性
6. 实用性

同时指出报告的优点和需要改进的地方。

分析报告：
{analysis_text}
"""
        
        try:
            result = self.generate_analysis(prompt)
            
            # 尝试解析评分结果
            quality_score = {
                "overall_score": 0,
                "detailed_scores": {},
                "strengths": [],
                "improvements": [],
                "validation_result": result
            }
            
            return quality_score
            
        except Exception as e:
            logger.error(f"分析质量验证失败: {e}")
            return {
                "overall_score": 0,
                "error": str(e)
            }
    
    def enhance_analysis_with_context(self, base_analysis: str, additional_context: Dict[str, Any]) -> str:
        """使用额外上下文增强分析"""
        prompt = f"""
请基于以下基础分析和额外信息，生成更加全面和深入的分析报告：

基础分析：
{base_analysis}

请结合额外信息，补充和完善分析内容，特别关注：
1. 数据的时效性和准确性
2. 市场环境的变化
3. 行业发展趋势
4. 公司最新动态
5. 风险因素的变化

生成更加准确和有价值的分析结论。
"""
        
        context = {"额外信息": additional_context}
        
        return self.generate_analysis(prompt, context)
    
    def generate_risk_assessment(self, company_name: str, risk_factors: List[str]) -> str:
        """生成风险评估"""
        prompt = f"""
请对{company_name}进行全面的风险评估，重点分析以下风险因素：

{chr(10).join([f"{i+1}. {factor}" for i, factor in enumerate(risk_factors)])}

风险评估应包括：
1. 各项风险的严重程度评级
2. 风险发生的可能性
3. 对公司的潜在影响
4. 风险缓解措施建议
5. 投资者应对策略

请提供专业的风险分析和建议。
"""
        
        context = {
            "公司名称": company_name,
            "风险因素": risk_factors
        }
        
        return self.generate_analysis(prompt, context)
    
    def batch_analyze(self, analysis_tasks: List[Dict[str, Any]]) -> List[str]:
        """批量分析"""
        results = []
        
        for i, task in enumerate(analysis_tasks):
            try:
                logger.info(f"执行批量分析任务 {i+1}/{len(analysis_tasks)}")
                
                result = self.generate_analysis(
                    task.get('prompt', ''),
                    task.get('context', {})
                )
                results.append(result)
                
                # 添加延迟避免API限制
                if i < len(analysis_tasks) - 1:
                    time.sleep(config.get('analysis.step_delay', 0.5))
                    
            except Exception as e:
                logger.error(f"批量分析任务 {i+1} 失败: {e}")
                results.append(f"分析失败: {e}")
        
        logger.info(f"批量分析完成，共处理 {len(results)} 个任务")
        return results