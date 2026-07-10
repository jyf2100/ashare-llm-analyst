#!/usr/bin/env python3
"""
重构后的财务分析器主类
整合所有功能模块，提供统一的分析接口
"""
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# 导入自定义模块
from config_manager import config
from logger_manager import get_logger, main_logger
from exception_handler import safe_call, log_exceptions, APIError, DataError
from cache_manager import cache_manager
from data_fetcher import DataFetcher
from search_engine import SearchEngine
from ai_analyzer import AIAnalyzer

logger = get_logger('financial_analyzer_v2')

class FinancialAnalyzerV2:
    """重构后的财务分析器"""
    
    def __init__(self) -> None:
        """初始化分析器"""
        logger.info("初始化财务分析器V2")
        
        # 初始化各个组件
        self.data_fetcher: DataFetcher = DataFetcher()
        self.search_engine: SearchEngine = SearchEngine()
        self.ai_analyzer: AIAnalyzer = AIAnalyzer()
        
        # 配置信息
        self.analysis_config: Dict[str, Any] = config.get_analysis_config()
        self.kb_config: Dict[str, Any] = config.get_knowledge_base_config()
        
        # 分析状态
        self.current_analysis: Dict[str, Any] = {}
        self.analysis_steps: List[str] = []
        
        logger.info("财务分析器V2初始化完成")
    
    @log_exceptions
    def analyze_company(self, company_name: str, stock_code: str = None, 
                       enable_enhancement: bool = True) -> Dict[str, Any]:
        """分析公司的主要入口方法"""
        logger.info(f"开始分析公司: {company_name} ({stock_code})")
        
        # 初始化分析结果
        analysis_result = {
            "company_name": company_name,
            "stock_code": stock_code,
            "analysis_time": datetime.now().isoformat(),
            "steps_completed": [],
            "basic_analysis": {},
            "enhanced_analysis": {},
            "final_report": "",
            "success": False,
            "errors": []
        }
        
        try:
            # 步骤1: 获取基本信息
            success, basic_info = self._step1_get_basic_info(company_name, stock_code)
            if success:
                analysis_result["basic_analysis"]["basic_info"] = basic_info
                analysis_result["steps_completed"].append("basic_info")
                if not stock_code and "code" in basic_info:
                    stock_code = basic_info["code"]
                    analysis_result["stock_code"] = stock_code
            
            # 步骤2: 获取财务数据
            success, financial_data = self._step2_get_financial_data(stock_code)
            if success:
                analysis_result["basic_analysis"]["financial_data"] = financial_data
                analysis_result["steps_completed"].append("financial_data")
            
            # 步骤3: 获取市场数据
            success, market_data = self._step3_get_market_data(stock_code)
            if success:
                analysis_result["basic_analysis"]["market_data"] = market_data
                analysis_result["steps_completed"].append("market_data")
            
            # 步骤4: 基础AI分析
            success, ai_analysis = self._step4_basic_ai_analysis(company_name, analysis_result["basic_analysis"])
            if success:
                analysis_result["basic_analysis"]["ai_analysis"] = ai_analysis
                analysis_result["steps_completed"].append("ai_analysis")
            
            # 步骤5: 增强分析（如果启用）
            if enable_enhancement and self.analysis_config.get('mcp_enhancement_enabled', True):
                success, enhanced_data = self._step5_enhanced_analysis(company_name, analysis_result["basic_analysis"])
                if success:
                    analysis_result["enhanced_analysis"] = enhanced_data
                    analysis_result["steps_completed"].append("enhanced_analysis")
            
            # 步骤6: 生成最终报告
            success, final_report = self._step6_generate_final_report(company_name, analysis_result)
            if success:
                analysis_result["final_report"] = final_report
                analysis_result["steps_completed"].append("final_report")
                analysis_result["success"] = True
            
            logger.info(f"公司分析完成: {company_name}, 成功步骤: {len(analysis_result['steps_completed'])}")
            return analysis_result
            
        except Exception as e:
            error_msg = f"分析过程发生异常: {e}"
            logger.error(error_msg)
            analysis_result["errors"].append(error_msg)
            return analysis_result
    
    def _step1_get_basic_info(self, company_name: str, stock_code: str = None) -> Tuple[bool, Dict[str, Any]]:
        """步骤1: 获取公司基本信息"""
        logger.info("执行步骤1: 获取公司基本信息")
        
        try:
            if stock_code:
                # 直接使用股票代码获取信息
                basic_info = self.data_fetcher.get_company_basic_info(stock_code)
                if basic_info:
                    basic_info["company_name"] = company_name
                    return True, basic_info
            
            # 通过搜索获取股票代码
            search_result = self.search_engine.web_search(f"{company_name} 股票代码 A股")
            
            # 从搜索结果中提取股票代码（简化实现）
            extracted_code = self._extract_stock_code_from_search(search_result, company_name)
            
            if extracted_code:
                basic_info = self.data_fetcher.get_company_basic_info(extracted_code)
                basic_info["company_name"] = company_name
                basic_info["code"] = extracted_code
                return True, basic_info
            
            # 如果无法获取详细信息，返回基本信息
            return True, {
                "company_name": company_name,
                "code": stock_code or "未知",
                "status": "基本信息获取受限"
            }
            
        except Exception as e:
            logger.error(f"获取基本信息失败: {e}")
            return False, {"error": str(e)}
    
    def _step2_get_financial_data(self, stock_code: str) -> Tuple[bool, Dict[str, Any]]:
        """步骤2: 获取财务数据"""
        logger.info("执行步骤2: 获取财务数据")
        
        if not stock_code or stock_code == "未知":
            return False, {"error": "股票代码无效"}
        
        try:
            financial_data = {}
            current_year = datetime.now().year
            
            # 获取最近几个季度的财务数据
            for year in [current_year, current_year - 1]:
                for quarter in [4, 3, 2, 1]:
                    if year == current_year and quarter > (datetime.now().month - 1) // 3 + 1:
                        continue
                    
                    try:
                        # 获取各类财务数据
                        profit_data = self.data_fetcher.get_financial_data(stock_code, year, quarter, "profit")
                        if not profit_data.empty:
                            financial_data[f"{year}Q{quarter}_profit"] = profit_data.to_dict('records')[0]
                        
                        balance_data = self.data_fetcher.get_financial_data(stock_code, year, quarter, "balance")
                        if not balance_data.empty:
                            financial_data[f"{year}Q{quarter}_balance"] = balance_data.to_dict('records')[0]
                        
                        # 只获取最近两个季度的详细数据
                        if len(financial_data) >= 4:
                            break
                            
                    except Exception as e:
                        logger.warning(f"获取{year}Q{quarter}财务数据失败: {e}")
                        continue
                
                if len(financial_data) >= 4:
                    break
            
            if financial_data:
                return True, financial_data
            else:
                return False, {"error": "未获取到财务数据"}
                
        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")
            return False, {"error": str(e)}
    
    def _step3_get_market_data(self, stock_code: str) -> Tuple[bool, Dict[str, Any]]:
        """步骤3: 获取市场数据"""
        logger.info("执行步骤3: 获取市场数据")
        
        if not stock_code or stock_code == "未知":
            return False, {"error": "股票代码无效"}
        
        try:
            market_data = {}
            
            # 获取K线数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            k_data = self.data_fetcher.get_stock_k_data(stock_code, start_date, end_date)
            if not k_data.empty:
                market_data["k_data_summary"] = {
                    "total_records": len(k_data),
                    "latest_price": float(k_data.iloc[-1]["close"]) if "close" in k_data.columns else 0,
                    "price_change": float(k_data.iloc[-1]["pctChg"]) if "pctChg" in k_data.columns else 0,
                    "volume": float(k_data.iloc[-1]["volume"]) if "volume" in k_data.columns else 0
                }
            
            # 获取实时数据
            try:
                real_time_data = self.data_fetcher.get_real_time_data(stock_code)
                market_data["real_time"] = real_time_data
            except Exception as e:
                logger.warning(f"获取实时数据失败: {e}")
            
            # 获取市场整体数据
            try:
                market_overview = self.data_fetcher.get_market_data()
                market_data["market_overview"] = market_overview
            except Exception as e:
                logger.warning(f"获取市场概况失败: {e}")
            
            return True, market_data
            
        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return False, {"error": str(e)}
    
    def _step4_basic_ai_analysis(self, company_name: str, basic_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """步骤4: 基础AI分析"""
        logger.info("执行步骤4: 基础AI分析")
        
        try:
            ai_analysis = {}
            
            # 财务分析
            if "financial_data" in basic_data:
                financial_analysis = self.ai_analyzer.analyze_financial_data(
                    company_name, basic_data["financial_data"]
                )
                ai_analysis["financial_analysis"] = financial_analysis
            
            # 市场分析
            if "market_data" in basic_data:
                market_analysis = self.ai_analyzer.analyze_market_trend(
                    company_name, basic_data["market_data"]
                )
                ai_analysis["market_analysis"] = market_analysis
            
            # 基本面分析
            if "basic_info" in basic_data:
                fundamental_analysis = self.ai_analyzer.analyze_company_fundamentals(
                    company_name, basic_data["basic_info"]
                )
                ai_analysis["fundamental_analysis"] = fundamental_analysis
            
            # 投资建议
            investment_advice = self.ai_analyzer.generate_investment_advice(
                company_name, basic_data
            )
            ai_analysis["investment_advice"] = investment_advice
            
            return True, ai_analysis
            
        except Exception as e:
            logger.error(f"基础AI分析失败: {e}")
            return False, {"error": str(e)}
    
    def _step5_enhanced_analysis(self, company_name: str, basic_analysis: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """步骤5: 增强分析"""
        logger.info("执行步骤5: 增强分析")
        
        try:
            enhanced_data = {}
            
            # 搜索最新新闻和公告
            news_result = self.search_engine.search_company_news(company_name)
            enhanced_data["latest_news"] = news_result
            
            # 行业分析
            industry_result = self.search_engine.search_industry_analysis(f"{company_name}所属行业")
            enhanced_data["industry_analysis"] = industry_result
            
            # 知识库查询
            try:
                kb_result = self.search_engine.knowledge_base_search(f"{company_name} 财务分析")
                enhanced_data["knowledge_base"] = kb_result
            except Exception as e:
                logger.warning(f"知识库查询失败: {e}")
                enhanced_data["knowledge_base"] = {"error": str(e)}
            
            # 使用额外信息增强基础分析
            if enhanced_data:
                enhanced_ai_analysis = self.ai_analyzer.enhance_analysis_with_context(
                    basic_analysis.get("ai_analysis", {}).get("investment_advice", ""),
                    enhanced_data
                )
                enhanced_data["enhanced_ai_analysis"] = enhanced_ai_analysis
            
            return True, enhanced_data
            
        except Exception as e:
            logger.error(f"增强分析失败: {e}")
            return False, {"error": str(e)}
    
    def _step6_generate_final_report(self, company_name: str, analysis_result: Dict[str, Any]) -> Tuple[bool, str]:
        """步骤6: 生成最终报告"""
        logger.info("执行步骤6: 生成最终报告")
        
        try:
            # 整合所有分析数据
            all_analysis = {
                "basic_analysis": analysis_result.get("basic_analysis", {}),
                "enhanced_analysis": analysis_result.get("enhanced_analysis", {}),
                "steps_completed": analysis_result.get("steps_completed", [])
            }
            
            # 生成综合报告
            final_report = self.ai_analyzer.summarize_research_report(company_name, all_analysis)
            
            return True, final_report
            
        except Exception as e:
            logger.error(f"生成最终报告失败: {e}")
            return False, f"报告生成失败: {e}"
    
    def _extract_stock_code_from_search(self, search_result: Dict[str, Any], company_name: str) -> Optional[str]:
        """从搜索结果中提取股票代码"""
        try:
            # 导入股票代码格式化函数
            from data_fetcher import format_stock_code
            
            # 简化的股票代码提取逻辑
            for result in search_result.get("results", []):
                content = result.get("content", "").lower()
                if company_name.lower() in content:
                    # 查找股票代码模式 (简化实现)
                    import re
                    patterns = [
                        r'(sh\.\d{6})',  # sh.开头
                        r'(sz\.\d{6})',  # sz.开头
                        r'(\d{6})',      # 6位数字
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            # 使用格式化函数确保返回正确格式
                            return format_stock_code(matches[0])
            
            return None
            
        except Exception as e:
            logger.error(f"提取股票代码失败: {e}")
            return None
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """获取分析状态"""
        return {
            "current_analysis": self.current_analysis,
            "steps_completed": self.analysis_steps,
            "cache_stats": cache_manager.get_stats(),
            "components_status": {
                "data_fetcher": "active",
                "search_engine": "active", 
                "ai_analyzer": "active"
            }
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        cache_manager.clear()
        logger.info("缓存已清空")
    
    def close(self) -> None:
        """关闭分析器"""
        logger.info("关闭财务分析器V2")
        
        try:
            self.data_fetcher.close()
            self.search_engine.close()
            logger.info("所有组件已关闭")
        except Exception as e:
            logger.error(f"关闭组件时发生错误: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

# 便捷函数
def analyze_company_simple(company_name: str, stock_code: str = None) -> Dict[str, Any]:
    """简化的公司分析函数"""
    with FinancialAnalyzerV2() as analyzer:
        return analyzer.analyze_company(company_name, stock_code)

if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="财务分析器V2")
    parser.add_argument("company_name", help="公司名称")
    parser.add_argument("--stock_code", help="股票代码")
    parser.add_argument("--no_enhancement", action="store_true", help="禁用增强分析")
    
    args = parser.parse_args()
    
    # 执行分析
    result = analyze_company_simple(
        args.company_name, 
        args.stock_code,
    )
    
    # 输出结果
    print(json.dumps(result, ensure_ascii=False, indent=2))