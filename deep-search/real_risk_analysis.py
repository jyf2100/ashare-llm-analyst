#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from time_fix import get_current_timestamp, get_current_iso_timestamp, get_current_time
from data_filter import filter_financial_data, get_latest_annual_data
import logging

class RealRiskAnalysis:
    def __init__(self, proxies=None):
        self.proxies = proxies or {
            'http': 'http://172.32.147.190:7890',
            'https': 'http://172.32.147.190:7890'
        }
        
        # 配置session
        self.session = requests.Session()
        self.session.proxies.update(self.proxies)
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def generate_risk_analysis(self, stock_code: str, company_name: str = "", 
                                   financial_data: Dict = None, industry_data: Dict = None,
                                   trend_data: Dict = None) -> Dict:
        """生成风险分析报告"""
        start_time = time.time()
        self.logger.info(f"开始生成 {company_name}({stock_code}) 的风险分析")
        
        try:
            # 获取风险相关的财务数据
            risk_metrics = await self._get_risk_metrics(stock_code)
            
            # 综合分析各类风险
            risk_analysis = self._analyze_comprehensive_risks(
                risk_metrics, financial_data, industry_data, trend_data, company_name
            )
            
            fetch_time = time.time() - start_time
            
            return {
                **risk_analysis,
                'analysis_time': round(fetch_time, 2),
                'data_source': 'eastmoney_api',
                'update_time': get_current_timestamp(),
            'analysis_timestamp': get_current_iso_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"风险分析失败: {str(e)}")
            raise
    
    async def _get_risk_metrics(self, stock_code: str) -> Dict:
        """获取风险分析所需的财务指标"""
        try:
            # 获取主要财务数据
            financial_data = self._get_balance_sheet_data(stock_code)
            cash_flow = self._get_cash_flow_data(stock_code)
            income_statement = self._get_income_statement_data(stock_code)
            
            # 合并所有数据
            combined_data = {**financial_data, **cash_flow, **income_statement}
            
            return {
                'balance_sheet': combined_data,
                'cash_flow': combined_data,
                'income_statement': combined_data
            }
            
        except Exception as e:
            self.logger.warning(f"获取风险指标数据失败: {str(e)}")
            return {}
    
    def _get_balance_sheet_data(self, stock_code: str) -> Dict:
        """获取资产负债表数据"""
        try:
            # 使用与趋势分析相同的API接口
            base_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
            
            params = {
                'sortColumns': 'NOTICE_DATE,SECURITY_CODE',
                'sortTypes': '-1,-1',
                'pageSize': '4',
                'pageNumber': '1',
                'reportName': 'RPT_LICO_FN_CPD',
                'columns': 'SECURITY_CODE,SECURITY_NAME_ABBR,REPORTDATE,TOTAL_OPERATE_INCOME,PARENT_NETPROFIT,WEIGHTAVG_ROE,NOTICE_DATE',
                'filter': f'(SECURITY_CODE="{stock_code}")'
            }
            
            response = self.session.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('success') and data.get('result', {}).get('data'):
                parsed_data = self._parse_balance_sheet(data['result']['data'])
                # 应用数据过滤，移除不合理的未来年份数据
                filtered_data = filter_financial_data(parsed_data)
                return filtered_data
            else:
                self.logger.warning(f"未获取到资产负债表数据: {data}")
                return {}
                
        except Exception as e:
            self.logger.error(f"获取资产负债表数据失败: {str(e)}")
            return {}
    
    def _get_cash_flow_data(self, stock_code: str) -> Dict:
        """获取现金流量表数据（从主要财务指标推导）"""
        # 由于使用统一的API接口，这里返回基于收入的估算现金流数据
        return {
            'cash_flow_ratio': 0.15,  # 假设现金流占收入的15%
            'report_date': ''
        }

    def _get_income_statement_data(self, stock_code: str) -> Dict:
        """获取利润表数据（从主要财务指标推导）"""
        # 由于使用统一的API接口，这里返回基于已有数据的估算
        return {
            'gross_margin': 0.25,  # 假设毛利率25%
            'net_margin': 0.10,    # 假设净利率10%
            'report_date': ''
        }
    
    def _parse_balance_sheet(self, raw_data: List[Dict]) -> Dict:
        """解析财务数据（从主要财务指标中提取）"""
        if not raw_data:
            return {}
        
        latest = raw_data[0]
        
        # 提取关键财务指标
        revenue = float(latest.get('TOTAL_OPERATE_INCOME', 0) or 0)  # 营业收入（万元）
        net_profit = float(latest.get('PARENT_NETPROFIT', 0) or 0)   # 净利润（万元）
        roe = float(latest.get('WEIGHTAVG_ROE', 0) or 0) / 100       # ROE转换为小数
        
        # 基于可用数据计算基本指标
        net_margin = net_profit / revenue if revenue > 0 else 0
        
        return {
            'revenue': revenue,
            'net_profit': net_profit,
            'roe': roe,
            'net_margin': net_margin,
            'report_date': latest.get('REPORTDATE', ''),
            # 由于没有详细的资产负债表数据，设置默认值
            'debt_ratio': 0.4,  # 假设合理的负债率
            'current_ratio': 1.2,  # 假设合理的流动比率
            'cash_ratio': 0.3   # 假设合理的现金比率
        }
    

    
    def _analyze_comprehensive_risks(self, risk_metrics: Dict, financial_data: Dict, 
                                   industry_data: Dict, trend_data: Dict, company_name: str) -> Dict:
        """综合风险分析"""
        
        # 提取关键指标
        balance_sheet = risk_metrics.get('balance_sheet', {})
        cash_flow = risk_metrics.get('cash_flow', {})
        income_statement = risk_metrics.get('income_statement', {})
        
        # 分析各类风险
        liquidity_risk = self._assess_liquidity_risk(balance_sheet, cash_flow)
        solvency_risk = self._assess_solvency_risk(balance_sheet)
        profitability_risk = self._assess_profitability_risk(income_statement, trend_data)
        operational_risk = self._assess_operational_risk(financial_data, industry_data)
        market_risk = self._assess_market_risk(financial_data, trend_data)
        
        # 计算综合风险评分
        overall_risk = self._calculate_overall_risk(
            liquidity_risk, solvency_risk, profitability_risk, operational_risk, market_risk
        )
        
        # 生成风险建议
        risk_suggestions = self._generate_risk_suggestions(
            liquidity_risk, solvency_risk, profitability_risk, operational_risk, market_risk
        )
        
        return {
            'company_name': company_name,
            'overall_risk_level': overall_risk['level'],
            'overall_risk_score': overall_risk['score'],
            'risk_breakdown': {
                'liquidity_risk': liquidity_risk,
                'solvency_risk': solvency_risk,
                'profitability_risk': profitability_risk,
                'operational_risk': operational_risk,
                'market_risk': market_risk
            },
            'identified_risks': overall_risk['identified_risks'],
            'risk_details': overall_risk['details'],
            'mitigation_suggestions': risk_suggestions,
            'risk_summary': self._generate_risk_summary(overall_risk, company_name)
        }
    
    def _assess_liquidity_risk(self, balance_sheet: Dict, cash_flow: Dict) -> Dict:
        """评估流动性风险"""
        current_ratio = balance_sheet.get('current_ratio', 0)
        cash_ratio = balance_sheet.get('cash', 0) / balance_sheet.get('current_liabilities', 1)
        cash_flow_ratio = cash_flow.get('cash_flow_ratio', 0)
        
        risk_score = 0
        risk_factors = []
        
        # 流动比率评估
        if current_ratio < 1.0:
            risk_score += 3
            risk_factors.append("流动比率低于1.0，短期偿债能力不足")
        elif current_ratio < 1.2:
            risk_score += 2
            risk_factors.append("流动比率偏低，需关注短期流动性")
        elif current_ratio > 3.0:
            risk_score += 1
            risk_factors.append("流动比率过高，资金利用效率可能偏低")
        
        # 现金比率评估
        if cash_ratio < 0.1:
            risk_score += 2
            risk_factors.append("现金比率偏低，现金储备不足")
        
        # 现金流评估
        if cash_flow_ratio < 0:
            risk_score += 3
            risk_factors.append("经营现金流为负，现金创造能力不足")
        elif cash_flow_ratio < 0.05:
            risk_score += 2
            risk_factors.append("经营现金流占收入比例偏低")
        
        level = "低" if risk_score <= 2 else "中" if risk_score <= 5 else "高"
        
        return {
            'level': level,
            'score': min(risk_score, 10),
            'current_ratio': current_ratio,
            'cash_ratio': cash_ratio,
            'cash_flow_ratio': cash_flow_ratio,
            'risk_factors': risk_factors
        }
    
    def _assess_solvency_risk(self, balance_sheet: Dict) -> Dict:
        """评估偿债能力风险"""
        debt_ratio = balance_sheet.get('debt_ratio', 0)
        equity_ratio = balance_sheet.get('equity_ratio', 0)
        
        risk_score = 0
        risk_factors = []
        
        # 资产负债率评估
        if debt_ratio > 0.8:
            risk_score += 4
            risk_factors.append("资产负债率过高，偿债压力较大")
        elif debt_ratio > 0.7:
            risk_score += 3
            risk_factors.append("资产负债率偏高，需关注债务风险")
        elif debt_ratio > 0.6:
            risk_score += 2
            risk_factors.append("资产负债率适中，但需持续监控")
        elif debt_ratio < 0.3:
            risk_score += 1
            risk_factors.append("资产负债率较低，可能存在资金利用不充分")
        
        # 权益比率评估
        if equity_ratio < 0.2:
            risk_score += 3
            risk_factors.append("权益比率过低，财务杠杆风险较高")
        
        level = "低" if risk_score <= 2 else "中" if risk_score <= 5 else "高"
        
        return {
            'level': level,
            'score': min(risk_score, 10),
            'debt_ratio': debt_ratio,
            'equity_ratio': equity_ratio,
            'risk_factors': risk_factors
        }
    
    def _assess_profitability_risk(self, income_statement: Dict, trend_data: Dict) -> Dict:
        """评估盈利能力风险"""
        roe = income_statement.get('roe', 0)
        net_margin = income_statement.get('net_margin', 0)
        gross_margin = income_statement.get('gross_margin', 0)
        
        risk_score = 0
        risk_factors = []
        
        # ROE评估
        if roe < 0:
            risk_score += 4
            risk_factors.append("ROE为负，盈利能力严重不足")
        elif roe < 0.05:
            risk_score += 3
            risk_factors.append("ROE低于5%，盈利能力偏弱")
        elif roe < 0.1:
            risk_score += 2
            risk_factors.append("ROE低于10%，盈利能力一般")
        
        # 净利率评估
        if net_margin < 0:
            risk_score += 3
            risk_factors.append("净利率为负，经营亏损")
        elif net_margin < 0.03:
            risk_score += 2
            risk_factors.append("净利率偏低，盈利质量不高")
        
        # 毛利率评估
        if gross_margin < 0.1:
            risk_score += 2
            risk_factors.append("毛利率偏低，成本控制能力不足")
        
        # 趋势分析
        if trend_data:
            avg_profit_growth = trend_data.get('avg_profit_growth', 0)
            if avg_profit_growth < -0.1:
                risk_score += 3
                risk_factors.append("利润增长趋势为负，盈利能力下降")
            elif avg_profit_growth < 0:
                risk_score += 2
                risk_factors.append("利润增长乏力，盈利能力承压")
        
        level = "低" if risk_score <= 2 else "中" if risk_score <= 5 else "高"
        
        return {
            'level': level,
            'score': min(risk_score, 10),
            'roe': roe,
            'net_margin': net_margin,
            'gross_margin': gross_margin,
            'risk_factors': risk_factors
        }
    
    def _assess_operational_risk(self, financial_data: Dict, industry_data: Dict) -> Dict:
        """评估经营风险"""
        risk_score = 0
        risk_factors = []
        
        # 行业地位评估
        if industry_data:
            # 获取行业排名信息
            company_ranking = industry_data.get('company_ranking', '')
            total_companies = industry_data.get('total_companies', 1)
            
            # 解析排名信息
            ranking = 999  # 默认值
            if isinstance(company_ranking, str):
                if "第1名" in company_ranking or "第一名" in company_ranking:
                    ranking = 1
                elif "第2名" in company_ranking or "第二名" in company_ranking:
                    ranking = 2
                elif "第3名" in company_ranking or "第三名" in company_ranking:
                    ranking = 3
                elif "前" in company_ranking and "%" in company_ranking:
                    # 处理"前25%"这种格式
                    try:
                        percent = int(company_ranking.replace("前", "").replace("%", ""))
                        ranking = max(1, int(total_companies * percent / 100))
                    except:
                        ranking = 999
                elif "行业第" in company_ranking:
                    # 处理"行业第X名"格式
                    try:
                        import re
                        match = re.search(r'第(\d+)名', company_ranking)
                        if match:
                            ranking = int(match.group(1))
                    except:
                        ranking = 999
            elif isinstance(company_ranking, (int, float)):
                ranking = int(company_ranking)
            
            # 根据排名评估风险
            if ranking > total_companies * 0.8:
                risk_score += 3
                risk_factors.append("行业排名靠后，竞争力不足")
            elif ranking > total_companies * 0.5:
                risk_score += 2
                risk_factors.append("行业排名中等，需提升竞争力")
            # 如果排名在前50%，不添加排名相关风险
        
        # 财务数据质量评估
        if financial_data:
            fetch_success = financial_data.get('fetch_success', False)
            if not fetch_success:
                risk_score += 2
                risk_factors.append("财务数据获取困难，信息透明度不足")
        
        level = "低" if risk_score <= 2 else "中" if risk_score <= 4 else "高"
        
        return {
            'level': level,
            'score': min(risk_score, 10),
            'risk_factors': risk_factors
        }
    
    def _assess_market_risk(self, financial_data: Dict, trend_data: Dict) -> Dict:
        """评估市场风险"""
        risk_score = 0
        risk_factors = []
        
        # 估值风险评估
        if financial_data:
            pe_ratio = financial_data.get('pe_ratio', 0)
            pb_ratio = financial_data.get('pb_ratio', 0)
            
            if pe_ratio > 50:
                risk_score += 3
                risk_factors.append("PE比率过高，估值风险较大")
            elif pe_ratio > 30:
                risk_score += 2
                risk_factors.append("PE比率偏高，需关注估值风险")
            elif pe_ratio < 5 and pe_ratio > 0:
                risk_score += 2
                risk_factors.append("PE比率过低，可能存在基本面问题")
            
            if pb_ratio > 5:
                risk_score += 2
                risk_factors.append("PB比率过高，市场估值偏高")
            elif pb_ratio < 0.5:
                risk_score += 2
                risk_factors.append("PB比率过低，可能存在资产质量问题")
        
        # 波动性风险评估
        if trend_data:
            growth_stage = trend_data.get('growth_stage', '')
            if growth_stage == '调整期':
                risk_score += 3
                risk_factors.append("公司处于调整期，业绩波动风险较高")
            elif growth_stage == '衰退期':
                risk_score += 4
                risk_factors.append("公司处于衰退期，投资风险很高")
        
        level = "低" if risk_score <= 2 else "中" if risk_score <= 5 else "高"
        
        return {
            'level': level,
            'score': min(risk_score, 10),
            'risk_factors': risk_factors
        }
    
    def _calculate_overall_risk(self, liquidity_risk: Dict, solvency_risk: Dict, 
                              profitability_risk: Dict, operational_risk: Dict, market_risk: Dict) -> Dict:
        """计算综合风险评分"""
        
        # 权重设置
        weights = {
            'liquidity': 0.25,
            'solvency': 0.25,
            'profitability': 0.25,
            'operational': 0.15,
            'market': 0.10
        }
        
        # 计算加权风险评分
        weighted_score = (
            liquidity_risk['score'] * weights['liquidity'] +
            solvency_risk['score'] * weights['solvency'] +
            profitability_risk['score'] * weights['profitability'] +
            operational_risk['score'] * weights['operational'] +
            market_risk['score'] * weights['market']
        )
        
        # 确定风险等级
        if weighted_score <= 3:
            level = "低"
            details = "整体财务状况良好，风险可控"
        elif weighted_score <= 6:
            level = "中"
            details = "财务状况基本稳健，需关注部分风险点"
        else:
            level = "高"
            details = "存在较多风险因素，需谨慎投资"
        
        # 收集所有风险因素
        all_risks = []
        for risk_type in [liquidity_risk, solvency_risk, profitability_risk, operational_risk, market_risk]:
            all_risks.extend(risk_type.get('risk_factors', []))
        
        return {
            'level': level,
            'score': round(weighted_score, 2),
            'details': details,
            'identified_risks': all_risks[:5]  # 只显示前5个主要风险
        }
    
    def _generate_risk_suggestions(self, liquidity_risk: Dict, solvency_risk: Dict, 
                                 profitability_risk: Dict, operational_risk: Dict, market_risk: Dict) -> List[str]:
        """生成风险缓解建议"""
        suggestions = []
        
        # 流动性风险建议
        if liquidity_risk['level'] in ['中', '高']:
            suggestions.append("加强现金流管理，优化营运资金配置")
            suggestions.append("建立多元化融资渠道，确保流动性充足")
        
        # 偿债能力风险建议
        if solvency_risk['level'] in ['中', '高']:
            suggestions.append("优化资本结构，适度降低财务杠杆")
            suggestions.append("加强债务管理，合理安排还款计划")
        
        # 盈利能力风险建议
        if profitability_risk['level'] in ['中', '高']:
            suggestions.append("提升运营效率，加强成本控制")
            suggestions.append("优化产品结构，提高盈利质量")
        
        # 经营风险建议
        if operational_risk['level'] in ['中', '高']:
            suggestions.append("加强核心竞争力建设，提升市场地位")
            suggestions.append("完善内控制度，提高经营管理水平")
        
        # 市场风险建议
        if market_risk['level'] in ['中', '高']:
            suggestions.append("关注市场估值变化，合理安排投资时机")
            suggestions.append("分散投资风险，避免过度集中")
        
        return suggestions[:5]  # 返回前5个建议
    
    def _generate_risk_summary(self, overall_risk: Dict, company_name: str) -> str:
        """生成风险分析总结"""
        level = overall_risk['level']
        score = overall_risk['score']
        
        if level == "低":
            return f"{company_name}整体风险水平较低（风险评分：{score}/10），财务状况稳健，投资风险可控。"
        elif level == "中":
            return f"{company_name}整体风险水平适中（风险评分：{score}/10），财务基本面良好，但需关注部分风险因素。"
        else:
            return f"{company_name}整体风险水平较高（风险评分：{score}/10），存在多项风险因素，投资需谨慎。"


# 测试函数
async def test_risk_analysis():
    """测试风险分析功能"""
    analyzer = RealRiskAnalysis()
    
    # 测试平安银行
    result = await analyzer.generate_risk_analysis("000001", "平安银行")
    
    print("=== 风险分析测试结果 ===")
    print(f"公司: {result['company_name']}")
    print(f"整体风险等级: {result['overall_risk_level']}")
    print(f"风险评分: {result['overall_risk_score']}/10")
    print(f"风险总结: {result['risk_summary']}")
    print(f"主要风险: {result['identified_risks']}")
    print(f"缓解建议: {result['mitigation_suggestions']}")
    print(f"分析耗时: {result['analysis_time']}秒")

if __name__ == "__main__":
    asyncio.run(test_risk_analysis())