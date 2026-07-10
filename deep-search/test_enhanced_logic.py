#!/usr/bin/env python3
"""
测试优化后的知识库+网络查询逻辑
验证知识库数据优先级高于网络查询数据
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import FinancialAnalyzer

async def test_enhanced_logic():
    """测试增强分析逻辑"""
    print("=" * 60)
    print("测试优化后的知识库+网络查询逻辑")
    print("=" * 60)
    
    analyzer = FinancialAnalyzer()
    
    # 测试公司信息
    test_company = "平安银行"
    test_stock_code = "000001"
    
    print(f"\n1. 测试公司: {test_company} ({test_stock_code})")
    print("-" * 40)
    
    try:
        # 测试增强洞察获取（知识库优先）
        print("正在获取增强洞察（知识库优先）...")
        insights = await analyzer._get_enhanced_insights_with_priority(test_company, test_stock_code)
        
        print(f"获取到 {len(insights)} 条洞察信息")
        for i, insight in enumerate(insights, 1):
            query = insight.get('query', '')
            source = insight.get('source', '')
            response = insight.get('response', '')[:100] + "..." if insight.get('response') else "无响应"
            print(f"  {i}. 查询: {query}")
            print(f"     来源: {source}")
            print(f"     响应: {response}")
            print()
        
        # 测试来源分类
        print("\n2. 测试来源分类...")
        print("-" * 40)
        categorized = analyzer._categorize_sources(insights)
        
        kb_count = len(categorized.get('knowledge_base_sources', []))
        web_count = len(categorized.get('web_search_sources', []))
        
        print(f"知识库来源: {kb_count} 条")
        print(f"网络搜索来源: {web_count} 条")
        
        # 显示分类详情
        if kb_count > 0:
            print("\n知识库来源详情:")
            for i, source in enumerate(categorized['knowledge_base_sources'], 1):
                print(f"  {i}. {source.get('query', '')}")
        
        if web_count > 0:
            print("\n网络搜索来源详情:")
            for i, source in enumerate(categorized['web_search_sources'], 1):
                print(f"  {i}. {source.get('query', '')}")
        
        # 测试摘要生成
        print("\n3. 测试摘要生成...")
        print("-" * 40)
        summary = analyzer._generate_enhanced_summary(categorized)
        print(f"生成的摘要: {summary}")
        
        # 测试完整的MCP增强分析流程
        print("\n4. 测试完整MCP增强分析流程...")
        print("-" * 40)
        
        # 模拟上下文数据
        mock_context = {
            'company_info': {'name': test_company, 'stock_code': test_stock_code},
            'final_report': {'summary': '测试报告'},
            'metrics': {'revenue': 1000000}
        }
        
        result = await analyzer._step9_mcp_enhance_analysis(
            mock_context['company_info'],
            mock_context['final_report'],
            mock_context['metrics'],
            mock_context
        )
        
        print("MCP增强分析结果:")
        print(f"  状态: {result.get('status', 'unknown')}")
        print(f"  摘要: {result.get('summary', 'N/A')}")
        
        enhanced_insights = result.get('enhanced_insights', {})
        if isinstance(enhanced_insights, dict):
            kb_sources = enhanced_insights.get('knowledge_base_sources', [])
            web_sources = enhanced_insights.get('web_search_sources', [])
            print(f"  知识库洞察: {len(kb_sources)} 条")
            print(f"  网络洞察: {len(web_sources)} 条")
        
        print("\n✅ 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_logic())