#!/usr/bin/env python3
"""
分析用户定义的股票
从 .env 文件读取 STOCKS_CONFIG_SELF 配置并进行分析
"""
import asyncio
import json
import os
from dotenv import load_dotenv
from complete_real_analyzer import CompleteRealAnalyzer

# 加载环境变量
load_dotenv()

async def analyze_user_defined_stocks():
    """分析用户定义的股票"""
    print("🎯 开始分析用户定义的股票")
    print("="*60)
    
    # 从环境变量读取股票配置
    stocks_config = os.getenv('STOCKS_CONFIG_SELF', '{}')
    
    try:
        stocks_dict = json.loads(stocks_config)
        print(f"📋 用户定义的股票列表: {stocks_dict}")
    except json.JSONDecodeError:
        print("❌ 股票配置格式错误")
        return
    
    if not stocks_dict:
        print("❌ 未找到用户定义的股票")
        return
    
    # 初始化分析器
    analyzer = CompleteRealAnalyzer()
    
    # 分析每只股票
    for stock_name, stock_code in stocks_dict.items():
        print(f"\n{'='*60}")
        print(f"🔍 正在分析: {stock_name} ({stock_code})")
        print(f"{'='*60}")
        
        try:
            # 执行完整分析
            result = await analyzer.complete_analysis(stock_name)
            
            # 显示分析摘要
            print(f"\n📊 {stock_name} 分析摘要")
            print("-" * 40)
            
            # 公司信息
            company_info = result.get('company_info', {})
            print(f"公司名称: {company_info.get('name', '未知')}")
            print(f"股票代码: {company_info.get('code', '未知')}")
            print(f"所属行业: {company_info.get('industry', '未知')}")
            
            # 财务数据
            financial_data = result.get('financial_data', {})
            if financial_data.get('fetch_success'):
                print(f"当前股价: {financial_data.get('current_price', 0):.2f} 元")
                print(f"涨跌幅: {financial_data.get('change_percent', 0):.2f}%")
                print(f"市盈率: {financial_data.get('pe_ratio', 0):.2f}")
                print(f"市净率: {financial_data.get('pb_ratio', 0):.2f}")
            
            # 投资建议
            investment_advice = result.get('investment_advice', {})
            print(f"投资建议: {investment_advice.get('recommendation', '未知')}")
            print(f"风险等级: {investment_advice.get('risk_level', '未知')}")
            print(f"置信度: {investment_advice.get('confidence_score', 0)}/100")
            
            # AI分析成功检查
            ai_analysis = result.get('ai_analysis', {})
            if ai_analysis.get('analysis_success'):
                print("✅ AI深度分析完成")
                # 检查时间逻辑
                analysis_content = ai_analysis.get('analysis_content', '')
                if '2025年' in analysis_content:
                    print("✅ 时间逻辑正确 (包含2025年)")
                if '结合2024年' not in analysis_content:
                    print("✅ 避免了错误的时间表述")
            else:
                print("❌ AI分析失败")
            
            print(f"分析耗时: {result.get('analysis_metadata', {}).get('total_duration', '未知')}")
            
        except Exception as e:
            print(f"❌ {stock_name} 分析失败: {e}")
        
        # 避免请求过快
        print("\n⏳ 等待3秒后分析下一只股票...")
        await asyncio.sleep(3)
    
    print(f"\n{'='*60}")
    print("🎉 所有用户定义股票分析完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(analyze_user_defined_stocks())