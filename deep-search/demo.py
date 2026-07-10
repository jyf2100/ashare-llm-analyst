#!/usr/bin/env python3
"""
财报分析工具演示脚本
"""
import asyncio
import json
from main import FinancialAnalyzer

async def demo():
    """演示财报分析功能"""
    print("🚀 财报分析工具演示")
    print("=" * 40)
    
    # 测试公司列表
    test_companies = ["大金重工", "招商银行", "000001"]
    
    analyzer = FinancialAnalyzer()
    
    for company in test_companies:
        print(f"\n🔍 正在分析: {company}")
        print("-" * 30)
        
        try:
            result = await analyzer.analyze_company(company)
            
            if result.get("status") == "completed":
                final_report = result.get("final_report", {})
                print(f"✅ 分析完成")
                print(f"📊 投资建议: {final_report.get('investment_recommendation', '无')}")
                print(f"🏆 行业地位: {final_report.get('industry_position', '无')}")
                print(f"⚠️  风险等级: {result.get('steps_completed', [{}])[-2].get('result', {}).get('risk_level', '无')}")
            else:
                print(f"❌ 分析失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            print(f"💥 发生异常: {str(e)}")
        
        # 短暂延迟
        await asyncio.sleep(1)
    
    print("\n🎉 演示完成!")
    print("\n💡 使用方法:")
    print("   python3 main.py <公司名称>")
    print("   python3 demo.py")

if __name__ == "__main__":
    asyncio.run(demo())
