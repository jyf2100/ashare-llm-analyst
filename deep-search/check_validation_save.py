#!/usr/bin/env python3
"""
检查校验结果是否正确保存到报告文件中
"""

import asyncio
import json
from main import FinancialAnalyzer

async def check_validation_save():
    """检查校验结果保存"""
    print("🔍 检查校验结果保存...")
    
    # 运行分析
    analyzer = FinancialAnalyzer()
    result = await analyzer.analyze_company("大金重工")
    
    # 检查结果结构
    print("\n📋 结果结构:")
    print(f"   状态: {result.get('status')}")
    print(f"   包含的键: {list(result.keys())}")
    
    # 检查校验结果
    if 'validation_result' in result:
        validation = result['validation_result']
        print(f"\n✅ 校验结果存在:")
        print(f"   校验分数: {validation.get('validation_score', 'N/A')}")
        print(f"   修正数量: {len(validation.get('corrections_made', []))}")
        print(f"   校验详情键: {list(validation.get('validation_details', {}).keys())}")
    else:
        print("\n❌ 校验结果不存在")
    
    # 检查最终报告中的校验信息
    final_report = result.get('final_report', {})
    if 'validation_info' in final_report:
        val_info = final_report['validation_info']
        print(f"\n✅ 最终报告包含校验信息:")
        print(f"   校验日期: {val_info.get('validation_date', 'N/A')}")
        print(f"   应用修正: {len(val_info.get('corrections_applied', []))}")
        print(f"   校验分数: {val_info.get('validation_scores', {})}")
    else:
        print(f"\n❌ 最终报告不包含校验信息")
        print(f"   最终报告键: {list(final_report.keys())}")
    
    # 保存测试文件
    test_file = "test_validation_save.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n📄 测试结果已保存到: {test_file}")
    
    return result

if __name__ == "__main__":
    asyncio.run(check_validation_save())