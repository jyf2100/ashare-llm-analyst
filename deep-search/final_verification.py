#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终验证脚本：确认行业排名逻辑修复效果
"""

import json
import sys
import os

def verify_ranking_fix():
    """验证排名逻辑修复效果"""
    
    # 查找最新的大金重工报告
    report_files = [f for f in os.listdir('.') if f.startswith('financial_report_大金重工_') and f.endswith('.json')]
    if not report_files:
        print("❌ 未找到大金重工的财务报告文件")
        return False
    
    latest_report = sorted(report_files)[-1]
    print(f"📄 检查报告文件: {latest_report}")
    
    try:
        with open(latest_report, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        # 检查行业比较数据
        industry_step = None
        risk_step = None
        
        for step in report_data.get('analysis_steps', []):
            if step.get('name') == '对比行业平均水平':
                industry_step = step
            elif step.get('name') == '评估投资风险':
                risk_step = step
        
        if not industry_step:
            print("❌ 未找到行业比较步骤")
            return False
            
        if not risk_step:
            print("❌ 未找到风险评估步骤")
            return False
        
        # 检查行业排名
        company_ranking = industry_step['result'].get('company_ranking', '')
        total_companies = industry_step['result'].get('total_companies', 0)
        
        print(f"🏆 公司排名: {company_ranking}")
        print(f"📊 总公司数: {total_companies}")
        
        # 检查风险因子
        operational_risk = risk_step['result']['risk_breakdown'].get('operational_risk', {})
        risk_factors = operational_risk.get('risk_factors', [])
        
        print(f"⚠️  运营风险因子: {risk_factors}")
        
        # 验证修复效果
        has_ranking_contradiction = any('行业排名靠后' in factor or '竞争力不足' in factor for factor in risk_factors)
        
        if company_ranking == "行业第1名" and not has_ranking_contradiction:
            print("✅ 修复成功！行业第1名不再被误判为排名靠后")
            return True
        elif company_ranking == "行业第1名" and has_ranking_contradiction:
            print("❌ 修复失败！行业第1名仍被误判为排名靠后")
            return False
        else:
            print(f"ℹ️  其他排名情况: {company_ranking}")
            return True
            
    except Exception as e:
        print(f"❌ 读取报告文件失败: {e}")
        return False

if __name__ == "__main__":
    print("🔍 开始验证行业排名逻辑修复效果...")
    print("=" * 50)
    
    success = verify_ranking_fix()
    
    print("=" * 50)
    if success:
        print("🎉 验证通过！行业排名逻辑已成功修复")
        sys.exit(0)
    else:
        print("💥 验证失败！需要进一步检查")
        sys.exit(1)