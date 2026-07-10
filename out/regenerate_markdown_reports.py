#!/usr/bin/env python3
"""
重新生成改进的Markdown报告
"""
import json
import os
from datetime import datetime
from pathlib import Path

def generate_improved_markdown_report(json_file_path: Path, stock_name: str, stock_code: str):
    """生成改进的Markdown格式的分析报告"""
    
    # 读取JSON报告
    with open(json_file_path, 'r', encoding='utf-8') as f:
        analysis_result = json.load(f)
    
    # 获取最终报告
    final_report = analysis_result.get('final_report', {})
    
    # 构建Markdown内容
    markdown_content = f"""# {stock_name} ({stock_code}) 财报分析报告

## 基本信息
- **股票名称**: {stock_name}
- **股票代码**: {stock_code}
- **分析日期**: {analysis_result.get('analysis_date', 'N/A')}
- **分析状态**: {analysis_result.get('status', 'N/A')}

## 分析步骤完成情况
"""
    
    # 添加分析步骤信息
    steps_completed = analysis_result.get('steps_completed', [])
    for step in steps_completed:
        status_emoji = "✅" if step.get('status') == 'completed' else "❌"
        markdown_content += f"- {status_emoji} 步骤{step.get('step', 'N/A')}: {step.get('name', 'N/A')}\n"
    
    # 添加最终报告内容
    if final_report:
        # 执行摘要
        executive_summary = final_report.get('executive_summary', '暂无数据')
        markdown_content += f"""
## 执行摘要
{executive_summary}

## 财务亮点
"""
        
        # 财务亮点
        financial_highlights = final_report.get('financial_highlights', {})
        if financial_highlights:
            markdown_content += f"""
- **营业收入**: {financial_highlights.get('revenue', 'N/A')}
- **净利润**: {financial_highlights.get('net_profit', 'N/A')}
- **ROE**: {financial_highlights.get('roe', 'N/A')}
- **市盈率**: {financial_highlights.get('pe_ratio', 'N/A')}
"""
        
        # 行业地位
        industry_position = final_report.get('industry_position', '暂无数据')
        markdown_content += f"""
## 行业地位
{industry_position}

## 主要风险
"""
        
        # 主要风险
        key_risks = final_report.get('key_risks', [])
        if key_risks:
            for risk in key_risks:
                markdown_content += f"- {risk}\n"
        else:
            markdown_content += "暂无风险信息\n"
        
        # 投资建议
        investment_recommendation = final_report.get('investment_recommendation', '暂无建议')
        analyst_rating = final_report.get('analyst_rating', 'N/A')
        confidence_level = final_report.get('confidence_level', 'N/A')
        
        markdown_content += f"""
## 投资建议
- **建议**: {investment_recommendation}
- **评级**: {analyst_rating}
- **信心水平**: {confidence_level}

## 专业分析师报告
"""
        
        # 添加OpenAI分析内容
        openai_analysis = final_report.get('openai_analysis', {})
        if openai_analysis and openai_analysis.get('analysis_result', {}).get('success'):
            analysis_text = openai_analysis['analysis_result'].get('analysis', '暂无详细分析')
            # 截取前3000字符以避免过长
            if len(analysis_text) > 3000:
                analysis_text = analysis_text[:3000] + "...\n\n*（完整分析请查看JSON文件）*"
            markdown_content += f"""
{analysis_text}
"""
        else:
            markdown_content += "暂无详细分析内容\n"
            
    else:
        markdown_content += """
## 财务分析报告
暂无最终报告数据，请检查分析过程是否完成。
"""
    
    # 添加原始数据链接
    markdown_content += f"""
## 附录
- **报告日期**: {final_report.get('report_date', 'N/A')}
- **免责声明**: {final_report.get('disclaimer', '本报告仅供参考，不构成投资建议')}
- 完整分析数据请查看同目录下的JSON文件
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # 生成新的Markdown文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_filename = f"{stock_name}_改进分析报告_{timestamp}.md"
    markdown_path = json_file_path.parent / markdown_filename
    
    # 保存Markdown文件
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return markdown_path

def main():
    """主函数"""
    reports_dir = Path('out/reports')
    
    print("🔄 开始重新生成改进的Markdown报告...")
    
    # 遍历所有股票目录
    for stock_dir in reports_dir.iterdir():
        if stock_dir.is_dir() and stock_dir.name != '__pycache__':
            stock_name = stock_dir.name
            
            # 查找JSON报告文件
            json_files = list(stock_dir.glob('*_财报分析报告_*.json'))
            if json_files:
                # 使用最新的JSON文件
                latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
                
                # 从文件名中提取股票代码
                with open(latest_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 从分析结果中获取股票代码
                stock_code = 'N/A'
                steps = data.get('steps_completed', [])
                for step in steps:
                    if step.get('step') == 1 and step.get('result', {}).get('stock_code'):
                        stock_code = step['result']['stock_code']
                        break
                
                print(f"📝 正在为 {stock_name} ({stock_code}) 生成改进报告...")
                
                # 生成改进的Markdown报告
                new_markdown_path = generate_improved_markdown_report(latest_json, stock_name, stock_code)
                print(f"✅ 已生成: {new_markdown_path}")
            else:
                print(f"❌ 未找到 {stock_name} 的JSON报告文件")
    
    print("\n🎉 所有改进报告生成完成!")

if __name__ == "__main__":
    main()