#!/usr/bin/env python3
"""
验证股票分析报告的质量和完整性
"""
import json
import os
from pathlib import Path
from datetime import datetime

def validate_json_report(json_file_path: Path):
    """验证JSON报告的完整性"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查必要字段
        required_fields = ['analysis_date', 'status', 'steps_completed', 'final_report']
        missing_fields = []
        
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
        
        # 检查步骤完成情况
        steps_completed = data.get('steps_completed', [])
        completed_steps = [step for step in steps_completed if step.get('status') == 'completed']
        
        # 检查最终报告
        final_report = data.get('final_report', {})
        has_openai_analysis = bool(final_report.get('openai_analysis', {}).get('analysis_result', {}).get('analysis'))
        
        return {
            'valid': len(missing_fields) == 0,
            'missing_fields': missing_fields,
            'total_steps': len(steps_completed),
            'completed_steps': len(completed_steps),
            'has_final_report': bool(final_report),
            'has_openai_analysis': has_openai_analysis,
            'file_size_kb': json_file_path.stat().st_size / 1024
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'file_size_kb': 0
        }

def validate_markdown_report(md_file_path: Path):
    """验证Markdown报告的完整性"""
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键章节
        required_sections = [
            '基本信息',
            '分析步骤完成情况',
            '执行摘要',
            '财务亮点',
            '投资建议'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        # 检查是否有实际内容（不只是"暂无数据"）
        has_meaningful_content = (
            '暂无数据' not in content[:1000] or  # 前1000字符中没有"暂无数据"
            len(content) > 2000  # 或者内容足够长
        )
        
        return {
            'valid': len(missing_sections) == 0,
            'missing_sections': missing_sections,
            'has_meaningful_content': has_meaningful_content,
            'content_length': len(content),
            'file_size_kb': md_file_path.stat().st_size / 1024
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'content_length': 0,
            'file_size_kb': 0
        }

def main():
    """主函数"""
    reports_dir = Path('out/reports')
    
    print("🔍 开始验证股票分析报告质量...")
    print("=" * 80)
    
    total_stocks = 0
    valid_reports = 0
    
    # 遍历所有股票目录
    for stock_dir in reports_dir.iterdir():
        if stock_dir.is_dir() and stock_dir.name != '__pycache__':
            stock_name = stock_dir.name
            total_stocks += 1
            
            print(f"\n📊 验证 {stock_name} 的报告:")
            print("-" * 50)
            
            # 查找JSON和Markdown文件
            json_files = list(stock_dir.glob('*_财报分析报告_*.json'))
            md_files = list(stock_dir.glob('*_改进分析报告_*.md'))
            
            if not json_files:
                print(f"❌ 未找到JSON报告文件")
                continue
                
            if not md_files:
                print(f"❌ 未找到改进的Markdown报告文件")
                continue
            
            # 验证最新的JSON文件
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            json_validation = validate_json_report(latest_json)
            
            print(f"📄 JSON报告验证:")
            print(f"   ✅ 文件有效: {json_validation['valid']}")
            if not json_validation['valid']:
                if 'error' in json_validation:
                    print(f"   ❌ 错误: {json_validation['error']}")
                if 'missing_fields' in json_validation:
                    print(f"   ❌ 缺失字段: {json_validation['missing_fields']}")
            else:
                print(f"   📈 完成步骤: {json_validation['completed_steps']}/{json_validation['total_steps']}")
                print(f"   📊 有最终报告: {json_validation['has_final_report']}")
                print(f"   🤖 有AI分析: {json_validation['has_openai_analysis']}")
                print(f"   💾 文件大小: {json_validation['file_size_kb']:.1f} KB")
            
            # 验证最新的Markdown文件
            latest_md = max(md_files, key=lambda x: x.stat().st_mtime)
            md_validation = validate_markdown_report(latest_md)
            
            print(f"📝 Markdown报告验证:")
            print(f"   ✅ 文件有效: {md_validation['valid']}")
            if not md_validation['valid']:
                if 'error' in md_validation:
                    print(f"   ❌ 错误: {md_validation['error']}")
                if 'missing_sections' in md_validation:
                    print(f"   ❌ 缺失章节: {md_validation['missing_sections']}")
            else:
                print(f"   📖 有意义内容: {md_validation['has_meaningful_content']}")
                print(f"   📏 内容长度: {md_validation['content_length']} 字符")
                print(f"   💾 文件大小: {md_validation['file_size_kb']:.1f} KB")
            
            # 判断整体质量
            if (json_validation['valid'] and md_validation['valid'] and 
                json_validation.get('completed_steps', 0) >= 8 and
                json_validation.get('has_openai_analysis', False) and
                md_validation.get('has_meaningful_content', False)):
                print(f"🎉 {stock_name} 报告质量: 优秀")
                valid_reports += 1
            else:
                print(f"⚠️  {stock_name} 报告质量: 需要改进")
    
    # 总结
    print("\n" + "=" * 80)
    print(f"📋 验证总结:")
    print(f"   📊 总股票数: {total_stocks}")
    print(f"   ✅ 优秀报告: {valid_reports}")
    print(f"   📈 成功率: {valid_reports/total_stocks*100:.1f}%" if total_stocks > 0 else "   📈 成功率: 0%")
    
    if valid_reports == total_stocks:
        print(f"🎉 所有报告质量优秀！")
    else:
        print(f"⚠️  还有 {total_stocks - valid_reports} 个报告需要改进")
    
    print(f"🕒 验证完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()