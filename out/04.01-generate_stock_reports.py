#!/usr/bin/env python3
"""
股票分析报告生成器
读取.env配置文件中的股票信息，使用deep-search/main.py的财报分析功能生成报告
"""
import asyncio
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# 添加deep-search目录到Python路径
# 获取项目根目录（out目录的上级目录）
project_root = os.path.dirname(os.path.dirname(__file__))
deep_search_path = os.path.join(project_root, 'deep-search')
sys.path.insert(0, deep_search_path)

# 导入财报分析器
try:
    from main import FinancialAnalyzer
except ImportError as e:
    print(f"❌ 导入FinancialAnalyzer失败: {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"deep-search路径: {deep_search_path}")
    print(f"路径是否存在: {os.path.exists(deep_search_path)}")
    sys.exit(1)

class StockReportGenerator:
    """股票报告生成器"""
    
    def __init__(self):
        print("🚀 启动股票分析报告生成器")
        
        # 确定当前工作目录和项目根目录
        current_dir = Path.cwd()
        if current_dir.name == 'out':
            # 如果在out目录下运行，使用相对路径
            env_path = '.env'
            reports_path = 'reports'
        else:
            # 如果在项目根目录下运行，使用out子目录
            env_path = 'out/.env'
            reports_path = 'out/reports'
        
        # 加载环境变量
        load_dotenv(env_path)
        
        # 初始化财报分析器
        self.analyzer = FinancialAnalyzer()
        
        # 设置报告输出目录
        self.reports_dir = Path(reports_path)
        self.reports_dir.mkdir(exist_ok=True)
        
        print("🚀 股票报告生成器初始化完成")
    
    def load_stock_configs(self):
        """加载股票配置"""
        stocks = {}
        
        # 加载自动选择的股票配置
        stocks_config = os.getenv('STOCKS_CONFIG', '{}')
        try:
            auto_stocks = json.loads(stocks_config)
            stocks.update(auto_stocks)
            print(f"📊 加载自动选择股票: {len(auto_stocks)} 只")
        except json.JSONDecodeError as e:
            print(f"❌ 解析STOCKS_CONFIG失败: {e}")
        
        # 加载自定义股票配置
        stocks_config_self = os.getenv('STOCKS_CONFIG_SELF', '{}')
        try:
            self_stocks = json.loads(stocks_config_self)
            stocks.update(self_stocks)
            print(f"📊 加载自定义股票: {len(self_stocks)} 只")
        except json.JSONDecodeError as e:
            print(f"❌ 解析STOCKS_CONFIG_SELF失败: {e}")
        
        print(f"📈 总共加载股票: {len(stocks)} 只")
        return stocks
    
    async def generate_report_for_stock(self, stock_name: str, stock_code: str):
        """为单个股票生成分析报告"""
        print(f"\n🔍 开始分析股票: {stock_name} ({stock_code})")
        print("=" * 60)
        
        try:
            # 使用财报分析器分析公司
            analysis_result = await self.analyzer.analyze_company(stock_name)
            
            # 创建股票专用目录
            stock_dir = self.reports_dir / stock_name
            stock_dir.mkdir(exist_ok=True)
            
            # 生成报告文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{stock_name}_财报分析报告_{timestamp}.json"
            report_path = stock_dir / report_filename
            
            # 保存完整分析结果
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            
            # 生成简化的Markdown报告
            markdown_filename = f"{stock_name}_分析报告_{timestamp}.md"
            markdown_path = stock_dir / markdown_filename
            
            await self.generate_markdown_report(analysis_result, markdown_path, stock_name, stock_code)
            
            print(f"✅ {stock_name} 分析完成")
            print(f"   📄 JSON报告: {report_path}")
            print(f"   📝 Markdown报告: {markdown_path}")
            
            return {
                'stock_name': stock_name,
                'stock_code': stock_code,
                'status': 'success',
                'json_report': str(report_path),
                'markdown_report': str(markdown_path),
                'analysis_result': analysis_result
            }
            
        except Exception as e:
            print(f"❌ {stock_name} 分析失败: {str(e)}")
            return {
                'stock_name': stock_name,
                'stock_code': stock_code,
                'status': 'failed',
                'error': str(e)
            }
    
    async def generate_markdown_report(self, analysis_result: dict, output_path: Path, stock_name: str, stock_code: str):
        """生成Markdown格式的分析报告"""
        
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
                # 截取前2000字符以避免过长
                if len(analysis_text) > 2000:
                    analysis_text = analysis_text[:2000] + "...\n\n*（完整分析请查看JSON文件）*"
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
        
        # 保存Markdown文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    async def generate_all_reports(self):
        """为所有配置的股票生成分析报告"""
        stocks = self.load_stock_configs()
        
        if not stocks:
            print("❌ 未找到股票配置，请检查.env文件")
            return
        
        print(f"\n🚀 开始为 {len(stocks)} 只股票生成分析报告")
        print("=" * 60)
        
        results = []
        
        for stock_name, stock_code in stocks.items():
            result = await self.generate_report_for_stock(stock_name, stock_code)
            results.append(result)
            
            # 短暂延迟，避免API请求过于频繁
            await asyncio.sleep(2)
        
        # 生成汇总报告
        await self.generate_summary_report(results)
        
        print(f"\n🎉 所有股票分析完成!")
        print(f"📊 成功: {len([r for r in results if r['status'] == 'success'])} 只")
        print(f"❌ 失败: {len([r for r in results if r['status'] == 'failed'])} 只")
        print(f"📁 报告保存在: {self.reports_dir}")
    
    async def generate_summary_report(self, results: list):
        """生成汇总报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.reports_dir / f"汇总报告_{timestamp}.md"
        
        summary_content = f"""# 股票分析汇总报告

## 分析概况
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **分析股票数量**: {len(results)}
- **成功分析**: {len([r for r in results if r['status'] == 'success'])} 只
- **分析失败**: {len([r for r in results if r['status'] == 'failed'])} 只

## 分析结果详情

### 成功分析的股票
"""
        
        success_results = [r for r in results if r['status'] == 'success']
        for result in success_results:
            summary_content += f"- ✅ **{result['stock_name']}** ({result['stock_code']})\n"
            summary_content += f"  - JSON报告: `{result['json_report']}`\n"
            summary_content += f"  - Markdown报告: `{result['markdown_report']}`\n\n"
        
        failed_results = [r for r in results if r['status'] == 'failed']
        if failed_results:
            summary_content += "### 分析失败的股票\n"
            for result in failed_results:
                summary_content += f"- ❌ **{result['stock_name']}** ({result['stock_code']})\n"
                summary_content += f"  - 错误信息: {result.get('error', 'N/A')}\n\n"
        
        summary_content += f"""
## 报告说明
- 每只股票的详细分析报告保存在对应的股票名称目录中
- JSON文件包含完整的分析数据和步骤信息
- Markdown文件提供易读的分析报告格式
- 所有报告文件都包含时间戳以便版本管理
"""
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"📋 汇总报告已生成: {summary_path}")

async def main():
    """主函数"""
    print("🚀 启动股票分析报告生成器")
    
    # 设置代理环境变量
    os.environ['http_proxy'] = 'http://172.32.147.190:7890'
    os.environ['https_proxy'] = 'http://172.32.147.190:7890'
    
    generator = StockReportGenerator()
    await generator.generate_all_reports()

if __name__ == "__main__":
    asyncio.run(main())