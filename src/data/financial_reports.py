"""
财务报告生成模块

提供财务分析报告生成功能，集成deep-search/FinancialAnalyzer进行9步财报分析。

主要功能:
- 从管道选股结果或环境变量配置加载股票列表
- 调用FinancialAnalyzer进行深度财务分析
- 生成JSON和Markdown格式的分析报告
- 生成汇总报告

Attributes:
    FinancialReportGenerator: 财务报告生成器类
"""
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.logger import get_logger

# 添加deep-search目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
deep_search_path = os.path.join(project_root, 'deep-search')
if deep_search_path not in os.sys.path:
    os.sys.path.insert(0, deep_search_path)

# 导入财报分析器
try:
    from main import FinancialAnalyzer
except ImportError as e:
    FinancialAnalyzer = None
    _import_error = str(e)
else:
    _import_error = None


logger = get_logger(__name__)


class FinancialReportGenerator(AnalyzerBase):
    """
    财务报告生成器

    使用deep-search/FinancialAnalyzer进行9步财务分析，生成JSON和Markdown报告。

    输入来源优先级:
    1. 管道选股结果 (stock_selection_results.json)
    2. 显式指定的 stock_codes 参数
    3. 环境变量配置 (STOCKS_CONFIG, STOCKS_CONFIG_SELF)

    Attributes:
        config: 配置实例
        reports_dir: 报告输出目录
        analyzer: FinancialAnalyzer实例
        timeout: 单股分析超时时间(秒)
        concurrent_limit: 并发分析数量限制

    Example:
        >>> generator = FinancialReportGenerator()
        >>> result = await generator.generate_all_reports()
        >>> print(f"成功: {result['success_count']}, 失败: {result['failed_count']}")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化财务报告生成器

        Args:
            config: 配置实例

        Raises:
            ImportError: 如果无法导入FinancialAnalyzer
        """
        super().__init__(config)

        if FinancialAnalyzer is None:
            raise ImportError(
                f"无法导入FinancialAnalyzer: {_import_error}。"
                f"请确保deep-search目录存在且包含main.py模块"
            )

        # 配置参数
        self.reports_dir = Path(self.config.data.financial_reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = self.config.data.financial_analysis_timeout
        self.concurrent_limit = self.config.data.concurrent_reports

        # 延迟初始化FinancialAnalyzer
        self._analyzer: Optional['FinancialAnalyzer'] = None

        self.logger.info(f"财务报告生成器初始化完成，输出目录: {self.reports_dir}")

    @property
    def analyzer(self) -> 'FinancialAnalyzer':
        """
        获取FinancialAnalyzer实例(延迟初始化)

        Returns:
            FinancialAnalyzer实例

        Raises:
            RuntimeError: 如果初始化失败
        """
        if self._analyzer is None:
            try:
                self._analyzer = FinancialAnalyzer()
                self.logger.debug("FinancialAnalyzer实例已创建")
            except Exception as e:
                raise RuntimeError(f"初始化FinancialAnalyzer失败: {e}")
        return self._analyzer

    def load_stock_configs(self) -> Dict[str, str]:
        """
        从环境变量加载股票配置

        读取 STOCKS_CONFIG 和 STOCKS_CONFIG_SELF 环境变量，
        合并返回股票名称到股票代码的映射。

        Returns:
            股票配置字典 {股票名称: 股票代码}
        """
        from dotenv import load_dotenv

        # 加载环境变量
        env_path = os.path.join(os.path.dirname(self.reports_dir), '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)

        stocks = {}

        # 加载自动选择的股票配置
        stocks_config = os.getenv('STOCKS_CONFIG', '{}')
        try:
            auto_stocks = json.loads(stocks_config)
            stocks.update(auto_stocks)
            if auto_stocks:
                self.logger.info(f"从STOCKS_CONFIG加载股票: {len(auto_stocks)} 只")
        except json.JSONDecodeError as e:
            self.logger.warning(f"解析STOCKS_CONFIG失败: {e}")

        # 加载自定义股票配置
        stocks_config_self = os.getenv('STOCKS_CONFIG_SELF', '{}')
        try:
            self_stocks = json.loads(stocks_config_self)
            stocks.update(self_stocks)
            if self_stocks:
                self.logger.info(f"从STOCKS_CONFIG_SELF加载股票: {len(self_stocks)} 只")
        except json.JSONDecodeError as e:
            self.logger.warning(f"解析STOCKS_CONFIG_SELF失败: {e}")

        self.logger.info(f"环境变量配置总股票数: {len(stocks)}")
        return stocks

    def _load_selected_stocks_from_pipeline(self) -> Dict[str, str]:
        """
        从管道选股结果加载股票

        读取 stock_selection_results.json 文件，获取选中的股票列表。

        Returns:
            股票配置字典 {股票名称: 股票代码}
        """
        results_file = os.path.join(
            self.config.data.data_dir,
            self.config.data.selection_results_file
        )

        if not os.path.exists(results_file):
            self.logger.debug(f"选股结果文件不存在: {results_file}")
            return {}

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                selection_data = json.load(f)

            selected_stocks = selection_data.get('selected_stocks', [])
            if not selected_stocks:
                self.logger.debug("选股结果为空")
                return {}

            # 转换为 {股票名称: 股票代码} 格式
            stocks = {}
            for item in selected_stocks:
                stock_code = item.get('stock_code', '')
                stock_name = item.get('stock_name', stock_code)
                if stock_code:
                    stocks[stock_name] = stock_code

            self.logger.info(f"从管道选股结果加载股票: {len(stocks)} 只")
            return stocks

        except Exception as e:
            self.logger.warning(f"读取选股结果文件失败: {e}")
            return {}

    def _get_stocks_to_analyze(
        self,
        stock_codes: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        获取需要分析的股票列表

        按优先级获取股票:
        1. 显式指定的 stock_codes 参数
        2. 管道选股结果
        3. 环境变量配置

        Args:
            stock_codes: 显式指定的股票代码列表

        Returns:
            股票配置字典 {股票名称: 股票代码}
        """
        # 优先级1: 显式指定的股票代码
        if stock_codes:
            self.logger.info(f"使用显式指定的股票代码: {len(stock_codes)} 只")
            return {code: code for code in stock_codes}

        # 优先级2: 管道选股结果
        pipeline_stocks = self._load_selected_stocks_from_pipeline()
        if pipeline_stocks:
            return pipeline_stocks

        # 优先级3: 环境变量配置
        env_stocks = self.load_stock_configs()
        if env_stocks:
            return env_stocks

        self.logger.warning("未找到任何股票配置")
        return {}

    async def generate_report_for_stock(
        self,
        stock_name: str,
        stock_code: str
    ) -> Dict[str, Any]:
        """
        为单个股票生成财务分析报告

        调用FinancialAnalyzer进行9步财务分析，生成JSON和Markdown报告。

        Args:
            stock_name: 股票名称
            stock_code: 股票代码

        Returns:
            分析结果字典，包含:
            - stock_name: 股票名称
            - stock_code: 股票代码
            - status: 分析状态 (success/failed)
            - analysis_date: 分析日期
            - json_report: JSON报告路径
            - markdown_report: Markdown报告路径
            - analysis_result: 完整分析结果(status=success时)
            - error: 错误信息(status=failed时)
        """
        self.logger.info(f"开始分析股票: {stock_name} ({stock_code})")

        try:
            # 使用timeout设置超时
            analysis_result = await asyncio.wait_for(
                self.analyzer.analyze_company(stock_name),
                timeout=self.timeout
            )

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

            await self.generate_markdown_report(
                analysis_result,
                markdown_path,
                stock_name,
                stock_code
            )

            self.logger.info(
                f"{stock_name} 分析完成 - "
                f"JSON: {report_path}, MD: {markdown_path}"
            )

            return {
                'stock_name': stock_name,
                'stock_code': stock_code,
                'status': 'success',
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'json_report': str(report_path),
                'markdown_report': str(markdown_path),
                'analysis_result': analysis_result
            }

        except asyncio.TimeoutError:
            error_msg = f"分析超时(超过{self.timeout}秒)"
            self.logger.error(f"{stock_name} {error_msg}")
            return {
                'stock_name': stock_name,
                'stock_code': stock_code,
                'status': 'failed',
                'error': error_msg
            }

        except Exception as e:
            self.logger.error(f"{stock_name} 分析失败: {e}")
            return {
                'stock_name': stock_name,
                'stock_code': stock_code,
                'status': 'failed',
                'error': str(e)
            }

    async def generate_markdown_report(
        self,
        analysis_result: Dict[str, Any],
        output_path: Path,
        stock_name: str,
        stock_code: str
    ) -> None:
        """
        生成Markdown格式的分析报告

        Args:
            analysis_result: 完整分析结果
            output_path: 输出文件路径
            stock_name: 股票名称
            stock_code: 股票代码
        """
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

        self.logger.debug(f"Markdown报告已生成: {output_path}")

    async def generate_all_reports(
        self,
        stock_codes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        为所有配置的股票生成分析报告

        Args:
            stock_codes: 显式指定的股票代码列表，优先级最高

        Returns:
            执行结果字典，包含:
            - success: 整体是否成功
            - success_count: 成功数量
            - failed_count: 失败数量
            - results: 详细结果列表
            - summary_report: 汇总报告路径
        """
        # 获取需要分析的股票
        stocks = self._get_stocks_to_analyze(stock_codes)

        if not stocks:
            self.logger.warning("未找到需要分析的股票")
            return {
                'success': False,
                'error': '未找到需要分析的股票',
                'success_count': 0,
                'failed_count': 0,
                'results': []
            }

        self.logger.info(f"开始为 {len(stocks)} 只股票生成财务分析报告")

        results = []

        # 处理每只股票
        for stock_name, stock_code in stocks.items():
            result = await self.generate_report_for_stock(stock_name, stock_code)
            results.append(result)

            # 延迟避免API请求过于频繁
            await asyncio.sleep(1)

        # 生成汇总报告
        summary_path = await self.generate_summary_report(results)

        success_count = len([r for r in results if r['status'] == 'success'])
        failed_count = len([r for r in results if r['status'] == 'failed'])

        self.logger.info(
            f"财务分析完成 - 成功: {success_count}, 失败: {failed_count}"
        )

        return {
            'success': failed_count == 0,
            'success_count': success_count,
            'failed_count': failed_count,
            'results': results,
            'summary_report': str(summary_path) if summary_path else None
        }

    async def generate_summary_report(self, results: List[Dict[str, Any]]) -> Optional[Path]:
        """
        生成汇总报告

        Args:
            results: 所有股票的分析结果列表

        Returns:
            汇总报告文件路径
        """
        if not results:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.reports_dir / f"汇总报告_{timestamp}.md"

        summary_content = f"""# 股票财务分析汇总报告

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

        self.logger.info(f"汇总报告已生成: {summary_path}")
        return summary_path

    def cleanup(self) -> None:
        """清理资源"""
        self._analyzer = None
        super().cleanup()


# 同步包装器，用于在同步管道中调用async方法
def _run_async(coro) -> Any:
    """
    在同步上下文中运行异步函数

    Args:
        coro: 协程对象

    Returns:
        协程的返回值
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
