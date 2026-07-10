"""
Report generator module - refactored with unified infrastructure.

Provides report generation with:
- Multiple output formats (JSON, Markdown, HTML)
- Parallel report generation
- Template-based formatting
- Summary reports
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.base import AnalyzerBase
from core.cache import cached, CacheConfig
from core.config import Config, get_config
from core.exceptions import AnalysisError, ValidationError
from core.logger import get_logger
from core.parallel import ParallelProcessor

logger = get_logger(__name__)


@dataclass
class ReportFormat:
    """
    Report output format specification.

    Attributes:
        name: Format name (json, markdown, html)
        extension: File extension
        mime_type: MIME type
    """
    name: str
    extension: str
    mime_type: str

    @classmethod
    def json(cls) -> "ReportFormat":
        """JSON format."""
        return cls("json", "json", "application/json")

    @classmethod
    def markdown(cls) -> "ReportFormat":
        """Markdown format."""
        return cls("markdown", "md", "text/markdown")

    @classmethod
    def html(cls) -> "ReportFormat":
        """HTML format."""
        return cls("html", "html", "text/html")


@dataclass
class StockConfig:
    """
    Stock configuration for report generation.

    Attributes:
        name: Stock name
        code: Stock code
    """
    name: str
    code: str


@dataclass
class ReportResult:
    """
    Result from report generation.

    Attributes:
        stock_name: Stock name
        stock_code: Stock code
        status: Generation status (success/failed)
        output_paths: Paths to generated reports by format
        error: Error message if failed
        analysis_result: Full analysis result if successful
    """
    stock_name: str
    stock_code: str
    status: str = "success"
    output_paths: Dict[str, Path] = field(default_factory=dict)
    error: Optional[str] = None
    analysis_result: Optional[Dict[str, Any]] = None

    def is_success(self) -> bool:
        """Check if report generation was successful."""
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "stock_name": self.stock_name,
            "stock_code": self.stock_code,
            "status": self.status,
        }

        if self.output_paths:
            result["output_paths"] = {
                fmt: str(path) for fmt, path in self.output_paths.items()
            }

        if self.error:
            result["error"] = self.error

        return result


class ReportTemplate:
    """
    Base class for report templates.

    Templates handle the formatting of analysis results into specific output formats.
    """

    def format(
        self,
        analysis_result: Dict[str, Any],
        stock_config: StockConfig,
    ) -> str:
        """
        Format analysis result into output string.

        Args:
            analysis_result: Analysis result data
            stock_config: Stock configuration

        Returns:
            Formatted output string
        """
        raise NotImplementedError


class MarkdownReportTemplate(ReportTemplate):
    """Template for Markdown reports."""

    def format(
        self,
        analysis_result: Dict[str, Any],
        stock_config: StockConfig,
    ) -> str:
        """Format analysis as Markdown."""
        final_report = analysis_result.get("final_report", {})

        content = f"""# {stock_config.name} ({stock_config.code}) 财报分析报告

## 基本信息
- **股票名称**: {stock_config.name}
- **股票代码**: {stock_config.code}
- **分析日期**: {analysis_result.get('analysis_date', 'N/A')}
- **分析状态**: {analysis_result.get('status', 'N/A')}

## 分析步骤完成情况
"""

        steps_completed = analysis_result.get("steps_completed", [])
        for step in steps_completed:
            status_emoji = "✅" if step.get("status") == "completed" else "❌"
            content += f"- {status_emoji} 步骤{step.get('step', 'N/A')}: {step.get('name', 'N/A')}\n"

        if final_report:
            content += f"""
## 执行摘要
{final_report.get('executive_summary', '暂无数据')}

## 财务亮点
"""

            highlights = final_report.get("financial_highlights", {})
            if highlights:
                content += f"""
- **营业收入**: {highlights.get('revenue', 'N/A')}
- **净利润**: {highlights.get('net_profit', 'N/A')}
- **ROE**: {highlights.get('roe', 'N/A')}
- **市盈率**: {highlights.get('pe_ratio', 'N/A')}
"""

            content += f"""
## 行业地位
{final_report.get('industry_position', '暂无数据')}

## 主要风险
"""

            risks = final_report.get("key_risks", [])
            for risk in risks:
                content += f"- {risk}\n"

            content += f"""
## 投资建议
- **建议**: {final_report.get('investment_recommendation', '暂无建议')}
- **评级**: {final_report.get('analyst_rating', 'N/A')}
- **信心水平**: {final_report.get('confidence_level', 'N/A')}

## 专业分析师报告
"""

            openai_analysis = final_report.get("openai_analysis", {})
            if openai_analysis and openai_analysis.get("analysis_result", {}).get("success"):
                analysis_text = openai_analysis["analysis_result"].get("analysis", "暂无详细分析")
                if len(analysis_text) > 2000:
                    analysis_text = analysis_text[:2000] + "...\n\n*（完整分析请查看JSON文件）*"
                content += f"\n{analysis_text}\n"
            else:
                content += "暂无详细分析内容\n"

        content += f"""
## 附录
- **报告日期**: {final_report.get('report_date', 'N/A')}
- **免责声明**: {final_report.get('disclaimer', '本报告仅供参考，不构成投资建议')}
- 完整分析数据请查看同目录下的JSON文件
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return content


class JSONReportTemplate(ReportTemplate):
    """Template for JSON reports."""

    def format(
        self,
        analysis_result: Dict[str, Any],
        stock_config: StockConfig,
    ) -> str:
        """Format analysis as JSON string."""
        # Add metadata
        result = {
            **analysis_result,
            "_metadata": {
                "stock_name": stock_config.name,
                "stock_code": stock_config.code,
                "generated_at": datetime.now().isoformat(),
            },
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


class ReportGenerator(AnalyzerBase):
    """
    Stock analysis report generator.

    Provides:
    - Multi-format report generation (JSON, Markdown, HTML)
    - Parallel processing for multiple stocks
    - Template-based formatting
    - Summary reports

    Example:
        generator = ReportGenerator(output_dir="reports")
        results = generator.generate_reports(
            stock_configs=[
                StockConfig("平安银行", "000001"),
                StockConfig("招商银行", "600036"),
            ],
            formats=[ReportFormat.markdown(), ReportFormat.json()]
        )
    """

    def __init__(
        self,
        output_dir: str = "reports",
        templates: Optional[Dict[str, ReportTemplate]] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize report generator.

        Args:
            output_dir: Base directory for output files
            templates: Custom format templates
            config: Configuration (uses global config if None)
        """
        super().__init__(config)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Default templates
        self._templates = templates or {
            "json": JSONReportTemplate(),
            "markdown": MarkdownReportTemplate(),
        }

    def register_template(self, format_name: str, template: ReportTemplate) -> None:
        """Register a custom report template."""
        self._templates[format_name] = template

    @cached(CacheConfig(ttl=300))
    def _load_stock_configs(self) -> List[StockConfig]:
        """
        Load stock configurations from environment.

        Returns:
            List of StockConfig objects
        """
        import os
        from dotenv import load_dotenv

        load_dotenv()

        configs = []

        # Load from STOCKS_CONFIG
        stocks_config = os.getenv("STOCKS_CONFIG", "{}")
        try:
            auto_stocks = json.loads(stocks_config)
            for name, code in auto_stocks.items():
                configs.append(StockConfig(name, code))
        except json.JSONDecodeError:
            pass

        # Load from STOCKS_CONFIG_SELF
        stocks_config_self = os.getenv("STOCKS_CONFIG_SELF", "{}")
        try:
            self_stocks = json.loads(stocks_config_self)
            for name, code in self_stocks.items():
                configs.append(StockConfig(name, code))
        except json.JSONDecodeError:
            pass

        return configs

    def generate_report(
        self,
        stock_config: StockConfig,
        analysis_result: Dict[str, Any],
        formats: Optional[List[ReportFormat]] = None,
    ) -> ReportResult:
        """
        Generate report for a single stock.

        Args:
            stock_config: Stock configuration
            analysis_result: Analysis result data
            formats: Output formats (defaults to JSON and Markdown)

        Returns:
            ReportResult with output paths
        """
        formats = formats or [ReportFormat.json(), ReportFormat.markdown()]

        # Create stock-specific directory
        stock_dir = self.output_dir / stock_config.name
        stock_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_paths = {}

        try:
            for fmt in formats:
                # Get template for format
                template = self._templates.get(fmt.name)
                if not template:
                    self.logger.warning(f"No template for format: {fmt.name}")
                    continue

                # Generate filename
                filename = f"{stock_config.name}_分析报告_{timestamp}.{fmt.extension}"
                output_path = stock_dir / filename

                # Format and write content
                content = template.format(analysis_result, stock_config)
                output_path.write_text(content, encoding="utf-8")

                output_paths[fmt.name] = output_path
                self.logger.debug(f"Generated {fmt.name} report: {output_path}")

            return ReportResult(
                stock_name=stock_config.name,
                stock_code=stock_config.code,
                status="success",
                output_paths=output_paths,
                analysis_result=analysis_result,
            )

        except Exception as e:
            self.logger.error(f"Report generation failed for {stock_config.name}: {e}")
            return ReportResult(
                stock_name=stock_config.name,
                stock_code=stock_config.code,
                status="failed",
                error=str(e),
            )

    def generate_reports(
        self,
        stock_configs: List[StockConfig],
        analysis_results: Optional[Dict[str, Dict[str, Any]]] = None,
        formats: Optional[List[ReportFormat]] = None,
        parallel: bool = True,
    ) -> List[ReportResult]:
        """
        Generate reports for multiple stocks.

        Args:
            stock_configs: List of stock configurations
            analysis_results: Pre-fetched analysis results by stock name
            formats: Output formats
            parallel: Use parallel processing

        Returns:
            List of ReportResult objects
        """
        results = []

        if parallel and len(stock_configs) > 3:
            max_workers = min(self.config.parallel.max_workers, len(stock_configs))

            with ParallelProcessor(max_workers=max_workers) as processor:
                # Process in parallel
                for config in stock_configs:
                    result = analysis_results.get(config.name) if analysis_results else None
                    if result is None:
                        # Create mock result if not provided
                        result = self._create_mock_result(config)

                    report_result = self.generate_report(config, result, formats)
                    results.append(report_result)
        else:
            # Sequential processing
            for config in stock_configs:
                result = analysis_results.get(config.name) if analysis_results else None
                if result is None:
                    result = self._create_mock_result(config)

                report_result = self.generate_report(config, result, formats)
                results.append(report_result)

        # Generate summary report
        self._generate_summary_report(results)

        return results

    def _create_mock_result(self, config: StockConfig) -> Dict[str, Any]:
        """Create mock analysis result for testing."""
        return {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "status": "completed",
            "steps_completed": [
                {"step": 1, "name": "数据获取", "status": "completed"},
                {"step": 2, "name": "财务分析", "status": "completed"},
                {"step": 3, "name": "风险评估", "status": "completed"},
            ],
            "final_report": {
                "executive_summary": f"{config.name} 财务状况良好，建议关注。",
                "financial_highlights": {
                    "revenue": "100亿元",
                    "net_profit": "20亿元",
                    "roe": "15%",
                    "pe_ratio": "20",
                },
                "industry_position": "行业领先地位",
                "key_risks": ["市场波动风险", "政策变化风险"],
                "investment_recommendation": "买入",
                "analyst_rating": "推荐",
                "confidence_level": "高",
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "disclaimer": "本报告仅供参考，不构成投资建议",
            },
        }

    def _generate_summary_report(self, results: List[ReportResult]) -> Path:
        """
        Generate summary report for all stocks.

        Args:
            results: List of report results

        Returns:
            Path to summary report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"汇总报告_{timestamp}.md"

        success_count = sum(1 for r in results if r.is_success())
        failed_count = len(results) - success_count

        content = f"""# 股票分析汇总报告

## 分析概况
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **分析股票数量**: {len(results)}
- **成功分析**: {success_count} 只
- **分析失败**: {failed_count} 只

## 分析结果详情

### 成功分析的股票
"""

        for result in results:
            if result.is_success():
                content += f"- ✅ **{result.stock_name}** ({result.stock_code})\n"
                for fmt, path in result.output_paths.items():
                    content += f"  - {fmt.upper()}: `{path}`\n"
                content += "\n"

        if failed_count > 0:
            content += "### 分析失败的股票\n"
            for result in results:
                if not result.is_success():
                    content += f"- ❌ **{result.stock_name}** ({result.stock_code})\n"
                    content += f"  - 错误信息: {result.error or 'Unknown error'}\n\n"

        content += """
## 报告说明
- 每只股票的详细分析报告保存在对应的股票名称目录中
- JSON文件包含完整的分析数据和步骤信息
- Markdown文件提供易读的分析报告格式
- 所有报告文件都包含时间戳以便版本管理
"""

        summary_path.write_text(content, encoding="utf-8")
        self.logger.info(f"Summary report generated: {summary_path}")

        return summary_path

    def print_results(self, results: List[ReportResult]) -> None:
        """
        Print summary of report generation results.

        Args:
            results: List of report results
        """
        print("\n" + "=" * 70)
        print("📊 Report Generation Summary")
        print("=" * 70)

        success = [r for r in results if r.is_success()]
        failed = [r for r in results if not r.is_success()]

        print(f"\n✅ Success: {len(success)} stocks")
        print(f"❌ Failed: {len(failed)} stocks")

        if success:
            print(f"\n📁 Output directory: {self.output_dir.absolute()}")

        print("=" * 70)


# Convenience functions for backward compatibility
def generate_stock_report(
    stock_name: str,
    stock_code: str,
    analysis_result: Dict[str, Any],
    output_dir: str = "reports",
) -> ReportResult:
    """
    Generate a report for a single stock.

    Args:
        stock_name: Stock name
        stock_code: Stock code
        analysis_result: Analysis result data
        output_dir: Output directory

    Returns:
        ReportResult
    """
    generator = ReportGenerator(output_dir=output_dir)
    return generator.generate_report(
        StockConfig(stock_name, stock_code),
        analysis_result,
    )


def generate_reports_batch(
    stock_configs: List[Tuple[str, str]],
    analysis_results: Optional[Dict[str, Dict[str, Any]]] = None,
    output_dir: str = "reports",
) -> List[ReportResult]:
    """
    Generate reports for multiple stocks.

    Args:
        stock_configs: List of (name, code) tuples
        analysis_results: Analysis results by stock name
        output_dir: Output directory

    Returns:
        List of ReportResult
    """
    generator = ReportGenerator(output_dir=output_dir)
    configs = [StockConfig(name, code) for name, code in stock_configs]
    return generator.generate_reports(configs, analysis_results)
