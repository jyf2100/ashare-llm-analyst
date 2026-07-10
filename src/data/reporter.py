"""
报告生成模块

生成数据管道执行报告
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)


class PipelineReporter(AnalyzerBase):
    """
    管道报告生成器

    生成数据管道执行结果的各种报告

    Attributes:
        config: 配置实例
        output_dir: 报告输出目录

    Example:
        >>> reporter = PipelineReporter()
        >>> report_file = reporter.generate_report("summary", "report.md")
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """初始化报告生成器"""
        super().__init__(config)

        if output_dir is None:
            output_dir = "reports"

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(
        self,
        report_type: str = "summary",
        output_file: Optional[str] = None,
        **context
    ) -> Optional[str]:
        """
        生成报告

        Args:
            report_type: 报告类型 (summary, detailed, evaluation)
            output_file: 输出文件路径
            **context: 报告上下文数据

        Returns:
            报告文件路径
        """
        if report_type == "summary":
            content = self._generate_summary_report(context)
        elif report_type == "detailed":
            content = self._generate_detailed_report(context)
        elif report_type == "evaluation":
            content = self._generate_evaluation_report(context)
        else:
            self.logger.warning(f"未知报告类型: {report_type}")
            return None

        # 确定输出文件
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"{report_type}_{timestamp}.md")

        # 保存报告
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        self.logger.info(f"报告已生成: {output_file}")
        return output_file

    def _generate_summary_report(self, context: Dict) -> str:
        """生成汇总报告"""
        lines = [
            "# 数据管道执行报告\n",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## 执行摘要\n",
        ]

        # 添加执行结果
        if "result" in context:
            result = context["result"]
            lines.extend([
                f"- **状态**: {'成功' if result.success else '失败'}\n",
                f"- **执行时长**: {result.duration:.2f} 秒\n" if result.duration else "- **执行时长**: N/A\n",
                f"- **完成步骤**: {len(result.step_results)}/{len(result.steps) if hasattr(result, 'steps') else len(result.step_results)}\n",
            ])

        lines.append("\n## 步骤详情\n")

        # 添加各步骤结果
        if "result" in context:
            for step_name, step_result in result.step_results.items():
                lines.append(f"### {step_name}\n")
                if isinstance(step_result, dict):
                    for key, value in step_result.items():
                        if key != "success":
                            lines.append(f"- **{key}**: {value}\n")
                lines.append("\n")

        return "".join(lines)

    def _generate_detailed_report(self, context: Dict) -> str:
        """生成详细报告"""
        lines = [
            "# 数据管道详细报告\n",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n## 1. 数据概览\n",
        ]

        # 添加数据统计
        if "data_stats" in context:
            stats = context["data_stats"]
            lines.extend([
                f"- 总文件数: {stats.get('total_files', 0)}\n",
                f"- 总大小: {stats.get('total_size_mb', 0):.2f} MB\n",
                f"- 日期范围: {stats.get('date_range', {})}\n",
            ])

        lines.append("\n## 2. 模型性能\n")

        # 添加模型评估结果
        if "evaluation" in context:
            eval_result = context["evaluation"]
            lines.extend([
                f"- 准确率: {eval_result.get('accuracy', 'N/A')}\n",
                f"- F1分数: {eval_result.get('f1_score', 'N/A')}\n",
                f"- AUC: {eval_result.get('auc', 'N/A')}\n",
            ])

        lines.append("\n## 3. 预测结果\n")

        # 添加预测汇总
        if "prediction_summary" in context:
            summary = context["prediction_summary"]
            lines.extend([
                f"- 总预测数: {summary.get('total_predictions', 0)}\n",
                f"- 正预测数: {summary.get('positive_predictions', 0)}\n",
                f"- 正例比例: {summary.get('positive_ratio', 0):.1%}\n",
            ])

        return "".join(lines)

    def _generate_evaluation_report(self, context: Dict) -> str:
        """生成评估报告"""
        lines = [
            "# 模型评估报告\n",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n## 模型性能指标\n",
        ]

        if "evaluation" in context:
            eval_result = context["evaluation"]
            lines.extend([
                "### 分类指标\n" if eval_result.get("accuracy") else "### 回归指标\n",
            ])

            if eval_result.get("accuracy") is not None:
                lines.extend([
                    f"| 指标 | 值 |\n",
                    f"|------|------|\n",
                    f"| 准确率 | {eval_result['accuracy']:.4f} |\n",
                    f"| F1分数 | {eval_result['f1_score']:.4f} |\n",
                    f"| AUC | {eval_result.get('auc', 'N/A')} |\n",
                ])

            if eval_result.get("rmse") is not None:
                lines.extend([
                    f"| 指标 | 值 |\n",
                    f"|------|------|\n",
                    f"| RMSE | {eval_result['rmse']:.4f} |\n",
                    f"| MAE | {eval_result['mae']:.4f} |\n",
                    f"| R² | {eval_result['r2']:.4f} |\n",
                ])

        return "".join(lines)

    def generate_html_report(
        self,
        report_type: str = "summary",
        output_file: Optional[str] = None,
        **context
    ) -> Optional[str]:
        """
        生成HTML格式报告

        Args:
            report_type: 报告类型
            output_file: 输出文件路径
            **context: 报告上下文

        Returns:
            报告文件路径
        """
        html_content = self._generate_html_report(report_type, context)

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"{report_type}_{timestamp}.html")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info(f"HTML报告已生成: {output_file}")
        return output_file

    def _generate_html_report(self, report_type: str, context: Dict) -> str:
        """生成HTML报告"""
        lines = [
            "<!DOCTYPE html>\n",
            "<html>\n",
            "<head>\n",
            "  <meta charset='UTF-8'>\n",
            f"  <title>数据管道报告 - {report_type}</title>\n",
            "  <style>\n",
            "    body { font-family: Arial, sans-serif; margin: 20px; }\n",
            "    h1 { color: #333; }\n",
            "    h2 { color: #666; margin-top: 30px; }\n",
            "    table { border-collapse: collapse; width: 100%; }\n",
            "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n",
            "    th { background-color: #4CAF50; color: white; }\n",
            "    .success { color: green; }\n",
            "    .error { color: red; }\n",
            "  </style>\n",
            "</head>\n",
            "<body>\n",
            f"  <h1>数据管道报告 - {report_type}</h1>\n",
            f"  <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n",
        ]

        # 添加执行状态
        if "result" in context:
            result = context["result"]
            status_class = "success" if result.success else "error"
            lines.extend([
                f"  <h2>执行状态</h2>\n",
                f"  <p class='{status_class}'>{'成功' if result.success else '失败'}</p>\n",
                f"  <p>执行时长: {result.duration:.2f} 秒</p>\n" if result.duration else "",
            ])

        lines.extend([
            "</body>\n",
            "</html>\n",
        ])

        return "".join(lines)
