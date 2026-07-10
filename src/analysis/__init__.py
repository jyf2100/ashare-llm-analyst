"""
Analysis module for stock analysis.

Contains refactored analyzers with unified infrastructure.
"""

from src.analysis.technical_analyzer import (
    TechnicalAnalyzer,
    Signal,
    generate_trading_signals,
)
from src.analysis.stock_selector import (
    StockSelector,
    StockPrediction,
    SelectionCriteria,
    SelectionResult,
    filter_stocks_by_rps,
    predict_stock_returns,
)
from src.analysis.llm_analyzer import (
    LLMAnalyzer,
    AnalysisRequest,
    AnalysisResult,
    PromptManager,
    PromptTemplate,
    analyze_stock,
    create_analysis_request,
)
from src.analysis.report_generator import (
    ReportGenerator,
    ReportFormat,
    ReportResult,
    StockConfig,
    ReportTemplate,
    MarkdownReportTemplate,
    JSONReportTemplate,
    generate_stock_report,
    generate_reports_batch,
)

__all__ = [
    "TechnicalAnalyzer",
    "Signal",
    "generate_trading_signals",
    "StockSelector",
    "StockPrediction",
    "SelectionCriteria",
    "SelectionResult",
    "filter_stocks_by_rps",
    "predict_stock_returns",
    "LLMAnalyzer",
    "AnalysisRequest",
    "AnalysisResult",
    "PromptManager",
    "PromptTemplate",
    "analyze_stock",
    "create_analysis_request",
    "ReportGenerator",
    "ReportFormat",
    "ReportResult",
    "StockConfig",
    "ReportTemplate",
    "MarkdownReportTemplate",
    "JSONReportTemplate",
    "generate_stock_report",
    "generate_reports_batch",
]
