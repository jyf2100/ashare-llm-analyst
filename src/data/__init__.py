"""
数据管道模块

提供股票数据下载、RPS计算、特征工程和模型训练的完整数据流程。

模块结构:
    providers              - 数据提供者接口和实现
    downloaders            - 数据下载器和反爬虫控制
    validators             - 数据质量验证
    rps_calculator         - RPS(相对价格强度)计算器
    feature_engineering    - 技术指标和训练数据生成
    model_trainer          - 模型训练器
    model_evaluator        - 模型评估器
    batch_predictor        - 批量预测器
    reporter               - 报告生成器
    archiver               - 数据归档器
    monitor                - 性能监控器
    stock_selector         - ML选股器
    stock_analyzer         - 股票技术分析器
    financial_reports      - 财务分析报告生成器
    email_sender           - 邮件发送器
    pipeline               - 数据管道编排器

使用示例:
    >>> from src.data import DataPipeline, PipelineStep
    >>>
    >>> # 创建管道
    >>> pipeline = DataPipeline()
    >>>
    >>> # 执行完整流程
    >>> result = pipeline.run_full_pipeline()
"""

from src.data.providers import (
    DataProvider,
    BaostockProvider,
    CSVProvider,
    CachedProvider,
)

from src.data.downloaders import (
    AntiCrawlerController,
    RequestRetryManager,
    IncrementalDownloader,
)

from src.data.validators import (
    DataValidator,
    ValidationResult,
)

from src.data.rps_calculator import (
    RPSPeriodsCalculator,
)

from src.data.feature_engineering import (
    FeatureCalculator,
    MLTrainingDataGenerator,
    MLTrainingConfig,
)

from src.data.model_trainer import (
    ModelTrainerBase,
    RandomForestTrainer,
)

from src.data.model_evaluator import (
    ModelEvaluator,
    EvaluationResult,
)

from src.data.batch_predictor import (
    BatchPredictor,
)

from src.data.reporter import (
    PipelineReporter,
)

from src.data.archiver import (
    DataArchiver,
)

from src.data.monitor import (
    PipelineMonitor,
    PerformanceMetrics,
)

from src.data.stock_selector import (
    StockSelector,
)

from src.data.stock_analyzer import (
    StockAnalyzer,
)

from src.data.email_sender import (
    EmailSender,
)

from src.data.financial_reports import (
    FinancialReportGenerator,
)

from src.data.pipeline import (
    DataPipeline,
    PipelineStep,
    PipelineResult,
    run_daily_pipeline,
)

__all__ = [
    # Providers
    "DataProvider",
    "BaostockProvider",
    "CSVProvider",
    "CachedProvider",
    # Downloaders
    "AntiCrawlerController",
    "RequestRetryManager",
    "IncrementalDownloader",
    # Validators
    "DataValidator",
    "ValidationResult",
    # RPS Calculator
    "RPSPeriodsCalculator",
    # Feature Engineering
    "FeatureCalculator",
    "MLTrainingDataGenerator",
    "MLTrainingConfig",
    # Model Trainer
    "ModelTrainerBase",
    "RandomForestTrainer",
    # Model Evaluator
    "ModelEvaluator",
    "EvaluationResult",
    # Batch Predictor
    "BatchPredictor",
    # Reporter
    "PipelineReporter",
    # Archiver
    "DataArchiver",
    # Monitor
    "PipelineMonitor",
    "PerformanceMetrics",
    # Stock Selector
    "StockSelector",
    # Stock Analyzer
    "StockAnalyzer",
    # Financial Reports
    "FinancialReportGenerator",
    # Email Sender
    "EmailSender",
    # Pipeline
    "DataPipeline",
    "PipelineStep",
    "PipelineResult",
    "run_daily_pipeline",
]
