"""
数据管道编排器模块

提供完整数据流程的编排和执行功能
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

from src.core.config import Config, get_config
from src.core.logger import get_logger
from src.data.downloaders import IncrementalDownloader
from src.data.feature_engineering import MLTrainingDataGenerator
from src.data.model_trainer import RandomForestTrainer
from src.data.rps_calculator import RPSPeriodsCalculator

logger = get_logger(__name__)


class PipelineStep(Enum):
    """数据管道步骤枚举"""
    DOWNLOAD = "download"       # 数据下载
    VALIDATE = "validate"       # 数据质量验证
    RPS = "rps"                 # RPS计算
    FEATURES = "features"       # 特征工程
    TRAIN = "train"             # 模型训练
    EVALUATE = "evaluate"       # 模型评估
    PREDICT = "predict"         # 批量预测
    REPORT = "report"           # 生成报告
    SELECT_STOCKS = "select_stocks"  # 选股
    ANALYZE_STOCKS = "analyze_stocks"  # 股票技术分析
    GENERATE_FINANCIAL_REPORTS = "generate_financial_reports"  # 财务分析报告
    ARCHIVE = "archive"         # 数据归档
    SEND_REPORTS = "send_reports"     # 发送报告
    MONITOR = "monitor"         # 性能监控


@dataclass
class PipelineResult:
    """
    管道执行结果

    Attributes:
        success: 是否成功
        step_results: 各步骤的结果
        errors: 错误信息列表
        start_time: 开始时间
        end_time: 结束时间
    """
    success: bool = False
    step_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[float]:
        """获取执行时长(秒)"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class DataPipeline:
    """
    数据管道编排器

    编排完整的数据流程: 下载 → RPS → 特征 → 训练

    Attributes:
        config: 配置实例
        steps: 要执行的步骤列表

    Example:
        >>> pipeline = DataPipeline()
        >>> # 执行完整流程
        >>> result = pipeline.run_full_pipeline()
        >>> # 从指定步骤开始
        >>> result = pipeline.run_from_step(PipelineStep.RPS)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        steps: Optional[List[PipelineStep]] = None,
    ):
        """
        初始化数据管道

        Args:
            config: 配置实例
            steps: 要执行的步骤列表，None则执行全部步骤
        """
        self.config = config or get_config()

        if steps is None:
            steps = [
                PipelineStep.DOWNLOAD,
                PipelineStep.VALIDATE,
                PipelineStep.RPS,
                PipelineStep.FEATURES,
                PipelineStep.TRAIN,
                PipelineStep.EVALUATE,
                PipelineStep.PREDICT,
                PipelineStep.REPORT,
            ]

        self.steps = steps

        # 初始化各组件
        self.downloader: Optional[IncrementalDownloader] = None
        self.data_validator: Optional[Any] = None
        self.rps_calculator: Optional[RPSPeriodsCalculator] = None
        self.feature_generator: Optional[MLTrainingDataGenerator] = None
        self.model_trainer: Optional[RandomForestTrainer] = None
        self.model_evaluator: Optional[Any] = None
        self.predictor: Optional[Any] = None
        self.reporter: Optional[Any] = None
        self.stock_selector: Optional[Any] = None
        self.stock_analyzer: Optional[Any] = None
        self.financial_report_generator: Optional[Any] = None
        self.email_sender: Optional[Any] = None
        self.archiver: Optional[Any] = None
        self.monitor: Optional[Any] = None

        # 初始化logger
        self._logger = logger

    @property
    def logger(self):
        """获取logger实例"""
        return self._logger

    def run_full_pipeline(self, **kwargs) -> PipelineResult:
        """
        执行完整的数据管道流程

        Args:
            **kwargs: 各步骤的参数

        Returns:
            管道执行结果
        """
        return self.run_from_step(None, **kwargs)

    def run_from_step(
        self, start_step: Optional[PipelineStep] = None, **kwargs
    ) -> PipelineResult:
        """
        从指定步骤开始执行管道

        Args:
            start_step: 起始步骤，None则从第一步开始
            **kwargs: 各步骤的参数

        Returns:
            管道执行结果
        """
        result = PipelineResult()
        result.start_time = datetime.now()

        self.logger.info("=" * 70)
        self.logger.info("数据管道开始执行")
        self.logger.info(f"执行步骤: {[s.value for s in self.steps]}")
        self.logger.info("=" * 70)

        # 确定起始步骤索引
        if start_step is None:
            start_idx = 0
        else:
            try:
                start_idx = [s for s in self.steps].index(start_step)
            except ValueError:
                result.errors.append(f"无效的起始步骤: {start_step}")
                result.end_time = datetime.now()
                return result

        # 执行各步骤
        for step in self.steps[start_idx:]:
            try:
                self.logger.info(f"\n{'=' * 70}")
                self.logger.info(f"执行步骤: {step.value}")
                self.logger.info(f"{'=' * 70}")

                step_result = self._execute_step(step, **kwargs)

                if step_result is None or isinstance(step_result, dict) and not step_result.get("success", True):
                    error_msg = f"步骤 {step.value} 执行失败"
                    self.logger.error(error_msg)
                    result.errors.append(error_msg)
                    result.end_time = datetime.now()
                    return result

                result.step_results[step.value] = step_result
                self.logger.info(f"步骤 {step.value} 完成")

            except Exception as e:
                error_msg = f"步骤 {step.value} 执行异常: {e}"
                self.logger.error(error_msg)
                result.errors.append(error_msg)
                result.end_time = datetime.now()
                return result

        result.success = True
        result.end_time = datetime.now()

        self.logger.info(f"\n{'=' * 70}")
        self.logger.info("数据管道执行完成")
        self.logger.info(f"总耗时: {result.duration:.2f} 秒")
        self.logger.info(f"成功率: {len(result.step_results)}/{len(self.steps)}")
        self.logger.info("=" * 70)

        return result

    def _execute_step(self, step: PipelineStep, **kwargs) -> Any:
        """执行单个步骤"""
        if step == PipelineStep.DOWNLOAD:
            return self._execute_download(**kwargs)
        elif step == PipelineStep.VALIDATE:
            return self._execute_validate(**kwargs)
        elif step == PipelineStep.RPS:
            return self._execute_rps(**kwargs)
        elif step == PipelineStep.FEATURES:
            return self._execute_features(**kwargs)
        elif step == PipelineStep.TRAIN:
            return self._execute_train(**kwargs)
        elif step == PipelineStep.EVALUATE:
            return self._execute_evaluate(**kwargs)
        elif step == PipelineStep.PREDICT:
            return self._execute_predict(**kwargs)
        elif step == PipelineStep.REPORT:
            return self._execute_report(**kwargs)
        elif step == PipelineStep.SELECT_STOCKS:
            return self._execute_select_stocks(**kwargs)
        elif step == PipelineStep.ANALYZE_STOCKS:
            return self._execute_analyze_stocks(**kwargs)
        elif step == PipelineStep.GENERATE_FINANCIAL_REPORTS:
            return self._execute_generate_financial_reports(**kwargs)
        elif step == PipelineStep.ARCHIVE:
            return self._execute_archive(**kwargs)
        elif step == PipelineStep.SEND_REPORTS:
            return self._execute_send_reports(**kwargs)
        elif step == PipelineStep.MONITOR:
            return self._execute_monitor(**kwargs)
        else:
            self.logger.warning(f"未知步骤: {step}")
            return None

    def _execute_download(self, **kwargs) -> Dict:
        """执行数据下载步骤"""
        if self.downloader is None:
            self.downloader = IncrementalDownloader(config=self.config)

        backfill_days = kwargs.get("backfill_days")
        forward_fill_date = kwargs.get("forward_fill_date")
        stock_codes = kwargs.get("stock_codes")
        initial_download_days = kwargs.get("initial_download_days", 300)
        max_stocks = kwargs.get("max_stocks")

        # 检查是否有现有数据
        existing_stocks = self.downloader.get_existing_stocks()
        has_existing_data = len(existing_stocks) > 0

        # 如果没有现有数据
        if not has_existing_data:
            if stock_codes:
                # 用户指定了股票代码，下载指定的股票
                self.logger.info(f"没有现有数据，执行初始下载，指定股票: {len(stock_codes)}只")
                result = self.downloader.initial_download(
                    stock_codes=stock_codes,
                    days=initial_download_days,
                    max_stocks=max_stocks,
                )
            else:
                # 没有指定股票代码，自动获取全量A股列表
                self.logger.info("没有现有数据，自动获取全量A股列表...")
                from src.data.providers import BaostockProvider

                provider = BaostockProvider(config=self.config)
                try:
                    all_stocks = provider.get_all_stocks()
                    self.logger.info(f"获取到 {len(all_stocks)} 只A股，开始下载...")
                    result = self.downloader.initial_download(
                        stock_codes=all_stocks,
                        days=initial_download_days,
                        max_stocks=max_stocks,
                    )
                finally:
                    provider.close()
        else:
            # 有现有数据
            if backfill_days or forward_fill_date:
                # 用户指定了增量更新参数，使用增量更新
                if stock_codes:
                    self.logger.info(f"增量更新指定股票: {len(stock_codes)}只 (往前补{backfill_days or 0}天)")
                else:
                    self.logger.info(f"增量更新现有股票: {len(existing_stocks)}只 (往前补{backfill_days or 0}天)")
                result = self.downloader.incremental_update(
                    backfill_days=backfill_days,
                    forward_fill_date=forward_fill_date,
                    stock_codes=stock_codes,
                )
            else:
                # 默认行为：下载最新数据（前一个交易日）
                if stock_codes:
                    self.logger.info(f"下载指定股票的最新数据: {len(stock_codes)}只")
                else:
                    self.logger.info(f"下载现有股票的最新数据: {len(existing_stocks)}只")
                result = self.downloader.download_latest(
                    stock_codes=stock_codes,
                    max_stocks=max_stocks,
                )

        return {"success": True, "data": result}

    def _execute_rps(self, **kwargs) -> Dict:
        """执行RPS计算步骤"""
        if self.rps_calculator is None:
            self.rps_calculator = RPSPeriodsCalculator(config=self.config)

        periods = kwargs.get("rps_periods", [5, 10, 20, 60, 120, 250])

        result = self.rps_calculator.calculate_multi_period_rps(periods=periods)

        if result is None:
            return {"success": False}

        return {"success": True, "rps_data": result}

    def _execute_features(self, **kwargs) -> Dict:
        """执行特征工程步骤"""
        if self.feature_generator is None:
            self.feature_generator = MLTrainingDataGenerator(config=self.config)

        max_stocks = kwargs.get("max_stocks")
        output_dir = kwargs.get("features_output_dir", "training_data")

        success = self.feature_generator.generate_training_data(output_dir=output_dir)

        return {"success": success, "output_dir": output_dir}

    def _execute_train(self, **kwargs) -> Dict:
        """执行模型训练步骤"""
        if self.model_trainer is None:
            self.model_trainer = RandomForestTrainer(config=self.config)

        target_label = kwargs.get("target_label", "return_5d_gt_5pct")
        training_data_dir = kwargs.get("training_data_dir", "training_data")

        results = self.model_trainer.train_models(target_label)

        return {"success": results is not None, "training_results": results}

    def _execute_validate(self, **kwargs) -> Dict:
        """执行数据质量验证步骤"""
        from src.data.validators import DataValidator

        if self.data_validator is None:
            self.data_validator = DataValidator(config=self.config)

        data_dir = kwargs.get("data_dir", self.config.data.data_dir)
        strict_mode = kwargs.get("strict_mode", False)

        validation_results = self.data_validator.validate_directory(
            data_dir=data_dir,
            strict_mode=strict_mode
        )

        return {
            "success": validation_results.passed,
            "validation_results": validation_results
        }

    def _execute_evaluate(self, **kwargs) -> Dict:
        """执行模型评估步骤"""
        from src.data.model_evaluator import ModelEvaluator

        if self.model_evaluator is None:
            self.model_evaluator = ModelEvaluator(config=self.config)

        target_label = kwargs.get("target_label", "return_5d_gt_5pct")
        training_data_dir = kwargs.get("training_data_dir", "training_data")

        evaluation_results = self.model_evaluator.evaluate(
            target_label=target_label,
            training_data_dir=training_data_dir
        )

        return {
            "success": evaluation_results is not None,
            "evaluation_results": evaluation_results
        }

    def _execute_predict(self, **kwargs) -> Dict:
        """执行批量预测步骤"""
        from src.data.batch_predictor import BatchPredictor

        if self.predictor is None:
            self.predictor = BatchPredictor(config=self.config)

        stock_codes = kwargs.get("stock_codes")
        output_file = kwargs.get("output_file", "predictions.csv")

        predictions = self.predictor.predict_batch(
            stock_codes=stock_codes,
            output_file=output_file
        )

        return {
            "success": predictions is not None,
            "predictions": predictions
        }

    def _execute_report(self, **kwargs) -> Dict:
        """执行报告生成步骤"""
        from src.data.reporter import PipelineReporter

        if self.reporter is None:
            self.reporter = PipelineReporter(config=self.config)

        report_type = kwargs.get("report_type", "summary")
        output_file = kwargs.get("output_file", "pipeline_report.md")

        report = self.reporter.generate_report(
            report_type=report_type,
            output_file=output_file
        )

        return {
            "success": report is not None,
            "report_file": output_file
        }

    def _execute_archive(self, **kwargs) -> Dict:
        """执行数据归档步骤"""
        from src.data.archiver import DataArchiver

        if self.archiver is None:
            self.archiver = DataArchiver(config=self.config)

        days_to_keep = kwargs.get("days_to_keep", 90)
        archive_type = kwargs.get("archive_type", "all")

        archive_results = self.archiver.archive_old_data(
            days_to_keep=days_to_keep,
            archive_type=archive_type
        )

        return {
            "success": True,
            "archived_files": archive_results.get("archived_count", 0)
        }

    def _execute_monitor(self, **kwargs) -> Dict:
        """执行性能监控步骤"""
        from src.data.monitor import PipelineMonitor

        if self.monitor is None:
            self.monitor = PipelineMonitor(config=self.config)

        metrics = self.monitor.collect_metrics()

        return {
            "success": True,
            "metrics": metrics
        }

    def _execute_select_stocks(self, **kwargs) -> Dict:
        """执行选股步骤"""
        from src.data.stock_selector import StockSelector

        if self.stock_selector is None:
            self.stock_selector = StockSelector(config=self.config)

        top_n = kwargs.get("select_top_n", 10)
        min_return = kwargs.get("min_return", 5.0)

        result = self.stock_selector.select_stocks(top_n=top_n, min_return=min_return)

        return {
            "success": True,
            "data": result
        }

    def _execute_analyze_stocks(self, **kwargs) -> Dict:
        """执行股票分析步骤"""
        from src.data.stock_analyzer import StockAnalyzer

        if self.stock_analyzer is None:
            self.stock_analyzer = StockAnalyzer(config=self.config)

        analysis_type = kwargs.get("analysis_type", "technical")

        result = self.stock_analyzer.analyze_stocks(analysis_type=analysis_type)

        return {
            "success": True,
            "data": result
        }

    def _execute_send_reports(self, **kwargs) -> Dict:
        """执行邮件发送步骤"""
        from src.data.email_sender import EmailSender

        if self.email_sender is None:
            self.email_sender = EmailSender(config=self.config)

        recipients = kwargs.get("email_recipients")

        result = self.email_sender.send_reports(recipients=recipients)

        return {
            "success": result.get("success", False),
            "data": result
        }

    def _execute_generate_financial_reports(self, **kwargs) -> Dict:
        """执行财务报告生成步骤"""
        from src.data.financial_reports import FinancialReportGenerator, _run_async

        if self.financial_report_generator is None:
            self.financial_report_generator = FinancialReportGenerator(config=self.config)

        stock_codes = kwargs.get("financial_report_stock_codes")

        # 使用同步包装器调用异步方法
        try:
            result = _run_async(
                self.financial_report_generator.generate_all_reports(
                    stock_codes=stock_codes
                )
            )

            return {
                "success": result.get("success", False),
                "data": result
            }
        except Exception as e:
            self.logger.error(f"财务报告生成失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def print_results(self, result: PipelineResult) -> None:
        """打印管道执行结果摘要"""
        print("\n" + "=" * 70)
        print("数据管道执行结果")
        print("=" * 70)
        print(f"状态: {'成功' if result.success else '失败'}")
        print(f"执行时长: {result.duration:.2f} 秒")
        print(f"已完成步骤: {len(result.step_results)}")

        for step_name, step_result in result.step_results.items():
            print(f"\n步骤: {step_name}")
            if isinstance(step_result, dict):
                for key, value in step_result.items():
                    if key != "success":
                        print(f"  {key}: {value}")

        if result.errors:
            print(f"\n错误信息:")
            for error in result.errors:
                print(f"  - {error}")

        print("=" * 70)


def run_daily_pipeline(
    backfill_days: int = 5,
    forward_fill_date: Optional[str] = None,
) -> PipelineResult:
    """
    运行每日数据管道(便捷函数)

    执行完整的数据更新流程:
    1. 增量下载最新数据
    2. 计算RPS指标
    3. 生成训练特征
    4. 训练模型

    Args:
        backfill_days: 往前补充的天数
        forward_fill_date: 往后补充到的日期

    Returns:
        管道执行结果

    Example:
        >>> result = run_daily_pipeline(backfill_days=5)
        >>> if result.success:
        >>>     print("数据管道执行成功!")
    """
    pipeline = DataPipeline()

    return pipeline.run_full_pipeline(
        backfill_days=backfill_days,
        forward_fill_date=forward_fill_date,
    )
