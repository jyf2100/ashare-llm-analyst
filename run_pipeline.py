#!/usr/bin/env python
"""
数据管道完整执行脚本

功能: 执行完整的数据流水线，从数据下载到报告生成
作者: AI Assistant
日期: 2024-12-24

用法:
    python run_pipeline.py [选项]

示例:
    # 执行完整流程
    python run_pipeline.py

    # 只执行部分步骤
    python run_pipeline.py --skip-train --skip-predict

    # 自定义参数
    python run_pipeline.py --backfill-days 10 --rps-periods 5,10,20,60,120
"""

import argparse
import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.pipeline import DataPipeline, PipelineStep
from src.core.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='数据管道完整执行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s                              # 执行完整流程
  %(prog)s --skip-train --skip-predict   # 跳过训练和预测步骤
  %(prog)s --backfill-days 10            # 往前补充10天
  %(prog)s --rps-periods 5,10,20,60,120 # 指定RPS周期
        '''
    )

    parser.add_argument(
        '--backfill-days',
        type=int,
        default=None,
        help='往前补充的天数 (默认: 不设置，只下载最新数据)'
    )

    parser.add_argument(
        '--forward-fill-date',
        type=str,
        default=None,
        help='往后补充到的日期 (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--target-label',
        type=str,
        default='return_5d_gt_5pct',
        help='目标标签 (默认: return_5d_gt_5pct)'
    )

    parser.add_argument(
        '--rps-periods',
        type=str,
        default='5,10,20,60',
        help='RPS计算周期，逗号分隔 (默认: 5,10,20,60)'
    )

    parser.add_argument(
        '--max-stocks',
        type=int,
        default=None,
        help='最大处理股票数 (默认: 无限制)'
    )

    parser.add_argument(
        '--stock-codes',
        type=str,
        default=None,
        help='股票代码列表，逗号分隔 (例如: sh.600000,sz.000001)。不指定则: 首次运行自动下载全量A股，后续更新现有股票'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='跳过数据下载步骤'
    )

    parser.add_argument(
        '--skip-validate',
        action='store_true',
        help='跳过数据验证步骤'
    )

    parser.add_argument(
        '--skip-rps',
        action='store_true',
        help='跳过RPS计算步骤'
    )

    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='跳过特征工程步骤'
    )

    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='跳过模型训练步骤'
    )

    parser.add_argument(
        '--skip-evaluate',
        action='store_true',
        help='跳过模型评估步骤'
    )

    parser.add_argument(
        '--skip-predict',
        action='store_true',
        help='跳过批量预测步骤'
    )

    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='跳过报告生成步骤'
    )

    parser.add_argument(
        '--generate-html-report',
        action='store_true',
        help='生成HTML格式报告'
    )

    parser.add_argument(
        '--archive-old-data',
        action='store_true',
        help='归档旧数据'
    )

    parser.add_argument(
        '--archive-days',
        type=int,
        default=90,
        help='归档N天前的数据 (默认: 90)'
    )

    parser.add_argument(
        '--select-top-n',
        type=int,
        default=10,
        help='ML选股选择前N只股票 (默认: 10)'
    )

    parser.add_argument(
        '--min-return',
        type=float,
        default=5.0,
        help='最低收益率阈值%% (默认: 5.0)'
    )

    parser.add_argument(
        '--analysis-type',
        type=str,
        default='technical',
        help='分析类型 (默认: technical)'
    )

    parser.add_argument(
        '--email-recipients',
        type=str,
        default=None,
        help='邮件收件人列表，逗号分隔'
    )

    parser.add_argument(
        '--skip-select-stocks',
        action='store_true',
        help='跳过ML选股步骤 (默认: 跳过)'
    )

    parser.add_argument(
        '--skip-analyze-stocks',
        action='store_true',
        help='跳过股票分析步骤 (默认: 跳过)'
    )

    parser.add_argument(
        '--skip-send-reports',
        action='store_true',
        help='跳过邮件发送步骤 (默认: 跳过)'
    )

    parser.add_argument(
        '--skip-generate-financial-reports',
        action='store_true',
        default=False,
        help='跳过财报分析报告生成步骤 (默认: 不跳过)'
    )

    parser.add_argument(
        '--financial-reports-dir',
        type=str,
        default='reports',
        help='财报报告输出目录 (默认: reports)'
    )

    parser.add_argument(
        '--concurrent-financial-reports',
        type=int,
        default=1,
        help='财报分析并发数量 (默认: 1)'
    )

    parser.add_argument(
        '--financial-analysis-timeout',
        type=int,
        default=300,
        help='财报分析超时时间(秒) (默认: 300)'
    )

    parser.add_argument(
        '--financial-report-stock-codes',
        type=str,
        default=None,
        help='财报分析股票代码列表，逗号分隔'
    )

    return parser.parse_args()


def print_banner():
    """打印横幅"""
    print("=" * 70)
    print("   数据管道完整执行脚本")
    print("=" * 70)
    print()


def print_execution_plan(args):
    """打印执行计划"""
    print("执行参数:")
    print(f"  往前补充天数: {args.backfill_days}")
    print(f"  往后补充日期: {args.forward_fill_date or '无'}")
    print(f"  目标标签: {args.target_label}")
    print(f"  RPS周期: {args.rps_periods}")
    print(f"  最大股票数: {args.max_stocks or '无限制'}")
    print(f"  股票代码: {args.stock_codes or '无'}")
    print()

    # 解析RPS周期
    rps_periods = [int(p.strip()) for p in args.rps_periods.split(',') if p.strip()]

    print("执行计划:")
    steps = _build_steps(args)
    for i, step in enumerate(steps, 1):
        step_names = {
            PipelineStep.DOWNLOAD: "数据下载",
            PipelineStep.VALIDATE: "数据验证",
            PipelineStep.RPS: "RPS计算",
            PipelineStep.FEATURES: "特征工程",
            PipelineStep.TRAIN: "模型训练",
            PipelineStep.EVALUATE: "模型评估",
            PipelineStep.PREDICT: "批量预测",
            PipelineStep.REPORT: "报告生成",
            PipelineStep.SELECT_STOCKS: "ML选股",
            PipelineStep.ANALYZE_STOCKS: "股票技术分析",
            PipelineStep.GENERATE_FINANCIAL_REPORTS: "财务分析报告",
            PipelineStep.SEND_REPORTS: "邮件发送",
            PipelineStep.ARCHIVE: "数据归档",
        }
        print(f"  [{i}] {step_names[step]:12s}: 执行")
    print()


def _build_steps(args):
    """根据参数构建执行步骤"""
    steps = []

    if not args.skip_download:
        steps.append(PipelineStep.DOWNLOAD)
    if not args.skip_validate:
        steps.append(PipelineStep.VALIDATE)
    if not args.skip_rps:
        steps.append(PipelineStep.RPS)
    if not args.skip_features:
        steps.append(PipelineStep.FEATURES)
    if not args.skip_train:
        steps.append(PipelineStep.TRAIN)
    if not args.skip_evaluate:
        steps.append(PipelineStep.EVALUATE)
    if not args.skip_predict:
        steps.append(PipelineStep.PREDICT)
    if not args.skip_report:
        steps.append(PipelineStep.REPORT)
    if not args.skip_select_stocks:
        steps.append(PipelineStep.SELECT_STOCKS)
    if not args.skip_analyze_stocks:
        steps.append(PipelineStep.ANALYZE_STOCKS)
    if not args.skip_generate_financial_reports:
        steps.append(PipelineStep.GENERATE_FINANCIAL_REPORTS)
    if not args.skip_send_reports:
        steps.append(PipelineStep.SEND_REPORTS)
    if args.archive_old_data:
        steps.append(PipelineStep.ARCHIVE)

    return steps


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 打印横幅和执行计划
    print_banner()
    print_execution_plan(args)

    print("开始时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()
    print("=" * 70)

    # 记录开始时间
    start_time = datetime.now()

    # 构建步骤
    steps = _build_steps(args)

    # 解析RPS周期
    rps_periods = [int(p.strip()) for p in args.rps_periods.split(',') if p.strip()]

    # 解析股票代码
    stock_codes = None
    if args.stock_codes:
        stock_codes = [s.strip() for s in args.stock_codes.split(',') if s.strip()]

    # 解析邮件收件人
    email_recipients = None
    if args.email_recipients:
        email_recipients = [e.strip() for e in args.email_recipients.split(',') if e.strip()]

    # 解析财报分析股票代码
    financial_report_stock_codes = None
    if args.financial_report_stock_codes:
        financial_report_stock_codes = [s.strip() for s in args.financial_report_stock_codes.split(',') if s.strip()]

    # 创建管道
    pipeline = DataPipeline(steps=steps)

    # 执行管道
    try:
        result = pipeline.run_full_pipeline(
            backfill_days=args.backfill_days,
            forward_fill_date=args.forward_fill_date,
            rps_periods=rps_periods,
            max_stocks=args.max_stocks,
            stock_codes=stock_codes,
            target_label=args.target_label,
            generate_html_report=args.generate_html_report,
            archive_days=args.archive_days,
            select_top_n=args.select_top_n,
            min_return=args.min_return,
            analysis_type=args.analysis_type,
            email_recipients=email_recipients,
            financial_reports_dir=args.financial_reports_dir,
            concurrent_reports=args.concurrent_financial_reports,
            financial_analysis_timeout=args.financial_analysis_timeout,
            financial_report_stock_codes=financial_report_stock_codes,
        )

        # 打印结果
        pipeline.print_results(result)

        # 计算执行时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print()
        print("=" * 70)
        if result.success:
            print("✅ 数据管道执行成功!")
        else:
            print("❌ 数据管道执行失败!")
            if result.errors:
                print("\n错误信息:")
                for error in result.errors:
                    print(f"  - {error}")

        print()
        print("结束时间:", end_time.strftime('%Y-%m-%d %H:%M:%S'))

        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"执行时长: {minutes}分{seconds}秒")
        print("=" * 70)

        sys.exit(0 if result.success else 1)

    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print("⚠️  用户中断执行")
        print("=" * 70)
        sys.exit(130)

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ 执行异常: {e}")
        print("=" * 70)

        import traceback
        traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
