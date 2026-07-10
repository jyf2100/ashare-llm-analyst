#!/bin/bash

#===============================================================================
# 数据管道完整执行脚本
#
# 功能: 执行完整的数据流水线，从数据下载到报告生成
# 作者: AI Assistant
# 日期: 2024-12-24
#===============================================================================

# 设置环境变量
export PATH="/home/ubuntu/miniconda3/bin:/home/ubuntu/miniconda3/condabin:$PATH"
export CONDA_EXE="/home/ubuntu/miniconda3/bin/conda"
export CONDA_PREFIX="/home/ubuntu/miniconda3"
export CONDA_PYTHON_EXE="/home/ubuntu/miniconda3/bin/python"
export CONDA_DEFAULT_ENV="base"

# 切换到工作目录
cd /mnt/disk01/workspaces/worksummary/ashare-llm-analyst

# 解析命令行参数
BACKFILL_DAYS=${BACKFILL_DAYS:-""}
FORWARD_FILL_DATE=${FORWARD_FILL_DATE:-""}
TARGET_LABEL=${TARGET_LABEL:-"return_5d_gt_5pct"}
RPS_PERIODS=${RPS_PERIODS:-"5,10,20,60"}
MAX_STOCKS=${MAX_STOCKS:-""}
STOCK_CODES=${STOCK_CODES:-""}
SELECT_TOP_N=${SELECT_TOP_N:-10}
MIN_RETURN=${MIN_RETURN:-5.0}
ANALYSIS_TYPE=${ANALYSIS_TYPE:-"technical"}
EMAIL_RECIPIENTS=${EMAIL_RECIPIENTS:-""}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
SKIP_VALIDATE=${SKIP_VALIDATE:-0}
SKIP_RPS=${SKIP_RPS:-0}
SKIP_FEATURES=${SKIP_FEATURES:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVALUATE=${SKIP_EVALUATE:-0}
SKIP_PREDICT=${SKIP_PREDICT:-0}
SKIP_REPORT=${SKIP_REPORT:-0}
SKIP_SELECT_STOCKS=${SKIP_SELECT_STOCKS:-1}
SKIP_ANALYZE_STOCKS=${SKIP_ANALYZE_STOCKS:-1}
SKIP_SEND_REPORTS=${SKIP_SEND_REPORTS:-1}
SKIP_GENERATE_FINANCIAL_REPORTS=${SKIP_GENERATE_FINANCIAL_REPORTS:-0}
FINANCIAL_REPORTS_DIR=${FINANCIAL_REPORTS_DIR:-"reports"}
CONCURRENT_FINANCIAL_REPORTS=${CONCURRENT_FINANCIAL_REPORTS:-1}
FINANCIAL_ANALYSIS_TIMEOUT=${FINANCIAL_ANALYSIS_TIMEOUT:-300}
FINANCIAL_REPORT_STOCK_CODES=${FINANCIAL_REPORT_STOCK_CODES:-""}
GENERATE_HTML_REPORT=${GENERATE_HTML_REPORT:-0}
ARCHIVE_OLD_DATA=${ARCHIVE_OLD_DATA:-0}
ARCHIVE_DAYS=${ARCHIVE_DAYS:-90}

# 显示帮助信息
function show_help() {
    cat << EOF
用法: $0 [选项]

数据管道完整执行脚本 - 使用重构后的模块执行完整的数据流程

选项:
    --backfill-days N          往前补充的天数 (默认: 不设置，只下载最新数据)
    --forward-fill-date DATE   往后补充到的日期 (YYYY-MM-DD)
    --target-label LABEL       目标标签 (默认: return_5d_gt_5pct)
    --rps-periods "5,10,20"    RPS计算周期 (默认: 5,10,20,60)
    --max-stocks N             最大处理股票数
    --stock-codes "CODES"      股票代码列表 (例如: "sh.600000,sz.000001")
    --skip-download            跳过数据下载步骤
    --skip-validate            跳过数据验证步骤
    --skip-rps                 跳过RPS计算步骤
    --skip-features            跳过特征工程步骤
    --skip-train               跳过模型训练步骤
    --skip-evaluate            跳过模型评估步骤
    --skip-predict             跳过批量预测步骤
    --skip-report              跳过报告生成步骤
    --generate-html-report     生成HTML格式报告
    --archive-old-data         归档旧数据
    --archive-days N           归档N天前的数据 (默认: 90)
    --select-top-n N           ML选股选择前N只 (默认: 10)
    --min-return N             最低收益率阈值% (默认: 5.0)
    --analysis-type TYPE       分析类型 (默认: technical)
    --email-recipients EMAILS  邮件收件人列表 (逗号分隔)
    --skip-select-stocks       跳过ML选股步骤
    --skip-analyze-stocks      跳过股票分析步骤
    --skip-generate-financial-reports  跳过财报分析报告步骤 (默认: 不跳过)
    --financial-reports-dir    财报报告输出目录 (默认: reports)
    --concurrent-financial-reports N  财报分析并发数量 (默认: 1)
    --financial-analysis-timeout N  财报分析超时时间(秒) (默认: 300)
    --financial-report-stock-codes CODES  财报分析股票代码列表
    --skip-send-reports        跳过邮件发送步骤
    -h, --help                 显示此帮助信息

示例:
    # 执行完整流程
    $0

    # 只下载和验证数据
    $0 --skip-rps --skip-features --skip-train --skip-evaluate --skip-predict

    # 自定义参数执行
    $0 --backfill-days 10 --rps-periods "5,10,20,60,120" --max-stocks 50

    # 执行完整流程包括选股、分析和邮件报告
    $0 --skip-select-stocks 0 --skip-analyze-stocks 0 --skip-send-reports 0

EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --backfill-days)
            BACKFILL_DAYS="$2"
            shift 2
            ;;
        --forward-fill-date)
            FORWARD_FILL_DATE="$2"
            shift 2
            ;;
        --target-label)
            TARGET_LABEL="$2"
            shift 2
            ;;
        --rps-periods)
            RPS_PERIODS="$2"
            shift 2
            ;;
        --max-stocks)
            MAX_STOCKS="$2"
            shift 2
            ;;
        --stock-codes)
            STOCK_CODES="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=1
            shift
            ;;
        --skip-validate)
            SKIP_VALIDATE=1
            shift
            ;;
        --skip-rps)
            SKIP_RPS=1
            shift
            ;;
        --skip-features)
            SKIP_FEATURES=1
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=1
            shift
            ;;
        --skip-evaluate)
            SKIP_EVALUATE=1
            shift
            ;;
        --skip-predict)
            SKIP_PREDICT=1
            shift
            ;;
        --skip-report)
            SKIP_REPORT=1
            shift
            ;;
        --generate-html-report)
            GENERATE_HTML_REPORT=1
            shift
            ;;
        --archive-old-data)
            ARCHIVE_OLD_DATA=1
            shift
            ;;
        --archive-days)
            ARCHIVE_DAYS="$2"
            shift 2
            ;;
        --select-top-n)
            SELECT_TOP_N="$2"
            shift 2
            ;;
        --min-return)
            MIN_RETURN="$2"
            shift 2
            ;;
        --analysis-type)
            ANALYSIS_TYPE="$2"
            shift 2
            ;;
        --email-recipients)
            EMAIL_RECIPIENTS="$2"
            shift 2
            ;;
        --skip-select-stocks)
            SKIP_SELECT_STOCKS=1
            shift
            ;;
        --skip-analyze-stocks)
            SKIP_ANALYZE_STOCKS=1
            shift
            ;;
        --skip-send-reports)
            SKIP_SEND_REPORTS=1
            shift
            ;;
        --skip-generate-financial-reports)
            SKIP_GENERATE_FINANCIAL_REPORTS=1
            shift
            ;;
        --financial-reports-dir)
            FINANCIAL_REPORTS_DIR="$2"
            shift 2
            ;;
        --concurrent-financial-reports)
            CONCURRENT_FINANCIAL_REPORTS="$2"
            shift 2
            ;;
        --financial-analysis-timeout)
            FINANCIAL_ANALYSIS_TIMEOUT="$2"
            shift 2
            ;;
        --financial-report-stock-codes)
            FINANCIAL_REPORT_STOCK_CODES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 将逗号分隔的周期转换为数组
IFS=',' read -ra RPS_PERIODS_ARRAY <<< "$RPS_PERIODS"

#===============================================================================
# 开始执行
#===============================================================================

START_TIME=$(date +%s)

echo "=========================================="
echo "   数据管道完整执行脚本"
echo "=========================================="
echo ""
echo "执行参数:"
echo "  往前补充天数: $BACKFILL_DAYS"
echo "  往后补充日期: ${FORWARD_FILL_DATE:-无}"
echo "  目标标签: $TARGET_LABEL"
echo "  RPS周期: ${RPS_PERIODS_ARRAY[@]}"
echo "  最大股票数: ${MAX_STOCKS:-无限制}"
echo "  股票代码: ${STOCK_CODES:-无}"
echo ""
echo "执行计划:"
echo "  [1] 数据下载     : $([ $SKIP_DOWNLOAD -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [2] 数据验证     : $([ $SKIP_VALIDATE -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [3] RPS计算      : $([ $SKIP_RPS -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [4] 特征工程     : $([ $SKIP_FEATURES -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [5] 模型训练     : $([ $SKIP_TRAIN -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [6] 模型评估     : $([ $SKIP_EVALUATE -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [7] 批量预测     : $([ $SKIP_PREDICT -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [8] 报告生成     : $([ $SKIP_REPORT -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [9] ML选股       : $([ $SKIP_SELECT_STOCKS -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [10] 股票分析    : $([ $SKIP_ANALYZE_STOCKS -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [11] 财务分析报告: $([ $SKIP_GENERATE_FINANCIAL_REPORTS -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [12] 邮件发送    : $([ $SKIP_SEND_REPORTS -eq 0 ] && echo '执行' || echo '跳过')"
echo "  [13] 数据归档     : $([ $ARCHIVE_OLD_DATA -eq 1 ] && echo '执行' || echo '跳过')"
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "=========================================="

# 激活conda环境
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate ashare-llm-analyst

# 创建临时Python脚本来运行管道
cat > /tmp/run_pipeline_$$.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python
"""数据管道执行脚本"""
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '/mnt/disk01/workspaces/worksummary/ashare-llm-analyst')

from src.data.pipeline import DataPipeline, PipelineStep
from src.core.logger import get_logger

logger = get_logger(__name__)

def main():
    # 获取环境变量
    backfill_days = int(os.environ.get('BACKFILL_DAYS', 5))
    forward_fill_date = os.environ.get('FORWARD_FILL_DATE') or None
    target_label = os.environ.get('TARGET_LABEL', 'return_5d_gt_5pct')
    rps_periods_str = os.environ.get('RPS_PERIODS', '5,10,20,60')
    max_stocks = os.environ.get('MAX_STOCKS') or None
    stock_codes_str = os.environ.get('STOCK_CODES') or None

    skip_download = bool(int(os.environ.get('SKIP_DOWNLOAD', 0)))
    skip_validate = bool(int(os.environ.get('SKIP_VALIDATE', 0)))
    skip_rps = bool(int(os.environ.get('SKIP_RPS', 0)))
    skip_features = bool(int(os.environ.get('SKIP_FEATURES', 0)))
    skip_train = bool(int(os.environ.get('SKIP_TRAIN', 0)))
    skip_evaluate = bool(int(os.environ.get('SKIP_EVALUATE', 0)))
    skip_predict = bool(int(os.environ.get('SKIP_PREDICT', 0)))
    skip_report = bool(int(os.environ.get('SKIP_REPORT', 0)))
    skip_select_stocks = bool(int(os.environ.get('SKIP_SELECT_STOCKS', 1)))
    skip_analyze_stocks = bool(int(os.environ.get('SKIP_ANALYZE_STOCKS', 1)))
    skip_send_reports = bool(int(os.environ.get('SKIP_SEND_REPORTS', 1)))
    skip_generate_financial_reports = bool(int(os.environ.get('SKIP_GENERATE_FINANCIAL_REPORTS', 0)))
    generate_html_report = bool(int(os.environ.get('GENERATE_HTML_REPORT', 0)))
    archive_old_data = bool(int(os.environ.get('ARCHIVE_OLD_DATA', 0)))
    archive_days = int(os.environ.get('ARCHIVE_DAYS', 90))

    # 新增参数
    select_top_n = int(os.environ.get('SELECT_TOP_N', 10))
    min_return = float(os.environ.get('MIN_RETURN', 5.0))
    analysis_type = os.environ.get('ANALYSIS_TYPE', 'technical')
    email_recipients_str = os.environ.get('EMAIL_RECIPIENTS') or None

    # 财务报告参数
    financial_reports_dir = os.environ.get('FINANCIAL_REPORTS_DIR', 'reports')
    concurrent_reports = int(os.environ.get('CONCURRENT_FINANCIAL_REPORTS', 1))
    financial_analysis_timeout = int(os.environ.get('FINANCIAL_ANALYSIS_TIMEOUT', 300))
    financial_report_stock_codes_str = os.environ.get('FINANCIAL_REPORT_STOCK_CODES') or None

    # 解析RPS周期
    rps_periods = [int(p.strip()) for p in rps_periods_str.split(',') if p.strip()]

    # 解析股票代码
    stock_codes = None
    if stock_codes_str:
        stock_codes = [s.strip() for s in stock_codes_str.split(',') if s.strip()]

    # 解析邮件收件人
    email_recipients = None
    if email_recipients_str:
        email_recipients = [e.strip() for e in email_recipients_str.split(',') if e.strip()]

    # 解析财报分析股票代码
    financial_report_stock_codes = None
    if financial_report_stock_codes_str:
        financial_report_stock_codes = [s.strip() for s in financial_report_stock_codes_str.split(',') if s.strip()]

    # 构建执行步骤
    steps = []
    if not skip_download:
        steps.append(PipelineStep.DOWNLOAD)
    if not skip_validate:
        steps.append(PipelineStep.VALIDATE)
    if not skip_rps:
        steps.append(PipelineStep.RPS)
    if not skip_features:
        steps.append(PipelineStep.FEATURES)
    if not skip_train:
        steps.append(PipelineStep.TRAIN)
    if not skip_evaluate:
        steps.append(PipelineStep.EVALUATE)
    if not skip_predict:
        steps.append(PipelineStep.PREDICT)
    if not skip_report:
        steps.append(PipelineStep.REPORT)
    if not skip_select_stocks:
        steps.append(PipelineStep.SELECT_STOCKS)
    if not skip_analyze_stocks:
        steps.append(PipelineStep.ANALYZE_STOCKS)
    if not skip_generate_financial_reports:
        steps.append(PipelineStep.GENERATE_FINANCIAL_REPORTS)
    if not skip_send_reports:
        steps.append(PipelineStep.SEND_REPORTS)
    if archive_old_data:
        steps.append(PipelineStep.ARCHIVE)

    # 创建管道
    pipeline = DataPipeline(steps=steps)

    # 执行管道
    result = pipeline.run_full_pipeline(
        backfill_days=backfill_days,
        forward_fill_date=forward_fill_date,
        rps_periods=rps_periods,
        max_stocks=max_stocks,
        stock_codes=stock_codes,
        target_label=target_label,
        generate_html_report=generate_html_report,
        archive_days=archive_days,
        select_top_n=select_top_n,
        min_return=min_return,
        analysis_type=analysis_type,
        email_recipients=email_recipients,
        financial_reports_dir=financial_reports_dir,
        concurrent_reports=concurrent_reports,
        financial_analysis_timeout=financial_analysis_timeout,
        financial_report_stock_codes=financial_report_stock_codes,
    )

    # 打印结果
    pipeline.print_results(result)

    # 返回状态码
    sys.exit(0 if result.success else 1)

if __name__ == '__main__':
    main()
PYTHON_SCRIPT

# 执行管道脚本
echo "开始执行数据管道..."
echo ""

python /tmp/run_pipeline_$$.py
PIPELINE_EXIT_CODE=$?

# 清理临时脚本
rm -f /tmp/run_pipeline_$$.py

# 计算执行时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo "✅ 数据管道执行成功!"
else
    echo "❌ 数据管道执行失败!"
fi
echo ""
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "执行时长: ${MINUTES}分${SECONDS}秒"
echo "=========================================="

exit $PIPELINE_EXIT_CODE
