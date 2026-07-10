<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个A股量化分析与财报分析的智能系统，结合了技术指标分析、机器学习预测和LLM智能分析功能。项目使用Python开发，主要依赖baostock获取A股数据，使用OpenAI兼容API进行智能分析。

## 常用命令

### 环境激活
```bash
conda activate ashare-llm-analyst
```

### 运行示例程序
```bash
# 测试baostock数据获取
python Test.py

# 运行财报分析主程序
cd deep-search
python main.py <公司名称或股票代码>
# 例如: python main.py 平安银行 或 python main.py 000001

# 生成股票分析报告
cd out
python 04-stock-analysis.py

# 运行K线形态分析
cd back
python independent_candle_pattern_analyzer.py
```

### 安装依赖
```bash
pip install -r requirement.txt
```

## 项目架构

### 核心目录结构

```
ashare-llm-analyst/
├── back/               # 后台分析模块
│   ├── independent_candle_pattern_analyzer.py  # 独立K线形态分析器
│   └── daily_stock_data_analyzer.py            # 每日股票数据分析
├── deep-search/        # 深度财报分析模块
│   ├── main.py                                # 财报分析主程序入口
│   ├── financial_analyzer_v2.py               # 重构后的财务分析器
│   ├── mcp_service.py                         # MCP服务客户端
│   ├── data_fetcher.py                        # 数据获取器
│   ├── search_engine.py                       # 搜索引擎
│   ├── ai_analyzer.py                         # AI分析器
│   ├── config_manager.py                      # 配置管理器
│   ├── logger_manager.py                      # 日志管理器
│   ├── cache_manager.py                       # 缓存管理器
│   └── exception_handler.py                   # 异常处理器
├── src/                # 核心源代码模块
│   ├── core/                                   # 核心基础设施
│   │   ├── base.py                            # 基类定义(AnalyzerBase, FileSystemProvider等)
│   │   ├── config.py                          # 配置管理(Config, get_config)
│   │   ├── logger.py                          # 日志系统(get_logger)
│   │   ├── cache.py                           # 缓存系统(cache decorator, CacheConfig)
│   │   ├── exceptions.py                      # 异常定义(DataFetchError等)
│   │   └── parallel.py                        # 并行处理工具
│   └── data/                                   # 数据管道模块
│       ├── providers.py                       # 数据提供者(DataProvider, BaostockProvider等)
│       ├── downloaders.py                     # 数据下载器(IncrementalDownloader等)
│       ├── rps_calculator.py                  # RPS计算器(RPSPeriodsCalculator)
│       ├── feature_engineering.py             # 特征工程(FeatureCalculator, MLTrainingDataGenerator)
│       ├── model_trainer.py                   # 模型训练器(RandomForestTrainer)
│       └── pipeline.py                        # 数据管道编排器(DataPipeline)
├── out/                # 输出和分析工具
│   ├── 04-stock-analysis.py                   # 股票技术分析工具
│   ├── llm.py                                 # LLM分析器
│   ├── Ashare.py                              # A股数据接口
│   └── MyTT_optimized.py                      # 优化的技术指标库
├── examples/           # 使用示例
│   └── data_pipeline_example.py               # 数据管道使用示例
└── requirement.txt     # 依赖清单
```

### 三大核心模块

#### 1. 技术分析模块 (back/ 和 out/)

- **independent_candle_pattern_analyzer.py**: 独立的K线形态识别器，不依赖其他模块，支持识别锤子线、吞没形态、十字星、早晨之星等多种K线形态
- **04-stock-analysis.py**: 完整的股票技术分析工具，整合了多指标综合分析和LLM智能分析

#### 2. 财报分析模块 (deep-search/)

财报分析采用9步流程：
1. 获取上市公司名称或股票代码
2. 搜索该公司最新财务报告
3. 提取财报关键财务指标
4. 对比行业平均水平
5. 分析财报趋势变化
6. 识别潜在风险或异常项
7. 生成综合分析报告
8. 校验修正报告内容（真实性、合理性、时效性）
9. MCP服务增强分析（知识库优先、网络搜索补充）

核心类：
- **FinancialAnalyzer**: 9步财报分析主类
- **FinancialAnalyzerV2**: 重构后的模块化分析器

#### 3. 数据源模块

- **baostock**: 主要的A股K线数据和财务数据来源
- **东方财富API**: 实时行情、行业数据、公司信息
- **Tavily搜索**: 新闻和公告搜索
- **MCP知识库**: 增强分析的知识库服务

#### 4. 数据管道模块 (src/data/)

重构后的数据管道模块提供完整的数据流程：下载 → RPS → 特征 → 训练

**核心组件：**

| 模块 | 类 | 功能 |
|------|-----|------|
| `providers.py` | `DataProvider` | 数据提供者抽象基类 |
| | `BaostockProvider` | baostock API封装，支持K线数据查询 |
| | `CSVProvider` | CSV文件读写，继承FileSystemProvider |
| | `CachedProvider` | 带缓存的数据提供者装饰器 |
| `downloaders.py` | `AntiCrawlerController` | 反爬虫控制器，请求频率限制 |
| | `RequestRetryManager` | 请求重试管理器，指数退避 |
| | `IncrementalDownloader` | 增量数据下载器 |
| `rps_calculator.py` | `RPSPeriodsCalculator` | 多周期RPS计算器 |
| `feature_engineering.py` | `FeatureCalculator` | 技术指标计算器(MACD, KDJ, RSI, BOLL等) |
| | `MLTrainingDataGenerator` | ML训练数据生成器 |
| | `MLTrainingConfig` | 训练配置类 |
| `model_trainer.py` | `ModelTrainerBase` | 模型训练器抽象基类 |
| | `RandomForestTrainer` | 随机森林训练器 |
| `pipeline.py` | `DataPipeline` | 数据管道编排器 |
| | `PipelineStep` | 管道步骤枚举(12个步骤) |
| | `run_daily_pipeline()` | 每日管道执行便捷函数 |
| `stock_selector.py` | `StockSelector` | ML选股器，基于预测结果筛选股票 |
| `stock_analyzer.py` | `StockAnalyzer` | 股票技术分析器，生成交易信号 |
| `email_sender.py` | `EmailSender` | 邮件发送器，发送分析报告 |
| `validators.py` | `DataValidator` | 数据质量验证器 |
| `model_evaluator.py` | `ModelEvaluator` | 模型评估器 |
| `batch_predictor.py` | `BatchPredictor` | 批量预测器 |
| `reporter.py` | `PipelineReporter` | 报告生成器 |
| `archiver.py` | `DataArchiver` | 数据归档器 |
| `monitor.py` | `PipelineMonitor` | 性能监控器 |

**使用示例：**

```python
# 导入数据管道模块
from src.data.providers import BaostockProvider, CSVProvider, create_provider
from src.data.downloaders import IncrementalDownloader
from src.data.rps_calculator import RPSPeriodsCalculator
from src.data.feature_engineering import FeatureCalculator, MLTrainingDataGenerator
from src.data.model_trainer import RandomForestTrainer
from src.data.pipeline import DataPipeline, PipelineStep, run_daily_pipeline
from src.data.stock_selector import StockSelector
from src.data.stock_analyzer import StockAnalyzer
from src.data.email_sender import EmailSender

# 1. 使用数据提供者获取数据
provider = BaostockProvider()
with provider:
    df = provider.load("sh.600000", start_date="2024-01-01", end_date="2024-12-31")

# 2. 增量下载最新数据
downloader = IncrementalDownloader()
result = downloader.incremental_update(backfill_days=5)

# 3. 计算RPS指标
calculator = RPSPeriodsCalculator()
rps_data = calculator.calculate_multi_period_rps(periods=[5, 10, 20, 60])

# 4. 计算技术指标
feat_calc = FeatureCalculator()
df_with_features = feat_calc.calculate_all_features(df)

# 5. 生成训练数据
generator = MLTrainingDataGenerator()
generator.generate_training_data(output_dir="training_data")

# 6. 训练模型
trainer = RandomForestTrainer()
results = trainer.train_models("return_5d_gt_5pct")

# 7. 执行完整管道
pipeline = DataPipeline()
result = pipeline.run_full_pipeline(
    backfill_days=5,
    rps_periods=[5, 10, 20, 60],
    target_label="return_5d_gt_5pct"
)

# 或使用便捷函数
result = run_daily_pipeline(backfill_days=5)

# 8. ML选股
selector = StockSelector()
result = selector.select_stocks(top_n=10, min_return=5.0)

# 9. 股票技术分析
analyzer = StockAnalyzer()
result = analyzer.analyze_stocks(analysis_type="technical")

# 10. 发送邮件报告
sender = EmailSender()
result = sender.send_reports(recipients=["user@example.com"])

# 11. 执行包含新步骤的完整管道
pipeline = DataPipeline(steps=[
    PipelineStep.DOWNLOAD,
    PipelineStep.VALIDATE,
    PipelineStep.RPS,
    PipelineStep.FEATURES,
    PipelineStep.TRAIN,
    PipelineStep.SELECT_STOCKS,
    PipelineStep.ANALYZE_STOCKS,
    PipelineStep.SEND_REPORTS,
])
result = pipeline.run_full_pipeline(
    select_top_n=10,
    min_return=5.0,
    analysis_type="technical",
    email_recipients=["user@example.com"]
)
```

**数据流程：**

```
原始数据下载 (IncrementalDownloader + BaostockProvider)
    ↓
数据质量验证 (DataValidator)
    ↓
股票K线数据 (CSV文件存储)
    ↓
RPS计算 (RPSPeriodsCalculator)
    ↓
技术指标计算 (FeatureCalculator)
    ↓
训练数据生成 (MLTrainingDataGenerator)
    ↓
模型训练 (RandomForestTrainer)
    ↓
模型评估 (ModelEvaluator)
    ↓
模型文件 (joblib保存)
    ↓
批量预测 (BatchPredictor)
    ↓
ML选股 (StockSelector) ← 新增
    ↓
股票技术分析 (StockAnalyzer) ← 新增
    ↓
邮件发送 (EmailSender) ← 新增
```

## 开发规范

### 错误处理原则
根据 `.trae/rules/project_rules.md` 中的规范：
- 代码执行前必须先激活conda环境
- **不要做假设条件**，获取不到数据就报错，不要掩盖错误
- **错误不可怕，伪正确才可怕**，要在错误中学习

### baostock API使用

API文档：http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3

核心API：
- `login()` / `logout()`: 登录登出
- `query_history_k_data_plus()`: 获取历史K线数据
- `query_profit_data()`: 季频盈利能力
- `query_operation_data()`: 季频营运能力
- `query_growth_data()`: 季频成长能力
- `query_balance_data()`: 季频偿债能力
- `query_cash_flow_data()`: 季频现金流量
- `query_stock_industry()`: 行业分类

### 环境变量配置

主要环境变量（通过.env文件配置）：
- `OPENAI_API_KEY`: OpenAI API密钥
- `OPENAI_BASE_URL`: OpenAI API地址（默认使用内部部署地址）
- `OPENAI_MODEL`: 使用的模型（默认Qwen3-Next-80B-A3B）
- `SEARCH_API_KEY`: 搜索API密钥（Tavily）
- `HTTP_PROXY` / `HTTPS_PROXY`: 代理配置
- `WEKNORA_KB_ID`: 知识库ID

## 技术指标说明

项目支持的技术指标（通过MyTT_optimized.py）：
- 趋势类：MACD、DIF、DEA、DMA、MA、BBI
- 震荡类：KDJ、RSI、CCI
- 压力支撑：BOLL（布林带上中下轨）
- 能量类：OBV、VR、ROC
- 方向指标：PDI、MDI、ADX

## 数据文件说明

- `history_k_data.csv`: 存储股票历史K线数据
- `downloads/`: 存储下载的财报PDF文件
- `out/reports/`: 存储生成的分析报告

## 常见任务

### 使用数据管道进行每日更新
```bash
# 方式1: 使用Python脚本
python -c "
from src.data.pipeline import run_daily_pipeline
result = run_daily_pipeline(backfill_days=5)
print(f'成功: {result.success}')
"

# 方式2: 运行示例程序
python examples/data_pipeline_example.py

# 方式3: 使用 run_pipeline.py 脚本
python run_pipeline.py

# 方式4: 使用 run_pipeline.sh 脚本
bash run_pipeline.sh

# 执行完整流程（包括选股、分析、邮件报告）
python run_pipeline.py --skip-select-stocks --skip-analyze-stocks --skip-send-reports
```

### 新增的管道步骤

项目新增了三个管道步骤，与 `out/09_run_stock_downloader.sh` 的功能对齐：

| 步骤 | 参数 | 说明 |
|------|------|------|
| SELECT_STOCKS | `--select-top-n N` | 选择前N只股票 (默认: 10) |
| | `--min-return N` | 最低收益率阈值% (默认: 5.0) |
| ANALYZE_STOCKS | `--analysis-type TYPE` | 分析类型 (默认: technical) |
| SEND_REPORTS | `--email-recipients EMAILS` | 邮件收件人列表 |

这些步骤默认**跳过**，需要显式启用：

```bash
# 启用新步骤执行完整流程
python run_pipeline.py --skip-select-stocks --skip-analyze-stocks --skip-send-reports

# 或使用 bash 脚本（注意：bash 参数值为 0 表示不跳过，即启用）
bash run_pipeline.sh --skip-select-stocks 0 --skip-analyze-stocks 0 --skip-send-reports 0
```

### 添加新的技术指标
在 `out/MyTT_optimized.py` 中添加新指标计算函数，然后在 `out/04-stock-analysis.py` 中调用。

或者在新架构中，在 `src/data/feature_engineering.py` 的 `FeatureCalculator` 类中添加新方法：

```python
@cached(config=CacheConfig(ttl=3600))
def calculate_new_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
    """计算新指标"""
    # 实现指标计算逻辑
    pass
```

### 修改财报分析流程
修改 `deep-search/main.py` 中的 `FinancialAnalyzer` 类的9步分析流程。

### 添加新的K线形态识别
在 `back/independent_candle_pattern_analyzer.py` 中的 `IndependentCandlePatternAnalyzer` 类添加新的检测方法。

### 自定义分析报告模板
修改 `out/04-stock-analysis.py` 中的报告生成部分或LLM提示词。

### 扩展数据管道添加新步骤
在 `src/data/pipeline.py` 中：

1. 在 `PipelineStep` 枚举中添加新步骤
2. 在 `DataPipeline._execute_step()` 中添加处理逻辑
3. 实现新的执行方法（如 `_execute_new_step()`）

```python
class PipelineStep(Enum):
    DOWNLOAD = "download"
    VALIDATE = "validate"
    RPS = "rps"
    FEATURES = "features"
    TRAIN = "train"
    EVALUATE = "evaluate"
    PREDICT = "predict"
    REPORT = "report"
    SELECT_STOCKS = "select_stocks"  # ML选股
    ANALYZE_STOCKS = "analyze_stocks"  # 股票分析
    SEND_REPORTS = "send_reports"  # 邮件发送
    ARCHIVE = "archive"
    NEW_STEP = "new_step"  # 添加新步骤
```

### 自定义模型训练参数
修改 `src/data/model_trainer.py` 中的 `RandomForestTrainer.model_params`：

```python
self.model_params = {
    "n_estimators": 300,  # 增加树的数量
    "max_depth": 30,      # 增加树的最大深度
    # ... 其他参数
}
```

## 依赖项说明

核心依赖：
- `baostock`: A股数据源
- `pandas`: 数据处理
- `numpy`: 数值计算
- `openai`: LLM API调用
- `matplotlib`/`plotly`: 图表绘制
- `requests`: HTTP请求
- `python-dotenv`: 环境变量管理
