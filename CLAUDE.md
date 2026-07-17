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

A股量化分析与财报分析系统：技术指标 + 机器学习选股 + LLM 智能分析。数据源以 baostock 为主，东方财富/akshare 辅助，LLM 走 OpenAI 兼容 API（默认 Qwen3-Next-80B-A3B）。

代码分两层：**`src/` 是版本化、重构后的核心**（commit `b2259b8` 起的分层架构），**`out/`、`back/`、`deep-search/` 是更早的独立脚本**，与 `src/` 有重复。新工作应落在 `src/`，旧脚本仅为参考。

## 环境与常用命令

**必须先激活环境**（项目根约定，见 `.trae/rules/project_rules.md`）：
```bash
conda activate ashare-llm-analyst
```

```bash
# 数据连通性冒烟测试（baostock 登录 + 拉一只票）
python Test.py

# 完整数据管道（下载→RPS→特征→训练→评估→预测→报告）
python run_pipeline.py          # 或 bash run_pipeline.sh
python examples/data_pipeline_example.py   # 带注释的最小示例

# 单元测试（unittest 风格，非 pytest —— requirement.txt 未装 pytest）
python tests/run_tests.py                 # 跑全部 6 个核心管道测试
python tests/test_providers.py            # 跑单个测试文件（直接当脚本运行）
python tests/run_tests.py providers       # 按模块名跑单个

# ML 信号分层回测（B2）：用 B1 修复后的特征训随机森林，样本外分层
python src/backtest/run_model_backtest.py
# 可选环境变量: N_STOCKS(默认150) HORIZON(默认5) THRESHOLD(默认0.05)

# 财报深度分析（9 步流程）
cd deep-search && python main.py 平安银行      # 按公司名
cd deep-search && python main.py 000001        # 或股票代码
```

依赖只通过 `requirement.txt` / `environment.yml` 管理（环境名 `ashare-llm-analyst`）。无 `pyproject.toml`、无 ruff/black/mypy 配置；`.ruff_cache` 是手动跑过留下的，不是 CI 强制项。

## 架构（需要跨多文件才能理解的全局图）

### 1. `src/` 分层包结构

| 包 | 职责 | 关键模块 |
|----|------|----------|
| `src/core/` | 基础设施：配置/日志/缓存/异常/并行/基类 | `config.py`(Config/get_config), `base.py`(AnalyzerBase, FileSystemProvider), `cache.py`(@cached 装饰器) |
| `src/data/` | **数据管道主体**（见下） | `pipeline.py`, `providers.py`, `feature_engineering.py`, `model_trainer.py`, … |
| `src/analysis/` | 三大分析面：技术/选股/LLM/报告 | `technical_analyzer.py`, `stock_selector.py`, `llm_analyzer.py`, `report_generator.py` |
| `src/backtest/` | 诚实分层回测（B2 新增） | `layered_backtester.py`(LayeredBacktester), `run_model_backtest.py`(入口) |
| `src/utils/` | 通用工具 | `data_utils.py`, `date_utils.py`, `validation.py` |

### 2. 数据管道编排（`src/data/pipeline.py`）

`DataPipeline` 通过 `PipelineStep` 枚举按序编排，支持 `run_full_pipeline()` / `run_from_step()` 断点续跑。步骤：

```
DOWNLOAD → VALIDATE → RPS → FEATURES → TRAIN → EVALUATE → PREDICT → REPORT
                                                              ↳ 可选: SELECT_STOCKS, ANALYZE_STOCKS,
                                                                       GENERATE_FINANCIAL_REPORTS,
                                                                       ARCHIVE, SEND_REPORTS, MONITOR
```

- **数据访问统一走 Provider 抽象**：`DataProvider`（基类）→ `BaostockProvider` / `CSVProvider`，可用 `CachedProvider` 装饰器叠加缓存。新数据源继承 `DataProvider`，不要在业务层直接调 baostock。
- **特征工程集中在 `feature_engineering.py`**：`FeatureCalculator`（MACD/KDJ/RSI/BOLL/ADX/ROC/BBI…）+ `MLTrainingDataGenerator`（生成训练集 + 标签）。新增指标加方法，并用 `@cached` 包裹。
- **CSV 落盘约定**：`market_data/<code>.csv`（K 线）、`training_data/`、`models/`（joblib）、`rps_results/`、`reports/`、`downloads/`（财报 PDF）。

### 3. ⚠️ ML 可信度红线（B1/B2，进行中方向，改动 ML 代码必读）

近期重构的核心目标是**消除 ML 的伪正确**。触及任何特征/标签/回测代码时必须遵守：

**B1 — 数据泄露修复**（commit `65e5f28`）：
- 训练/测试**按时序切分**，绝不随机 shuffle 跨期数据。
- 特征**禁止 `bfill`**（会偷看未来）；标签用 `valid_mask` 过滤无效样本。
- 特征**平稳化**：剔除 `close`、`OBV` 等非平稳原始量，只用比值/技术指标类平稳特征。B1 后的特征列模式见 `src/backtest/run_model_backtest.py` 顶部 `PATTERNS`。

**B2 — 诚实分层回测**（commit `8700f46`/`7fbbb0f`）：
- `LayeredBacktester` 必须计入**交易成本、T+1、涨跌停（涨停买不进/跌停卖不出）、停牌**。
- `run_model_backtest.py` 目前是 **pseudo walk-forward**（前 70% 训练后固定，对后 30% 预测），**非每调仓日重训** —— 严格 walk-forward 是后续工作，不要误把它当成滚动训练。

当前 B1 后的 AUC 应回落到 0.52–0.58（近乎无信号）才说明泄露已修；若特征工程后 AUC 飙高，先怀疑泄露而非庆祝。

### 4. 三大分析面（各有新旧两套，优先 `src/`）

| 面向 | 旧脚本 | `src/` 新实现 |
|------|--------|----------------|
| 技术分析 | `out/04-stock-analysis.py` + `out/MyTT_optimized.py` | `src/analysis/technical_analyzer.py` |
| 选股 | `out/03-custom_stock_selection.py` | `src/analysis/stock_selector.py` / `src/data/stock_selector.py` |
| LLM 分析 | `out/llm.py` | `src/analysis/llm_analyzer.py` |
| 财报分析 | `deep-search/`（9 步 `FinancialAnalyzer`/`FinancialAnalyzerV2`） | `src/data/financial_reports.py`（管道步骤 `GENERATE_FINANCIAL_REPORTS`） |
| K 线形态 | `back/independent_candle_pattern_analyzer.py` | （尚未迁入 src） |

## 领域规则（跨多文件、不易从代码直接看出）

### baostock API（主数据源，文档 http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3 ）
- 必须 `login()` / `logout()` 成对（`BaostockProvider` 用上下文管理器包好）。
- K 线：`query_history_k_data_plus()`；除权除息/复权因子：`query_dividend_data()` / `query_adjust_factor()`。
- 季频财务：盈利 `query_profit_data`、营运 `query_operation_data`、成长 `query_growth_data`、偿债 `query_balance_data`、现金流 `query_cash_flow_data`、杜邦 `query_dupont_data`。
- 元信息：交易日 `query_trade_dates`、全证券 `query_all_stock`、基本资料 `query_stock_basic`、行业 `query_stock_industry`、指数成分 `query_sz50/hs300/zz500_stocks`。

### 配置（`.env`，见 `.env.example`）
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`（默认 Qwen3-Next-80B-A3B，走内部部署地址）
- `SEARCH_API_KEY`（Tavily，新闻/公告搜索）、`WEKNORA_KB_ID`（MCP 知识库）
- `HTTP_PROXY` / `HTTPS_PROXY`

## 工程约定（项目核心铁律，源自 `.trae/rules/project_rules.md`）

> **不要做假设条件，获取不到数据就报错，不要掩盖错误。错误不可怕，伪正确才可怕。**

- 取数失败必须 raise，禁止静默 `try/except` 兜底成假数据。这条优先级最高，凌驾于通用「优雅降级」直觉之上。
- 代码执行前先 `conda activate ashare-llm-analyst`。

## 变更管理

架构/能力/破坏性改动走 **OpenSpec**：提案放 `openspec/changes/`，规格 `openspec/specs/`，动手前先读 `openspec/AGENTS.md`。当前推进方向：A1 工程化地基已完成，正在做 B（ML 可信度，B1/B2 进行中）/ C（财报可信度）。

## 项目推进流水线·dev-loop 自治守则（ADR-0003）

> 本节给 `scripts/dev-agent.py` 触发的自治 dev loop 用（控制面 dispatch 投递 PRD 后，agent 在 feature 分支上自主改码 + 自验）。仅约束「自治 dev」场景，不影响人肉开发。

**身份与触发**
- 由 `scripts/dev-agent.py`（claude-agent-sdk `query()`，`permission_mode=acceptEdits` + 有界 `allowedTools`）驱动；模型走 roc 代理默认（glm-5.2）。
- 分支模式（branch-only，基点 `master`）：在 `auto/<stamp>-<slug>` feature 分支上干，**永不直推 `master`**（主干有 branch protection）。
- push / 开 PR 由 `dev-agent.py` 在你停下后代办——**你不要自己 push、不要自己开 PR**。

**作业范围**
- 只在当前 feature 分支、本仓范围内改。
- 新工作落 `src/`（版本化分层核心，见「架构 §1」）。`out/`、`back/`、`deep-search/` 是更早的独立脚本，**仅为参考、不要在里面改**。
- 数据产物目录（`training_data/`、`market_data/`、`downloads/` 等）**只读，勿改勿删**。

**验证闸（铁律）**
- 改完必须跑 `python tests/run_tests.py`（= unittest 全套；也可 `python tests/test_providers.py` 单文件）。
- **绿才算完事**。红（test 失败 / import 错 / 取数 raise）= 没做完，继续修或回滚，**绝不留红提交**。
- 遵循「工程约定」：取数失败必须 raise，**禁止静默 `try/except` 兜底成假数据**——伪正确比报错更可怕。

**子代理分工（Agent 工具）**
- 满足任一才考虑 spawn 子代理，否则单干（子代理独立 context 有成本，别为分工而分工）：PRD 横跨多个独立关注点 / 估摸单干 30+ turn / 读改 5+ 文件。
- 分工纪律：每个子代理单一职责 + 明确目标 + 限定文件范围；子代理产出回 parent 整合，parent 跑 `python tests/run_tests.py` 验证整体；**commit / push 只由 parent 守**。

**提交与停止**
- commit 用 conventional commits（`feat:` / `fix:` / `refactor:` …），只 commit 与 PRD 相关的改动。
- 到「可提交且 test 绿」即停；停下前用一段话总结：改了什么 / test 结果 / 遗留风险。
- 无进展刹车（SPEC #27）：验证红后连续 N 轮（默认 3）无写类 tool_use → `dev-agent.py` 主动 stalled（exit 12，不开 PR，半成品靠 `state/runs/*.jsonl` 留痕）。

**禁区**
- 不直推 / 不 force push `master`；不删分支；不 `.github` branch protection。
- 不 `pip publish`；不动 `training_data/` `market_data/` `downloads/`。
- 不静默兜底假数据；不在仓内引入未经 `requirement.txt` / `environment.yml` 登记的依赖。
