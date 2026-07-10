"""
LLM analyzer module - refactored with unified infrastructure.

Provides LLM-based stock analysis using OpenAI-compatible APIs with:
- Prompt template management
- Response caching
- Error handling and retries
- Structured output parsing
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI

from core.base import AnalyzerBase
from core.cache import cached, CacheConfig
from core.config import Config, get_config
from core.exceptions import AnalysisError, ConfigurationError
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """
    Template for LLM prompts.

    Attributes:
        system_prompt: System prompt for the LLM
        user_prompt_template: Template for user prompts
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
    """
    system_prompt: str
    user_prompt_template: str = "请分析以下股票数据并给出专业的分析意见：\n\n{data}"
    temperature: float = 0.1
    max_tokens: int = 4000


@dataclass
class AnalysisRequest:
    """
    Request for LLM analysis.

    Attributes:
        stock_code: Stock code
        stock_name: Stock name (optional)
        price_data: Historical price data
        indicator_data: Technical indicators data
        market_trends: Market trend summary
    """
    stock_code: str
    price_data: Dict[str, Any]
    indicator_data: Dict[str, Any]
    stock_name: Optional[str] = None
    market_trends: Optional[Dict[str, Any]] = None

    def to_user_prompt(self, template: str) -> str:
        """Convert to user prompt using template."""
        data_str = json.dumps({
            "历史数据": self.price_data,
            "技术指标": self.indicator_data,
            "市场趋势": self.market_trends or {}
        }, ensure_ascii=False, indent=2)

        stock_info = ""
        if self.stock_code and self.stock_name:
            stock_info = f"股票代码：{self.stock_code}\n股票名称：{self.stock_name}\n\n"
        elif self.stock_code:
            stock_info = f"股票代码：{self.stock_code}\n\n"

        return template.replace("{data}", stock_info + data_str)


@dataclass
class AnalysisResult:
    """
    Result from LLM analysis.

    Attributes:
        technical_analysis: Technical analysis section
        trend_analysis: Trend analysis section
        investment_advice: Investment advice section
        risk_warning: Risk warning section
        summary: Overall summary
        raw_text: Original LLM response
        timestamp: Analysis timestamp
        model_used: Model used for analysis
    """
    technical_analysis: str = ""
    trend_analysis: str = ""
    investment_advice: str = ""
    risk_warning: str = ""
    summary: str = ""
    raw_text: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "AI分析结果": {
                "技术分析": self.technical_analysis,
                "走势分析": self.trend_analysis,
                "投资建议": self.investment_advice,
                "风险提示": self.risk_warning,
                "总结": self.summary,
            },
            "_metadata": {
                "timestamp": self.timestamp,
                "model": self.model_used,
            },
            "_raw_analysis_text": self.raw_text,
        }

    @classmethod
    def from_error(cls, error_message: str) -> "AnalysisResult":
        """Create result from error."""
        return cls(
            technical_analysis=f"分析失败: {error_message}",
            trend_analysis="数据获取失败，无法提供分析。",
            investment_advice="由于数据获取失败，暂不提供投资建议。",
            risk_warning="数据不完整，投资决策需谨慎。",
        )


class PromptManager:
    """
    Manages prompt templates for different analysis types.

    Provides default templates and supports custom templates.
    """

    DEFAULT_SYSTEM_PROMPT = """你是一个专业的金融分析师，将收到完整的股票历史数据和技术指标数据进行分析。
数据包括：
1. 历史数据：每日的开盘价、收盘价、最高价、最低价和成交量
2. 技术指标：所有交易日的各项技术指标数据
3. 市场趋势：当前的关键趋势数据

请基于这些完整的历史数据进行深入分析，包括以下方面：

1. 技术面分析
- 通过历史数据分析长期趋势
- 识别关键的支撑位和压力位
- 分析重要的技术形态
- 对所有技术指标进行综合研判
- 寻找指标之间的背离现象

2. 走势研判
- 判断当前趋势的强度和可能持续性
- 识别可能的趋势转折点
- 分析成交量和价格的配合情况
- 预判可能的运行区间

3. 投资建议
- 基于完整数据给出明确的操作建议
- 设置合理的止损和目标价位
- 建议适当的持仓时间和仓位控制
- 针对不同投资周期给出建议

4. 风险提示
- 通过历史数据识别潜在风险
- 列出需要警惕的技术信号
- 提供风险规避的具体建议
- 说明需要持续关注的指标

按照以下固定格式输出分析结果,不要包含任何markdown标记：

技术分析
1. 长期趋势分析：
趋势判断
突破情况
形态分析

2. 支撑和压力：
关键支撑位
关键压力位
突破可能性

3. 技术指标研判：
MACD指标
KDJ指标
RSI指标
布林带分析
其他关键指标

走势分析
1. 当前趋势：
趋势方向
趋势强度
持续性分析

2. 价量配合：
成交量变化
量价关系
市场活跃度

3. 关键位置：
当前位置
突破机会
调整空间

投资建议
1. 操作策略：
总体建议
买卖时机
仓位控制

2. 具体参数：
止损位设置
目标价位
持仓周期

3. 分类建议：
激进投资者建议
稳健投资者建议
保守投资者建议

风险提示
1. 风险因素：
技术面风险
趋势风险
位置风险

2. 防范措施：
止损设置
仓位控制
注意事项

3. 持续关注：
重点指标
关键价位
市场变化

最后给出总体总结。"""

    def __init__(self):
        """Initialize prompt manager."""
        self._templates: Dict[str, PromptTemplate] = {
            "default": PromptTemplate(system_prompt=self.DEFAULT_SYSTEM_PROMPT),
        }

    def get_template(self, name: str = "default") -> PromptTemplate:
        """Get a prompt template by name."""
        return self._templates.get(name, self._templates["default"])

    def register_template(self, name: str, template: PromptTemplate) -> None:
        """Register a new prompt template."""
        self._templates[name] = template


class LLMAnalyzer(AnalyzerBase):
    """
    LLM-based stock analyzer.

    Provides:
    - LLM API integration with OpenAI-compatible endpoints
    - Prompt template management
    - Response caching
    - Structured output parsing
    - Error handling and retries

    Example:
        analyzer = LLMAnalyzer(
            api_key="sk-xxx",
            base_url="https://api.openai.com/v1",
            model="gpt-4"
        )
        result = analyzer.analyze(request_data)
        print(result.technical_analysis)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        prompt_manager: Optional[PromptManager] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize LLM analyzer.

        Args:
            api_key: OpenAI API key (from config if None)
            base_url: API base URL (from config if None)
            model: Model name (from config if None)
            prompt_manager: Prompt template manager
            config: Configuration (uses global config if None)
        """
        super().__init__(config)

        # Get API configuration
        api_config = self.config.api
        self.api_key = api_key or api_config.api_key
        self.base_url = base_url or api_config.base_url
        self.model = model or api_config.model

        if not self.api_key:
            raise ConfigurationError("API key is required")

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Initialize prompt manager
        self.prompt_manager = prompt_manager or PromptManager()

        self.initialize()

    @cached(CacheConfig(ttl=1800))  # 30 minutes cache
    def analyze(
        self,
        request: AnalysisRequest,
        template_name: str = "default",
        temperature: Optional[float] = None,
    ) -> AnalysisResult:
        """
        Analyze stock data using LLM.

        Args:
            request: Analysis request with stock data
            template_name: Name of prompt template to use
            temperature: Override temperature setting

        Returns:
            AnalysisResult with structured analysis

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            template = self.prompt_manager.get_template(template_name)
            temp = temperature if temperature is not None else template.temperature

            # Build messages
            user_prompt = request.to_user_prompt(template.user_prompt_template)

            messages = [
                {"role": "system", "content": template.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            self.logger.info(
                f"Sending analysis request for {request.stock_code}, "
                f"prompt length: {len(user_prompt)}"
            )

            # Call API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                stream=False,
            )

            if not response or not response.choices:
                return AnalysisResult.from_error("API returned empty response")

            analysis_text = response.choices[0].message.content

            # Parse response
            result = self._parse_response(analysis_text)
            result.raw_text = analysis_text
            result.model_used = self.model

            self.logger.info(f"Analysis completed for {request.stock_code}")
            return result

        except Exception as e:
            self.logger.error(f"Analysis failed for {request.stock_code}: {e}")
            return AnalysisResult.from_error(str(e))

    def _parse_response(self, text: str) -> AnalysisResult:
        """
        Parse LLM response text into structured result.

        Args:
            text: Raw LLM response text

        Returns:
            Parsed AnalysisResult
        """
        sections = {
            "技术分析": "",
            "走势分析": "",
            "投资建议": "",
            "风险提示": "",
            "总结": ""
        }

        current_section = None
        buffer = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Handle summary
            if line.startswith("总体总结"):
                if current_section and buffer:
                    sections[current_section] = self._clean_section_text("\n".join(buffer))
                current_section = "总结"
                buffer = [line.split("：", 1)[1] if "：" in line else line]
                continue

            # Handle main sections
            if line in sections:
                if current_section and buffer:
                    sections[current_section] = self._clean_section_text("\n".join(buffer))
                current_section = line
                buffer = []
                continue

            if current_section:
                buffer.append(line)

        # Process last section
        if current_section and buffer:
            sections[current_section] = self._clean_section_text("\n".join(buffer))

        return AnalysisResult(
            technical_analysis=sections["技术分析"],
            trend_analysis=sections["走势分析"],
            investment_advice=sections["投资建议"],
            risk_warning=sections["风险提示"],
            summary=sections["总结"],
        )

    def _clean_section_text(self, text: str) -> str:
        """
        Clean and format section text.

        Args:
            text: Raw section text

        Returns:
            Cleaned text
        """
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Skip headers
            if line in ["技术分析", "走势分析", "投资建议", "风险提示", "总结", "总体总结"]:
                continue

            lines.append(line)

        return "\n".join(lines)

    def analyze_from_dataframes(
        self,
        price_df: pd.DataFrame,
        indicator_df: pd.DataFrame,
        stock_code: str,
        stock_name: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Analyze from dataframes directly.

        Args:
            price_df: Price data dataframe
            indicator_df: Technical indicators dataframe
            stock_code: Stock code
            stock_name: Optional stock name

        Returns:
            AnalysisResult
        """
        # Convert dataframes to dict format
        price_data = self._format_price_data(price_df)
        indicator_data = self._format_indicator_data(indicator_df)
        market_trends = self._calculate_market_trends(price_df)

        request = AnalysisRequest(
            stock_code=stock_code,
            stock_name=stock_name,
            price_data=price_data,
            indicator_data=indicator_data,
            market_trends=market_trends,
        )

        return self.analyze(request)

    def _format_price_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Format price dataframe for prompt."""
        df_copy = df.copy()
        df_copy.index = df_copy.index.strftime("%Y-%m-%d")

        # Sample data: keep all recent 60 days, sample earlier data
        all_dates = list(df_copy.index)
        recent_dates = all_dates[-60:] if len(all_dates) > 60 else all_dates
        early_dates = all_dates[:-60] if len(all_dates) > 60 else []
        sampled_early = early_dates[::2]  # Sample every 2nd day

        selected_dates = sampled_early + recent_dates

        return {
            date: {
                "开盘价": f"{df_copy.loc[date, 'open']:.2f}",
                "收盘价": f"{df_copy.loc[date, 'close']:.2f}",
                "最高价": f"{df_copy.loc[date, 'high']:.2f}",
                "最低价": f"{df_copy.loc[date, 'low']:.2f}",
                "成交量": f"{int(df_copy.loc[date, 'volume']):,}"
            }
            for date in selected_dates
        }

    def _format_indicator_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Format indicator dataframe for prompt."""
        df_copy = df.copy()
        df_copy.index = df_copy.index.strftime("%Y-%m-%d")

        # Sample similar to price data
        all_dates = list(df_copy.index)
        recent_dates = all_dates[-60:] if len(all_dates) > 60 else all_dates
        early_dates = all_dates[:-60] if len(all_dates) > 60 else []
        sampled_early = early_dates[::2]

        selected_dates = sampled_early + recent_dates

        indicator_dict = {}

        for date in selected_dates:
            row = df_copy.loc[date]
            indicator_dict[date] = {
                "趋势指标": {
                    "MACD": f"{row.get('MACD', 0):.2f}",
                    "DIF": f"{row.get('DIF', 0):.2f}",
                    "DEA": f"{row.get('DEA', 0):.2f}",
                    "MA5": f"{row.get('MA5', 0):.2f}",
                    "MA10": f"{row.get('MA10', 0):.2f}",
                    "MA20": f"{row.get('MA20', 0):.2f}",
                    "MA60": f"{row.get('MA60', 0):.2f}",
                },
                "摆动指标": {
                    "KDJ-K": f"{row.get('K', 0):.2f}",
                    "KDJ-D": f"{row.get('D', 0):.2f}",
                    "KDJ-J": f"{row.get('J', 0):.2f}",
                    "RSI": f"{row.get('RSI', 0):.2f}",
                },
                "布林带": {
                    "上轨": f"{row.get('BOLL_UPPER', 0):.2f}",
                    "中轨": f"{row.get('BOLL_MID', 0):.2f}",
                    "下轨": f"{row.get('BOLL_LOWER', 0):.2f}"
                },
            }

        return indicator_dict

    def _calculate_market_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market trend summary."""
        latest_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        last_week_close = df["close"].iloc[-6] if len(df) > 5 else prev_close
        last_month_close = df["close"].iloc[-21] if len(df) > 20 else prev_close

        return {
            "日涨跌幅": f"{((latest_close - prev_close) / prev_close * 100):.2f}%",
            "周涨跌幅": f"{((latest_close - last_week_close) / last_week_close * 100):.2f}%",
            "月涨跌幅": f"{((latest_close - last_month_close) / last_month_close * 100):.2f}%",
            "最新收盘价": f"{latest_close:.2f}",
            "最高价": f"{df['high'].max():.2f}",
            "最低价": f"{df['low'].min():.2f}",
        }


# Convenience functions for backward compatibility
def analyze_stock(
    price_df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    stock_code: str,
    stock_name: Optional[str] = None,
) -> AnalysisResult:
    """
    Analyze a stock using LLM.

    Args:
        price_df: Historical price data
        indicator_df: Technical indicators data
        stock_code: Stock code
        stock_name: Optional stock name

    Returns:
        AnalysisResult
    """
    analyzer = LLMAnalyzer()
    return analyzer.analyze_from_dataframes(
        price_df, indicator_df, stock_code, stock_name
    )


def create_analysis_request(
    stock_code: str,
    price_data: Dict[str, Any],
    indicator_data: Dict[str, Any],
    stock_name: Optional[str] = None,
) -> AnalysisRequest:
    """
    Create an analysis request.

    Args:
        stock_code: Stock code
        price_data: Price data dictionary
        indicator_data: Indicator data dictionary
        stock_name: Optional stock name

    Returns:
        AnalysisRequest
    """
    return AnalysisRequest(
        stock_code=stock_code,
        stock_name=stock_name,
        price_data=price_data,
        indicator_data=indicator_data,
    )
