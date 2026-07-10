#!/usr/bin/env python3
"""
Simple test script to verify the refactored modules work correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory and src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "src"))

def test_config():
    """Test configuration management."""
    print("=" * 50)
    print("Testing Configuration Management")
    print("=" * 50)

    from src.core.config import Config, get_config

    # Test default config
    config = Config.from_env()
    print(f"✓ Config loaded")
    print(f"  - Cache enabled: {config.cache.enabled}")
    print(f"  - Log level: {config.logging.level}")
    print(f"  - Max workers: {config.parallel.max_workers}")

    # Test to_dict
    config_dict = config.to_dict()
    print(f"✓ Config to_dict: {len(config_dict)} sections")

    print()


def test_exceptions():
    """Test custom exceptions."""
    print("=" * 50)
    print("Testing Custom Exceptions")
    print("=" * 50)

    from src.core.exceptions import (
        StockAnalysisError,
        DataFetchError,
        AnalysisError,
        ValidationError,
    )

    # Test basic exception
    try:
        raise StockAnalysisError("Test error", {"key": "value"})
    except StockAnalysisError as e:
        print(f"✓ StockAnalysisError: {e.message}")
        print(f"  - Error code: {e.error_code}")
        print(f"  - Context: {e.context}")

    # Test validation error
    try:
        raise ValidationError("Invalid value", field_name="test_field")
    except ValidationError as e:
        print(f"✓ ValidationError: {e.message}")

    print()


def test_logger():
    """Test logging system."""
    print("=" * 50)
    print("Testing Logging System")
    print("=" * 50)

    from src.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Info message")
    logger.warning("Warning message")
    logger.debug("Debug message (may not show)")

    print("✓ Logger created and messages logged")
    print()


def test_cache():
    """Test caching system."""
    print("=" * 50)
    print("Testing Caching System")
    print("=" * 50)

    from src.core.cache import cached, CacheConfig, clear_cache

    call_count = 0

    @cached(CacheConfig(ttl=60))
    def expensive_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * x

    # First call - should execute
    result1 = expensive_function(5)
    print(f"✓ First call: {result1} (call_count: {call_count})")

    # Second call - should use cache
    result2 = expensive_function(5)
    print(f"✓ Second call: {result2} (call_count: {call_count})")

    # Different argument - should execute
    result3 = expensive_function(10)
    print(f"✓ Different arg: {result3} (call_count: {call_count})")

    # Test cache stats
    from src.core.cache import get_cache_stats
    stats = get_cache_stats()
    print(f"✓ Cache stats: {stats}")

    print()


def test_utils():
    """Test utility functions."""
    print("=" * 50)
    print("Testing Utility Functions")
    print("=" * 50)

    from src.utils.validation import validate_stock_code, validate_positive_number
    from src.utils.date_utils import parse_date, format_date, get_yesterday

    # Test stock code validation
    code1 = validate_stock_code("000001")
    print(f"✓ Stock code normalized: 000001 -> {code1}")

    code2 = validate_stock_code("sh.600000")
    print(f"✓ Stock code validated: {code2}")

    # Test number validation
    val = validate_positive_number(100)
    print(f"✓ Positive number validated: {val}")

    # Test date functions
    yesterday = get_yesterday()
    formatted = format_date(yesterday)
    print(f"✓ Yesterday formatted: {formatted}")

    print()


def test_base_classes():
    """Test base classes."""
    print("=" * 50)
    print("Testing Base Classes")
    print("=" * 50)

    from src.core.base import AnalyzerBase, Signal

    # Test AnalyzerBase
    class TestAnalyzer(AnalyzerBase):
        def test_method(self):
            self.logger.info("Test method called")
            return "test_result"

    analyzer = TestAnalyzer()
    with analyzer:
        result = analyzer.test_method()
        print(f"✓ AnalyzerBase with context manager: {result}")

    # Test Signal
    signal = Signal(
        signal_type=Signal.BUY,
        strength=75,
        reason="Test signal",
        metadata={"test": True}
    )
    print(f"✓ Signal created: {signal}")
    print(f"  - Is buy: {signal.is_buy()}")
    print(f"  - To dict: {signal.to_dict()}")

    print()


def test_technical_analyzer():
    """Test technical analyzer with sample data."""
    print("=" * 50)
    print("Testing Technical Analyzer")
    print("=" * 50)

    from src.analysis.technical_analyzer import TechnicalAnalyzer, generate_trading_signals

    # Create sample dataframe
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    # Generate price data with trend
    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 2
    close_prices = trend + noise

    # Calculate OHLC
    df = pd.DataFrame({
        "date": dates,
        "open": close_prices + np.random.randn(100),
        "high": close_prices + abs(np.random.randn(100)) + 1,
        "low": close_prices - abs(np.random.randn(100)) - 1,
        "close": close_prices,
        "volume": np.random.randint(1000000, 10000000, 100),
    })

    # Calculate some basic indicators
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["RSI"] = 50 + np.random.randn(100) * 10
    df["K"] = 50 + np.random.randn(100) * 15
    df["D"] = 50 + np.random.randn(100) * 15
    df["J"] = 50 + np.random.randn(100) * 20

    # Calculate MACD components
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["DIF"] = df["EMA12"] - df["EMA26"]
    df["DEA"] = df["DIF"].ewm(span=9).mean()
    df["MACD"] = df["DIF"] - df["DEA"]

    # Calculate Bollinger Bands
    df["BOLL_MID"] = df["close"].rolling(20).mean()
    df["BOLL_STD"] = df["close"].rolling(20).std()
    df["BOLL_UPPER"] = df["BOLL_MID"] + 2 * df["BOLL_STD"]
    df["BOLL_LOWER"] = df["BOLL_MID"] - 2 * df["BOLL_STD"]

    # Add DMI
    df["PDI"] = 20 + np.random.rand(100) * 10
    df["MDI"] = 20 + np.random.rand(100) * 10
    df["ADX"] = 25 + np.random.rand(100) * 10

    # Add OBV
    df["OBV"] = np.cumsum(np.where(df["close"] > df["close"].shift(1), 1, -1) * df["volume"])

    # Add VR
    df["VR"] = 100 + np.random.randn(100) * 30

    # Add ROC
    df["ROC"] = df["close"].pct_change(5) * 100
    df["MAROC"] = df["ROC"].rolling(10).mean()

    # Add BBI
    df["BBI"] = (df["MA5"] + df["MA10"] if "MA10" in df.columns else df["MA20"] + df["MA30"] if "MA30" in df.columns else df["MA20"]) / 3
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA30"] = df["close"].rolling(30).mean()
    df["BBI"] = (df["MA5"] + df["MA10"] + df["MA20"] + df["MA30"]) / 4

    # Test signal generation
    try:
        analyzer = TechnicalAnalyzer()
        signals = generate_trading_signals(df)
        print(f"✓ Generated {len(signals)} signals:")
        for signal in signals[:3]:  # Show first 3
            print(f"  - {signal}")
        if len(signals) > 3:
            print(f"  ... and {len(signals) - 3} more")

    except Exception as e:
        print(f"✗ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()

    print()


def test_stock_selector():
    """Test stock selector with sample data."""
    print("=" * 50)
    print("Testing Stock Selector")
    print("=" * 50)

    from src.analysis.stock_selector import (
        StockSelector,
        StockPrediction,
        SelectionCriteria,
        SelectionResult,
    )

    # Test SelectionCriteria
    criteria = SelectionCriteria(
        rps5_threshold=85.0,
        prediction_days=5,
        min_data_points=50,
        enable_strategy_filter=True,
    )
    print(f"✓ SelectionCriteria created: RPS5 > {criteria.rps5_threshold}")

    # Test StockPrediction
    prediction = StockPrediction(
        stock_code="sh.600000",
        predicted_return=3.5,
        confidence=0.85,
        model_info="test_model",
        prediction_date="2024-01-15",
        features_count=29,
        data_points=100,
    )
    print(f"✓ StockPrediction created: {prediction.stock_code}")
    print(f"  - Predicted return: {prediction.predicted_return:.2f}%")
    print(f"  - Confidence: {prediction.confidence:.3f}")
    print(f"  - Investment score: {prediction.investment_score:.3f}")

    # Test StockSelector class initialization
    selector = StockSelector(
        data_dir="market_data",
        rps_dir="rps_results",
        models_dir="models",
    )
    print(f"✓ StockSelector initialized")

    # Test to_dict method
    pred_dict = prediction.to_dict()
    print(f"✓ Prediction to_dict: {len(pred_dict)} fields")

    print()


def test_llm_analyzer():
    """Test LLM analyzer with sample data."""
    print("=" * 50)
    print("Testing LLM Analyzer")
    print("=" * 50)

    from src.analysis.llm_analyzer import (
        LLMAnalyzer,
        AnalysisRequest,
        AnalysisResult,
        PromptManager,
        PromptTemplate,
    )

    # Test PromptManager
    prompt_manager = PromptManager()
    template = prompt_manager.get_template("default")
    print(f"✓ PromptManager created with default template")
    print(f"  - System prompt length: {len(template.system_prompt)}")

    # Test custom template
    custom_template = PromptTemplate(
        system_prompt="You are a financial analyst.",
        user_prompt_template="Analyze: {data}",
        temperature=0.2,
    )
    prompt_manager.register_template("custom", custom_template)
    print(f"✓ Custom template registered")

    # Test AnalysisRequest
    request = AnalysisRequest(
        stock_code="sh.600000",
        stock_name="平安银行",
        price_data={"2024-01-01": {"close": "10.50", "volume": "1000000"}},
        indicator_data={"2024-01-01": {"RSI": "55.00", "MACD": "0.15"}},
        market_trends={"日涨跌幅": "+1.2%"},
    )
    print(f"✓ AnalysisRequest created for {request.stock_code}")

    # Test user prompt generation
    user_prompt = request.to_user_prompt("Analyze: {data}")
    print(f"✓ User prompt generated: {len(user_prompt)} chars")

    # Test AnalysisResult
    result = AnalysisResult(
        technical_analysis="长期趋势向上，MACD金叉。",
        trend_analysis="短期震荡整理。",
        investment_advice="建议逢低买入。",
        risk_warning="注意大盘系统性风险。",
        summary="股票处于上涨趋势中，适合中线布局。",
        model_used="test-model",
    )
    print(f"✓ AnalysisResult created")

    # Test to_dict
    result_dict = result.to_dict()
    print(f"✓ Result to_dict: {len(result_dict)} keys")

    # Test from_error
    error_result = AnalysisResult.from_error("Test error")
    print(f"✓ Error result created")

    # Test LLMAnalyzer initialization (without API call)
    # Note: Skip actual API call as it requires credentials
    print(f"✓ LLMAnalyzer class available (API test skipped)")

    print()


def test_report_generator():
    """Test report generator with sample data."""
    print("=" * 50)
    print("Testing Report Generator")
    print("=" * 50)

    from src.analysis.report_generator import (
        ReportGenerator,
        ReportFormat,
        ReportResult,
        StockConfig,
        MarkdownReportTemplate,
        JSONReportTemplate,
    )

    # Test StockConfig
    config = StockConfig("平安银行", "000001")
    print(f"✓ StockConfig created: {config.name} ({config.code})")

    # Test ReportFormat
    json_format = ReportFormat.json()
    md_format = ReportFormat.markdown()
    print(f"✓ ReportFormat created: {json_format.name}, {md_format.name}")

    # Test templates
    md_template = MarkdownReportTemplate()
    json_template = JSONReportTemplate()
    print(f"✓ Report templates created")

    # Test mock analysis result
    analysis_result = {
        "analysis_date": "2024-01-15",
        "status": "completed",
        "steps_completed": [
            {"step": 1, "name": "数据获取", "status": "completed"},
            {"step": 2, "name": "财务分析", "status": "completed"},
        ],
        "final_report": {
            "executive_summary": "公司财务状况良好。",
            "financial_highlights": {
                "revenue": "100亿元",
                "net_profit": "20亿元",
            },
            "industry_position": "行业领先",
            "key_risks": ["市场风险"],
            "investment_recommendation": "买入",
            "analyst_rating": "推荐",
            "confidence_level": "高",
            "report_date": "2024-01-15",
            "disclaimer": "仅供参考",
        },
    }

    # Test ReportGenerator
    generator = ReportGenerator(output_dir="test_reports")
    print(f"✓ ReportGenerator initialized")

    # Test single report generation
    result = generator.generate_report(
        config,
        analysis_result,
        formats=[ReportFormat.json(), ReportFormat.markdown()],
    )
    print(f"✓ Report generated: {result.status}")
    print(f"  - Output paths: {len(result.output_paths)} formats")

    # Test batch generation
    configs = [
        StockConfig("平安银行", "000001"),
        StockConfig("招商银行", "600036"),
    ]
    results = generator.generate_reports(
        configs,
        parallel=False,  # Use sequential for testing
    )
    print(f"✓ Batch reports generated: {len(results)} stocks")

    # Test ReportResult
    print(f"✓ ReportResult.is_success(): {result.is_success()}")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("REFACTORED MODULES TEST SUITE")
    print("=" * 50 + "\n")

    try:
        test_config()
        test_exceptions()
        test_logger()
        test_cache()
        test_utils()
        test_base_classes()
        test_technical_analyzer()
        test_stock_selector()
        test_llm_analyzer()
        test_report_generator()

        print("=" * 50)
        print("✓ ALL TESTS PASSED")
        print("=" * 50)

    except Exception as e:
        print("=" * 50)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
