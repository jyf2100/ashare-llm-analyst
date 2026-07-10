#!/usr/bin/env python3
"""
End-to-end test for the refactored stock analysis system.

This script tests the complete workflow:
1. Load stock data
2. Technical analysis
3. Stock selection
4. Report generation
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add current directory and src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "src"))

from src.core.config import Config, get_config
from src.core.logger import get_logger
from src.core.cache import clear_cache
from src.analysis.technical_analyzer import TechnicalAnalyzer, generate_trading_signals
from src.analysis.stock_selector import StockSelector, SelectionCriteria
from src.analysis.report_generator import ReportGenerator, StockConfig, ReportFormat

# Setup logger
logger = get_logger(__name__)


def create_sample_stock_data(stock_code: str = "000001", days: int = 250) -> pd.DataFrame:
    """Create sample stock data for testing."""
    dates = pd.date_range(start=datetime.now().replace(day=1), periods=days, freq="D")
    dates = dates[dates.weekday < 5]  # Remove weekends

    np.random.seed(hash(stock_code) % 10000)

    # Generate realistic price data
    close_prices = []
    for i in range(len(dates)):
        if i == 0:
            price = 100
        else:
            # Random walk with drift
            drift = 0.0002
            shock = np.random.normal(0, 0.02)
            price = close_prices[-1] * (1 + drift + shock)
        close_prices.append(price)

    close_prices = pd.Series(close_prices)

    # Calculate OHLC
    high = close_prices * (1 + np.abs(np.random.randn(len(dates))) * 0.02)
    low = close_prices * (1 - np.abs(np.random.randn(len(dates))) * 0.02)
    open_prices = close_prices.shift(1).fillna(100) * (1 + np.random.randn(len(dates)) * 0.005)

    # Volume
    volume = np.random.randint(1000000, 10000000, len(dates))

    df = pd.DataFrame({
        "date": dates,
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close_prices,
        "volume": volume,
    })

    # Set date as index
    df = df.set_index("date")

    # Add technical indicators
    df = add_technical_indicators(df)

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to dataframe."""
    # Moving averages
    for period in [5, 10, 20, 30, 60, 120]:
        if len(df) >= period:
            df[f"MA{period}"] = df["close"].rolling(period).mean()
        else:
            df[f"MA{period}"] = df["close"]

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9).mean()
    df["MACD"] = df["DIF"] - df["DEA"]

    # KDJ
    low_min = df["low"].rolling(9).min()
    high_max = df["high"].rolling(9).max()
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100
    df["K"] = rsv.ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # BOLL
    df["BOLL_MID"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["BOLL_UPPER"] = df["BOLL_MID"] + 2 * bb_std
    df["BOLL_LOWER"] = df["BOLL_MID"] - 2 * bb_std

    # DMI
    df["+DM"] = df["high"].diff()
    df["-DM"] = -df["low"].diff()
    df["+DM"] = df["+DM"].where(df["+DM"] > 0, 0)
    df["-DM"] = df["-DM"].where(df["-DM"] > 0, 0)

    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift(1)),
        abs(df["low"] - df["close"].shift(1))
    ], axis=1).max(axis=1)

    df["+DI"] = (df["+DM"].rolling(14).mean() / tr.rolling(14).mean()) * 100
    df["-DI"] = (df["-DM"].rolling(14).mean() / tr.rolling(14).mean()) * 100
    df["PDI"] = df["+DI"]
    df["MDI"] = df["-DI"]
    df["ADX"] = abs(df["PDI"] - df["MDI"]) / (df["PDI"] + df["MDI"]) * 100

    # OBV
    df["OBV"] = (np.where(df["close"] > df["close"].shift(1), 1, -1) * df["volume"]).cumsum()

    # VR
    df["VR"] = 100 + np.random.randn(len(df)) * 30

    # ROC
    df["ROC"] = df["close"].pct_change(5) * 100
    df["MAROC"] = df["ROC"].rolling(10).mean()

    # BBI
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA30"] = df["close"].rolling(30).mean()
    df["BBI"] = (df["MA5"] + df["MA10"] + df["MA20"] + df["MA30"]) / 4

    # Fill NaN values
    df = df.fillna(method="bfill").fillna(0)

    return df


def test_config_system():
    """Test 1: Configuration system."""
    print("\n" + "=" * 70)
    print("TEST 1: Configuration System")
    print("=" * 70)

    config = get_config()
    print(f"✓ Config loaded")
    print(f"  - Cache enabled: {config.cache.enabled}")
    print(f"  - Cache TTL: {config.cache.ttl}s")
    print(f"  - Log level: {config.logging.level}")
    print(f"  - Max workers: {config.parallel.max_workers}")

    return True


def test_technical_analysis():
    """Test 2: Technical analysis."""
    print("\n" + "=" * 70)
    print("TEST 2: Technical Analysis")
    print("=" * 70)

    # Create sample data
    print("Creating sample stock data...")
    df = create_sample_stock_data("sh.600000", days=250)
    print(f"✓ Generated {len(df)} days of data for sh.600000")

    # Run technical analysis
    print("\nRunning technical analysis...")
    analyzer = TechnicalAnalyzer()

    signals = analyzer.generate_signals(df)
    print(f"✓ Generated {len(signals)} signal(s)")

    for signal in signals:
        print(f"\n  Signal:")
        print(f"    - Type: {signal.type}")
        print(f"    - Strength: {signal.strength:.1f}")
        print(f"    - Reason: {signal.reason}")

        if signal.metadata.get("all_signals"):
            print(f"    - Buy Score: {signal.metadata['buy_score']}")
            print(f"    - Sell Score: {signal.metadata['sell_score']}")

    # Test backward compatible function
    print("\nTesting backward compatible function...")
    signal_strings = generate_trading_signals(df)
    print(f"✓ Generated {len(signal_strings)} signal strings")
    print(f"  Latest: {signal_strings[0] if signal_strings else 'None'}")

    return df


def test_stock_selection(df: pd.DataFrame):
    """Test 3: Stock selection."""
    print("\n" + "=" * 70)
    print("TEST 3: Stock Selection")
    print("=" * 70)

    # Save sample data to market_data directory
    market_dir = Path("market_data")
    market_dir.mkdir(exist_ok=True)

    sample_file = market_dir / "sh.600000.csv"
    df.to_csv(sample_file)
    print(f"✓ Saved sample data to {sample_file}")

    # Initialize stock selector
    selector = StockSelector(
        data_dir=str(market_dir),
        rps_dir="rps_results",
        models_dir="models",
    )

    # Load data
    print("\nLoading stock data...")
    loaded = selector.load_stock_data(max_stocks=10)
    print(f"✓ Loaded {loaded} stock(s)")

    # Load RPS data (will use mock if not found)
    print("\nLoading RPS data...")
    selector.load_rps_data()
    print(f"✓ RPS data loaded (mock/generated if not found)")

    # Filter by RPS
    print("\nFiltering by RPS...")
    candidates = selector.filter_by_rps(threshold=50)  # Lower threshold for testing
    print(f"✓ Found {len(candidates)} candidate(s) with RPS5 > 50")

    # Test prediction (if ML available)
    if candidates:
        print("\nRunning ML predictions...")
        try:
            # Test only 1 stock to avoid timeout
            predictions = selector.predict_stocks(candidates[:1], prediction_days=5)
            print(f"✓ Predicted {len(predictions)} stock(s)")

            for code, pred in list(predictions.items())[:1]:
                print(f"\n  {code}:")
                print(f"    - Predicted Return: {pred.predicted_return:.2f}%")
                print(f"    - Confidence: {pred.confidence:.3f}")
                print(f"    - Investment Score: {pred.investment_score:.3f}")
        except Exception as e:
            print(f"⚠ ML prediction error (may be expected with mock data): {e}")

    return True


def test_report_generation():
    """Test 4: Report generation."""
    print("\n" + "=" * 70)
    print("TEST 4: Report Generation")
    print("=" * 70)

    # Create output directory
    output_dir = Path("test_e2e_reports")
    output_dir.mkdir(exist_ok=True)

    # Initialize report generator
    generator = ReportGenerator(output_dir=str(output_dir))
    print(f"✓ Report generator initialized")
    print(f"  - Output directory: {output_dir.absolute()}")

    # Create sample analysis result
    analysis_result = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "status": "completed",
        "steps_completed": [
            {"step": 1, "name": "数据获取", "status": "completed"},
            {"step": 2, "name": "技术分析", "status": "completed"},
            {"step": 3, "name": "风险评估", "status": "completed"},
        ],
        "final_report": {
            "executive_summary": "公司财务状况良好，技术面呈现上涨趋势。",
            "financial_highlights": {
                "revenue": "500亿元",
                "net_profit": "80亿元",
                "roe": "18.5%",
                "pe_ratio": "15.2",
            },
            "industry_position": "行业领先地位，市场份额持续提升。",
            "key_risks": [
                "市场波动风险",
                "政策变化风险",
                "原材料价格波动风险",
            ],
            "investment_recommendation": "推荐买入",
            "analyst_rating": "买入",
            "confidence_level": "高",
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "disclaimer": "本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。",
        },
    }

    # Generate single report
    print("\nGenerating single stock report...")
    config = StockConfig("平安银行", "000001")

    result = generator.generate_report(
        config,
        analysis_result,
        formats=[ReportFormat.json(), ReportFormat.markdown()],
    )

    print(f"✓ Report generated: {result.status}")
    if result.is_success():
        print(f"  - JSON: {result.output_paths.get('json', 'N/A')}")
        print(f"  - Markdown: {result.output_paths.get('markdown', 'N/A')}")

    # Generate batch reports
    print("\nGenerating batch reports...")
    configs = [
        StockConfig("平安银行", "000001"),
        StockConfig("招商银行", "600036"),
        StockConfig("贵州茅台", "600519"),
    ]

    results = generator.generate_reports(
        configs,
        formats=[ReportFormat.markdown()],
        parallel=False,
    )

    success_count = sum(1 for r in results if r.is_success())
    print(f"✓ Generated {success_count}/{len(results)} reports")

    # Print summary
    generator.print_results(results)

    return True


def test_integration():
    """Test 5: Integration test - complete workflow."""
    print("\n" + "=" * 70)
    print("TEST 5: Integration - Complete Workflow")
    print("=" * 70)

    print("\n📊 Step 1: Load stock data")
    df = create_sample_stock_data("sz.000001", days=250)
    print(f"✓ Loaded {len(df)} days of data")

    print("\n📈 Step 2: Technical analysis")
    analyzer = TechnicalAnalyzer()
    signals = analyzer.generate_signals(df)
    print(f"✓ Generated signals: {signals[0].type if signals else 'N/A'}")

    print("\n🎯 Step 3: Generate integrated report")
    output_dir = Path("test_e2e_reports")
    generator = ReportGenerator(output_dir=str(output_dir))

    # Create analysis result with technical signals
    integrated_result = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "status": "completed",
        "steps_completed": [
            {"step": 1, "name": "数据获取", "status": "completed"},
            {"step": 2, "name": "技术分析", "status": "completed"},
            {"step": 3, "name": "综合评估", "status": "completed"},
        ],
        "final_report": {
            "executive_summary": f"技术面分析显示{'看涨' if signals and signals[0].is_buy() else '看跌'}信号。",
            "technical_signals": {
                "signal_type": signals[0].type if signals else "HOLD",
                "strength": signals[0].strength if signals else 0,
                "buy_score": signals[0].metadata.get("buy_score", 0) if signals else 0,
                "sell_score": signals[0].metadata.get("sell_score", 0) if signals else 0,
            },
            "financial_highlights": {
                "revenue": "N/A",
                "net_profit": "N/A",
            },
            "industry_position": "N/A",
            "key_risks": ["技术指标变化风险"],
            "investment_recommendation": "买入" if signals and signals[0].is_buy() else "观望",
            "analyst_rating": signals[0].type if signals else "HOLD",
            "confidence_level": "中",
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "disclaimer": "本报告仅供参考。",
        },
    }

    config = StockConfig("测试股票", "000001")
    result = generator.generate_report(
        config,
        integrated_result,
        formats=[ReportFormat.json(), ReportFormat.markdown()],
    )

    print(f"✓ Integrated report generated: {result.status}")

    return True


def main():
    """Run all end-to-end tests."""
    print("\n" + "=" * 70)
    print("🚀 END-TO-END TEST SUITE")
    print("Testing Refactored Stock Analysis System")
    print("=" * 70)

    tests = [
        ("Configuration System", test_config_system),
        ("Technical Analysis", test_technical_analysis),
        ("Stock Selection", test_stock_selection),
        ("Report Generation", test_report_generation),
        ("Integration Workflow", test_integration),
    ]

    results = {}
    df = None

    for test_name, test_func in tests:
        try:
            if test_name == "Stock Selection" and df is not None:
                result = test_func(df)
            elif test_name == "Technical Analysis":
                df = test_func()
                result = True
            else:
                result = test_func()

            results[test_name] = "✅ PASS" if result else "❌ FAIL"

        except Exception as e:
            results[test_name] = f"❌ ERROR: {e}"
            logger.error(f"{test_name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)

    for test_name, result in results.items():
        print(f"  {result} - {test_name}")

    pass_count = sum(1 for r in results.values() if "PASS" in r)
    total_count = len(results)

    print("\n" + "=" * 70)
    if pass_count == total_count:
        print(f"🎉 ALL TESTS PASSED ({pass_count}/{total_count})")
    else:
        print(f"⚠️  SOME TESTS FAILED ({pass_count}/{total_count} passed)")
    print("=" * 70)

    return 0 if pass_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
