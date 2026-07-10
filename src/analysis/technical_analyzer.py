"""
Technical analyzer module - refactored with unified infrastructure.

Provides technical analysis and signal generation using the new core infrastructure.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from src.core.base import AnalyzerBase, Signal
from src.core.cache import cached, CacheConfig
from src.core.config import Config, get_config
from src.core.exceptions import AnalysisError, ValidationError
from src.core.logger import get_logger

logger = get_logger(__name__)


class TechnicalAnalyzer(AnalyzerBase):
    """
    Technical analyzer for stock price data.

    Provides:
    - Trading signal generation
    - Technical indicator analysis
    - Multi-timeframe analysis
    - Cached calculations

    Example:
        analyzer = TechnicalAnalyzer()
        signals = analyzer.generate_signals(df)
        for signal in signals:
            print(f"{signal.type}: {signal.reason}")
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize technical analyzer."""
        super().__init__(config)
        self._indicators_cache = {}

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate input dataframe.

        Args:
            df: Input dataframe

        Raises:
            ValidationError: If data is invalid
        """
        if df is None or df.empty:
            raise ValidationError(
                "Dataframe is empty or None",
                field_name="df"
            )

        required_columns = ["open", "high", "low", "close"]
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValidationError(
                f"Missing required columns: {', '.join(missing)}",
                field_name="df.columns"
            )

        if len(df) < 3:
            raise ValidationError(
                f"Insufficient data: {len(df)} rows (minimum 3)",
                field_name="df"
            )

    @cached(CacheConfig(ttl=600))
    def calculate_signals(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Calculate buy/sell scores based on technical indicators.

        Args:
            df: Input dataframe with OHLCV data

        Returns:
            Dictionary with buy_score, sell_score, and signals list
        """
        self.validate_data(df)

        buy_score = 0
        sell_score = 0
        signals = []

        # Get latest data points
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        # 1. MACD Analysis
        macd_signals = self._analyze_macd(df, latest, prev, prev2)
        signals.extend(macd_signals["signals"])
        buy_score += macd_signals["buy_score"]
        sell_score += macd_signals["sell_score"]

        # 2. KDJ Analysis
        kdj_signals = self._analyze_kdj(df, latest, prev)
        signals.extend(kdj_signals["signals"])
        buy_score += kdj_signals["buy_score"]
        sell_score += kdj_signals["sell_score"]

        # 3. RSI Analysis
        rsi_signals = self._analyze_rsi(df, latest)
        signals.extend(rsi_signals["signals"])
        buy_score += rsi_signals["buy_score"]
        sell_score += rsi_signals["sell_score"]

        # 4. BOLL Analysis
        boll_signals = self._analyze_boll(df, latest, prev)
        signals.extend(boll_signals["signals"])
        buy_score += boll_signals["buy_score"]
        sell_score += boll_signals["sell_score"]

        # 5. DMI Analysis
        dmi_signals = self._analyze_dmi(df, latest)
        signals.extend(dmi_signals["signals"])
        buy_score += dmi_signals["buy_score"]
        sell_score += dmi_signals["sell_score"]

        # 6. OBV Analysis (if available)
        if "OBV" in df.columns:
            obv_signals = self._analyze_obv(df, latest, prev, prev2)
            signals.extend(obv_signals["signals"])
            buy_score += obv_signals["buy_score"]
            sell_score += obv_signals["sell_score"]

        # 7. VR Analysis (if available)
        if "VR" in df.columns:
            vr_signals = self._analyze_vr(df, latest)
            signals.extend(vr_signals["signals"])
            buy_score += vr_signals["buy_score"]
            sell_score += vr_signals["sell_score"]

        # 8. ROC Analysis (if available)
        if "ROC" in df.columns and "MAROC" in df.columns:
            roc_signals = self._analyze_roc(df, latest, prev)
            signals.extend(roc_signals["signals"])
            buy_score += roc_signals["buy_score"]
            sell_score += roc_signals["sell_score"]

        # 9. BBI Analysis (if available)
        if "BBI" in df.columns:
            bbi_signals = self._analyze_bbi(df, latest, prev)
            signals.extend(bbi_signals["signals"])
            buy_score += bbi_signals["buy_score"]
            sell_score += bbi_signals["sell_score"]

        # Calculate net score
        net_score = buy_score - sell_score

        # Add overall assessment
        overall = self._get_overward_assessment(net_score)
        signals.insert(0, overall)

        # Add score summary
        signals.append(
            f"📊 信号评分：买入 {buy_score} 分，卖出 {sell_score} 分，净值 {net_score} 分"
        )

        return {
            "buy_score": buy_score,
            "sell_score": sell_score,
            "net_score": net_score,
            "signals": signals,
        }

    def _analyze_macd(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        prev: pd.Series,
        prev2: pd.Series,
    ) -> dict:
        """Analyze MACD indicator."""
        signals = []
        buy_score = 0
        sell_score = 0

        dif_col = "DIF" if "DIF" in df.columns else "DIF_DMA"
        dea_col = "DEA" if "DEA" in df.columns else "DIFMA_DMA"

        if dif_col in df.columns and dea_col in df.columns:
            # Golden cross
            if latest[dif_col] > latest[dea_col] and prev[dif_col] <= prev[dea_col]:
                if latest["MACD"] > 0:
                    signals.append("🔥 MACD零轴上方金叉，强烈买入信号")
                    buy_score += 3
                else:
                    signals.append("📈 MACD金叉，买入信号")
                    buy_score += 2
            # Death cross
            elif latest[dif_col] < latest[dea_col] and prev[dif_col] >= prev[dea_col]:
                if latest["MACD"] < 0:
                    signals.append("🔻 MACD零轴下方死叉，强烈卖出信号")
                    sell_score += 3
                else:
                    signals.append("📉 MACD死叉，卖出信号")
                    sell_score += 2

        # MACD histogram
        if latest["MACD"] > prev["MACD"] > prev2["MACD"]:
            signals.append("📊 MACD柱状图连续放大，动能增强")
            buy_score += 1

        return {"signals": signals, "buy_score": buy_score, "sell_score": sell_score}

    def _analyze_kdj(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        prev: pd.Series,
    ) -> dict:
        """Analyze KDJ indicator."""
        signals = []
        buy_score = 0
        sell_score = 0

        # Low golden cross
        if (
            latest["K"] > latest["D"]
            and prev["K"] <= prev["D"]
            and latest["K"] < 30
            and latest["D"] < 30
        ):
            signals.append("🚀 KDJ低位金叉，超跌反弹信号")
            buy_score += 3
        # High death cross
        elif (
            latest["K"] < latest["D"]
            and prev["K"] >= prev["D"]
            and latest["K"] > 70
            and latest["D"] > 70
        ):
            signals.append("⚠️ KDJ高位死叉，超买回调信号")
            sell_score += 3

        # J value extremes
        if latest["J"] < 10:
            signals.append("💎 J值极度超卖，关注抄底机会")
            buy_score += 2
        elif latest["J"] > 90:
            signals.append("🔥 J值极度超买，注意风险")
            sell_score += 2

        return {"signals": signals, "buy_score": buy_score, "sell_score": sell_score}

    def _analyze_rsi(self, df: pd.DataFrame, latest: pd.Series) -> dict:
        """Analyze RSI indicator."""
        signals = []
        buy_score = 0
        sell_score = 0

        rsi_val = latest["RSI"]

        if rsi_val < 20:
            signals.append("💪 RSI严重超卖，强烈反弹信号")
            buy_score += 3
        elif rsi_val < 30:
            signals.append("📈 RSI超卖，关注买入机会")
            buy_score += 2
        elif rsi_val > 80:
            signals.append("⚡ RSI严重超买，强烈回调信号")
            sell_score += 3
        elif rsi_val > 70:
            signals.append("📉 RSI超买，关注卖出机会")
            sell_score += 2

        return {"signals": signals, "buy_score": buy_score, "sell_score": sell_score}

    def _analyze_boll(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        prev: pd.Series,
    ) -> dict:
        """Analyze Bollinger Bands."""
        signals = []
        buy_score = 0
        sell_score = 0

        close_price = latest["close"]
        prev_close = prev["close"]

        # Break upper band
        if close_price > latest["BOLL_UPPER"] and prev_close <= prev["BOLL_UPPER"]:
            signals.append("🚀 突破布林带上轨，强势上涨")
            buy_score += 2
        # Break lower band
        elif close_price < latest["BOLL_LOWER"] and prev_close >= prev["BOLL_LOWER"]:
            signals.append("💎 跌破布林带下轨，超跌反弹机会")
            buy_score += 2
        # Return to mid
        elif abs(close_price - latest["BOLL_MID"]) / latest["BOLL_MID"] < 0.01:
            signals.append("⚖️ 价格回归布林带中轨，趋势可能转换")

        return {"signals": signals, "buy_score": buy_score, "sell_score": sell_score}

    def _analyze_dmi(self, df: pd.DataFrame, latest: pd.Series) -> dict:
        """Analyze DMI indicator."""
        signals = []
        buy_score = 0
        sell_score = 0

        adx_val = latest["ADX"]

        if latest["PDI"] > latest["MDI"] and adx_val > 30:
            signals.append(f"📈 DMI多头趋势强劲 (ADX: {adx_val:.1f})")
            buy_score += 2
        elif latest["MDI"] > latest["PDI"] and adx_val > 30:
            signals.append(f"📉 DMI空头趋势强劲 (ADX: {adx_val:.1f})")
            sell_score += 2
        elif adx_val < 20:
            signals.append("😴 ADX显示趋势较弱，市场盘整")

        return {"signals": signals, "buy_score": buy_score, "sell_score": sell_score}

    def _analyze_obv(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        prev: pd.Series,
        prev2: pd.Series,
    ) -> dict:
        """Analyze OBV indicator."""
        signals = []
        buy_score = 0
        sell_score = 0

        if latest["OBV"] > prev["OBV"] > prev2["OBV"]:
            signals.append("📊 OBV连续上升，资金持续流入")
            buy_score += 1
        elif latest["OBV"] < prev["OBV"] < prev2["OBV"]:
            signals.append("📊 OBV连续下降，资金持续流出")
            sell_score += 1

        return {"signals": signals, "buy_score": buy_score, "sell_score": sell_score}

    def _analyze_vr(self, df: pd.DataFrame, latest: pd.Series) -> dict:
        """Analyze VR indicator."""
        signals = []
        buy_score = 0
        sell_score = 0

        vr_val = latest["VR"]

        if vr_val > 400:
            signals.append(f"🔥 VR过高 ({vr_val:.1f})，市场过热风险")
            sell_score += 2
        elif vr_val < 50:
            signals.append(f"💎 VR过低 ({vr_val:.1f})，市场低迷机会")
            buy_score += 2
        elif 80 <= vr_val <= 120:
            signals.append(f"⚖️ VR正常范围 ({vr_val:.1f})，市场健康")

        return {"signals": signals, "buy_score": buy_score, "sell_score": sell_score}

    def _analyze_roc(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        prev: pd.Series,
    ) -> dict:
        """Analyze ROC indicator."""
        signals = []
        buy_score = 0
        sell_score = 0

        if latest["ROC"] > latest["MAROC"] and prev["ROC"] <= prev["MAROC"]:
            signals.append("🚀 ROC上穿均线，动能转强")
            buy_score += 2
        elif latest["ROC"] < latest["MAROC"] and prev["ROC"] >= prev["MAROC"]:
            signals.append("📉 ROC下穿均线，动能转弱")
            sell_score += 2

        return {"signals": signals, "buy_score": buy_score, "sell_score": sell_score}

    def _analyze_bbi(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        prev: pd.Series,
    ) -> dict:
        """Analyze BBI indicator."""
        signals = []
        buy_score = 0
        sell_score = 0

        if latest["close"] > latest["BBI"] and prev["close"] <= prev["BBI"]:
            signals.append("📈 价格突破多空线，多头占优")
            buy_score += 1
        elif latest["close"] < latest["BBI"] and prev["close"] >= prev["BBI"]:
            signals.append("📉 价格跌破多空线，空头占优")
            sell_score += 1

        return {"signals": signals, "buy_score": buy_score, "sell_score": sell_score}

    def _get_overward_assessment(self, net_score: int) -> str:
        """Get overall assessment based on net score."""
        if net_score >= 5:
            return "🔥🔥🔥 综合评估：强烈买入信号！"
        elif net_score >= 3:
            return "🔥🔥 综合评估：买入信号"
        elif net_score >= 1:
            return "📈 综合评估：偏多信号"
        elif net_score <= -5:
            return "🔻🔻🔻 综合评估：强烈卖出信号！"
        elif net_score <= -3:
            return "🔻🔻 综合评估：卖出信号"
        elif net_score <= -1:
            return "📉 综合评估：偏空信号"
        else:
            return "⚖️ 综合评估：震荡整理，观望为主"

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals from dataframe.

        Args:
            df: Input dataframe with OHLCV and indicators

        Returns:
            List of Signal objects

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            result = self.calculate_signals(df)

            # Determine signal type
            net_score = result["net_score"]
            if net_score >= 3:
                signal_type = Signal.BUY
            elif net_score <= -3:
                signal_type = Signal.SELL
            else:
                signal_type = Signal.HOLD

            # Create signal
            signal = Signal(
                signal_type=signal_type,
                strength=min(abs(net_score) * 10, 100),
                reason=result["signals"][0],  # Overall assessment
                metadata={
                    "buy_score": result["buy_score"],
                    "sell_score": result["sell_score"],
                    "net_score": net_score,
                    "all_signals": result["signals"],
                },
            )

            return [signal]

        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            raise AnalysisError(f"Failed to generate signals: {e}")


# Convenience function for backward compatibility
def generate_trading_signals(df: pd.DataFrame) -> List[str]:
    """
    Generate trading signals (backward compatible function).

    Args:
        df: Input dataframe

    Returns:
        List of signal strings
    """
    try:
        analyzer = TechnicalAnalyzer()
        result = analyzer.calculate_signals(df)
        return result["signals"]
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return [f"技术分析计算出错: {e}"]
