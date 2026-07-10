"""
分层回测引擎（诚实回测，含A股交易摩擦）

依据 quant-ml-code-audit skill 反模式6（零真实回测）。
给定历史每日预测分值，做分层回测，输出可信指标：
年化收益 / 夏普 / 最大回撤 / 胜率 / 分层单调性。
含交易成本、T+1（次日开盘）、涨跌停过滤、停牌过滤。

局限（B2-最小）：用当前模型对历史特征预测（pseudo，非 walk-forward），
回测偏乐观。严格 walk-forward 留 B2-完整。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """回测结果。"""
    equity_curve: pd.Series
    annual_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    layer_returns: Dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        mono = self._layer_monotonic()
        return (
            f"年化: {self.annual_return:.2%} | 夏普: {self.sharpe:.2f} | "
            f"最大回撤: {self.max_drawdown:.2%} | 胜率: {self.win_rate:.2%} | "
            f"交易: {self.n_trades} | 分层单调: {mono}"
        )

    def _layer_monotonic(self) -> str:
        if not self.layer_returns:
            return "N/A"
        layers = sorted(self.layer_returns.keys())
        rets = [self.layer_returns[l] for l in layers]
        is_mono = all(rets[i] <= rets[i + 1] for i in range(len(rets) - 1))
        return "是(最高档收益最高,信号有效)" if is_mono else "否(信号无效或不足)"


class LayeredBacktester:
    """
    分层回测引擎（A股，含交易摩擦）。

    Args:
        cost_bps: 双边交易成本(基点)。A股默认15(0.15%)。
        n_layers: 分层数。默认5。
        hold_days: 持有/调仓周期(交易日)。默认5。
        limit_pct: 涨跌停判定阈值(%)。主板9.5。
    """

    def __init__(self, cost_bps: float = 15, n_layers: int = 5,
                 hold_days: int = 5, limit_pct: float = 9.5):
        self.cost_bps = cost_bps
        self.cost_single = cost_bps / 2 / 10000.0  # 单边成本(小数)
        self.n_layers = n_layers
        self.hold_days = hold_days
        self.limit_pct = limit_pct

    def run(self, predictions: pd.DataFrame,
            price_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        运行分层回测。

        Args:
            predictions: DataFrame[date, stock_code, score]
            price_data: dict[stock_code -> DataFrame]，每个含
                        [date, open, high, low, close, preclose, pctChg, tradestatus]
        Returns:
            BacktestResult
        """
        predictions = predictions.copy()
        predictions["date"] = pd.to_datetime(predictions["date"])
        all_dates = sorted(predictions["date"].unique())
        rebalance_dates = all_dates[:: self.hold_days]

        if len(rebalance_dates) < 2:
            return BacktestResult(pd.Series([1.0]), 0.0, 0.0, 0.0, 0.0, 0, {})

        # 预处理价格：建日期索引
        px: Dict[str, pd.DataFrame] = {}
        for code, df in price_data.items():
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"])
            px[code] = d.set_index("date").sort_index()

        layer_equities: Dict[int, List[float]] = {l: [1.0] for l in range(self.n_layers)}
        eq_dates: List = [rebalance_dates[0]]
        trades: List[dict] = []

        for i in range(len(rebalance_dates) - 1):
            rb_date = rebalance_dates[i]
            next_rb = rebalance_dates[i + 1]
            day_preds = predictions[predictions["date"] == rb_date].sort_values("score", ascending=False)

            if len(day_preds) < self.n_layers:
                for l in range(self.n_layers):
                    layer_equities[l].append(layer_equities[l][-1])
                eq_dates.append(next_rb)
                continue

            try:
                day_preds = day_preds.assign(
                    layer=pd.qcut(day_preds["score"], self.n_layers, labels=False, duplicates="drop")
                )
            except Exception:
                day_preds = day_preds.assign(layer=0)

            for l in range(self.n_layers):
                stocks = day_preds[day_preds["layer"] == l]["stock_code"].tolist()
                ret, layer_trades = self._layer_return(stocks, rb_date, next_rb, px)
                layer_equities[l].append(layer_equities[l][-1] * (1 + ret))
                if l == self.n_layers - 1:
                    trades.extend(layer_trades)
            eq_dates.append(next_rb)

        top = self.n_layers - 1
        eq_len = len(layer_equities[top])
        equity = pd.Series(layer_equities[top], index=eq_dates[:eq_len])
        layer_returns = {l: layer_equities[l][-1] - 1 for l in range(self.n_layers)}
        return self._metrics(equity, trades, layer_returns)

    def _layer_return(self, stocks: List[str], entry_date, exit_date,
                      px: Dict[str, pd.DataFrame]) -> Tuple[float, List[dict]]:
        """一组股票 entry_date 次日开盘买入 → exit_date 次日开盘卖出（T+1，含成本，涨跌停/停牌过滤）。"""
        rets: List[float] = []
        trades: List[dict] = []
        for stock in stocks:
            df = px.get(stock)
            if df is None or entry_date not in df.index or exit_date not in df.index:
                continue
            ei = df.index.get_loc(entry_date)
            xi = df.index.get_loc(exit_date)
            if ei + 1 >= len(df) or xi + 1 >= len(df):
                continue
            buy_row = df.iloc[ei + 1]
            sell_row = df.iloc[xi + 1]
            # 停牌过滤
            if buy_row.get("tradestatus", 1) != 1 or sell_row.get("tradestatus", 1) != 1:
                continue
            # 涨跌停过滤
            if self._is_limit_up(buy_row):
                continue
            if self._is_limit_down(sell_row):
                continue
            buy_price = buy_row["open"]
            sell_price = sell_row["open"]
            if buy_price <= 0:
                continue
            gross = sell_price / buy_price - 1
            net = gross - 2 * self.cost_single  # 买卖各扣单边成本
            rets.append(net)
            trades.append({"stock": stock, "buy": float(buy_price),
                           "sell": float(sell_price), "ret": float(net)})
        return (float(np.mean(rets)) if rets else 0.0), trades

    def _is_limit_up(self, row) -> bool:
        """一字涨停：开盘≈最高(全天封板) 且 涨幅≥阈值 → 买不进。"""
        pct = row.get("pctChg", 0)
        return row["open"] >= row["high"] * 0.999 and pct >= self.limit_pct

    def _is_limit_down(self, row) -> bool:
        """一字跌停：开盘≈最低 且 跌幅≥阈值 → 卖不出。"""
        pct = row.get("pctChg", 0)
        return row["open"] <= row["low"] * 1.001 and pct <= -self.limit_pct

    def _metrics(self, equity: pd.Series, trades: List[dict],
                 layer_returns: Dict[int, float]) -> BacktestResult:
        if len(equity) < 2:
            return BacktestResult(equity, 0.0, 0.0, 0.0, 0.0, 0, layer_returns)
        total_ret = equity.iloc[-1] - 1
        days = max((equity.index[-1] - equity.index[0]).days, 1)
        annual = (1 + total_ret) ** (365 / days) - 1
        daily = equity.pct_change().dropna()
        sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
        max_dd = float(((equity / equity.cummax()) - 1).min())
        win = float(np.mean([t["ret"] > 0 for t in trades])) if trades else 0.0
        return BacktestResult(equity, float(annual), float(sharpe),
                              max_dd, win, len(trades), layer_returns)
