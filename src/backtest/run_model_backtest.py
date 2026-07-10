"""
真模型分层回测（B2-完整·真模型回测）

用 B1 修复后的特征（无 bfill/无 close/OBV/标签 valid_mask）训练随机森林，
按日期时序切分（训练可信），对样本外预测，分层回测验证 ML 信号是否有效。

局限：模型用全样本前 70% 训练后固定，对后 30% 预测（pseudo walk-forward，
非每调仓日重训）。比 B2-最小 的动量 score 真实得多；严格 walk-forward 留后续。

用法:
    conda run -n ashare-llm-analyst python src/backtest/run_model_backtest.py
可选环境变量: N_STOCKS(默认150) HORIZON(默认5) THRESHOLD(默认0.05)
"""
import sys
import glob
import re
import os
import warnings

sys.path.insert(0, "/mnt/disk01/workspaces/worksummary/ashare-llm-analyst")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from src.data.feature_engineering import FeatureCalculator
from src.backtest.layered_backtester import LayeredBacktester

N = int(os.environ.get("N_STOCKS", 150))
H = int(os.environ.get("HORIZON", 5))
THR = float(os.environ.get("THRESHOLD", 0.05))

# B1 后特征列模式（无 close/OBV）
PATTERNS = [r"^MA\d+$", r"^DIF$", r"^DEA$", r"^MACD$", r"^K$", r"^D_$", r"^J$",
            r"^RSI$", r"^BOLL_UP$", r"^BOLL_MID$", r"^BOLL_LOW$", r"^PDI$", r"^MDI$",
            r"^ADX$", r"^ROC$", r"^MAROC$", r"^BBI$"]


def main():
    files = sorted(glob.glob("market_data/*.csv"))[:N]
    fc = FeatureCalculator()
    parts = []
    prices = {}
    for f in files:
        code = f.split("/")[-1].replace(".csv", "")
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        prices[code] = df
        feats = fc.calculate_all_features(df.copy())
        feats["future"] = feats["close"].shift(-H) / feats["close"] - 1
        mask = (feats["close"].shift(-H) / feats["close"] - 1).notna()  # valid_mask
        cols = []
        for p in PATTERNS:
            cols += [c for c in feats.columns if re.match(p, c)]
        cols = list(dict.fromkeys(cols))
        sub = feats.loc[mask, ["date"] + cols + ["future"]].copy()
        sub["stock_code"] = code
        sub["label"] = (sub["future"] > THR).astype(int)
        parts.append(sub)

    all_df = pd.concat(parts, ignore_index=True).dropna(subset=cols)
    X = np.nan_to_num(all_df[cols].values.astype(float))
    dates = all_df["date"].values
    stocks = all_df["stock_code"].values
    y = all_df["label"].values
    print(f"股票: {len(files)} | 样本: {len(X)} | 特征: {X.shape[1]} | 正样本率: {y.mean():.2%}")

    # 时序切分训练（按日期，训练可信）
    order = np.argsort(dates)
    Xo, yo, do, so = X[order], y[order], dates[order], stocks[order]
    split = int(len(Xo) * 0.7)
    scaler = RobustScaler().fit(Xo[:split])  # scaler 仅训练集 fit（B1 修复）
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=50,
                                 random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(scaler.transform(Xo[:split]), yo[:split])
    scores = clf.predict_proba(scaler.transform(Xo))[:, 1]

    # 只回测样本外（后 30% 日期）
    test_start = pd.Timestamp(do[split])
    predictions = pd.DataFrame({"date": pd.to_datetime(do), "stock_code": so, "score": scores})
    predictions_test = predictions[predictions["date"] >= test_start]
    print(f"训练期止: {do[split-1]} | 样本外回测预测: {len(predictions_test)} 起: {test_start.date()}")

    bt = LayeredBacktester(cost_bps=15, n_layers=5, hold_days=H)
    result = bt.run(predictions_test, prices)
    print("=" * 60)
    print("真模型分层回测（B1 后特征 + 时序切分 + 样本外）")
    print("=" * 60)
    print(result.summary())
    print("各层收益:", {k: f"{v:.2%}" for k, v in sorted(result.layer_returns.items())})
    print("解读: 分层单调(最高档收益最高)=ML信号有效；否=信号无效或不足")


if __name__ == "__main__":
    main()
