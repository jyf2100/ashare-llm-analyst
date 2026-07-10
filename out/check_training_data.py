#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查训练数据中的RPS质量
"""

import pandas as pd
import glob

# 找到最新的训练数据文件
files = glob.glob('training_data/training_data_*.csv')
latest_file = max(files)
df = pd.read_csv(latest_file)

print(f'训练数据文件: {latest_file}')
print(f'总股票数: {df["stock_code"].nunique()}')
print(f'总样本数: {len(df)}')
print()

print('前10只股票的RPS120覆盖率:')
for stock in df["stock_code"].unique()[:10]:
    stock_data = df[df["stock_code"]==stock]
    non_default_rate = (stock_data["rps120"] != 50.0).mean()
    print(f'  {stock}: {non_default_rate:.1%}')

print()
print('整体RPS特征统计:')
for col in ["rps20", "rps60", "rps120"]:
    non_default_rate = (df[col] != 50.0).mean()
    print(f'  {col}: 非默认值比例 {non_default_rate:.1%}')

print()
print('检查是否所有股票的RPS120都是100%覆盖率:')
for stock in df["stock_code"].unique():
    stock_data = df[df["stock_code"]==stock]
    rps120_coverage = (stock_data["rps120"] != 50.0).mean()
    if rps120_coverage != 1.0:
        print(f'  {stock}: RPS120覆盖率 {rps120_coverage:.1%} (不是100%)')
        break
else:
    print('  所有股票的RPS120覆盖率都是100%')