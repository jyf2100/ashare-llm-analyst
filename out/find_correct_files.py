#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

csv_files = list(Path('./market_data').glob('*.csv'))
correct_files = []

for f in csv_files:
    try:
        df = pd.read_csv(f, usecols=['date'])
        if not df.empty and df.iloc[-1]['date'] == '2025-08-05':
            correct_files.append(f.name)
    except Exception as e:
        print(f"Error reading {f.name}: {e}")

print(f'最后日期为2025-08-05的文件数量: {len(correct_files)}')
print(f'最后日期为2025-08-05的文件: {correct_files}')