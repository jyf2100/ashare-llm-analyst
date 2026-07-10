#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看所有CSV文件的最后日期
"""

import os
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict

def get_last_date_from_csv(csv_file):
    """
    获取CSV文件的最后一行日期
    
    Args:
        csv_file (str): CSV文件路径
        
    Returns:
        str: 最后一行的日期，如果出错返回None
    """
    try:
        # 读取CSV文件的最后一行
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            return None
        
        # 获取最后一行的日期
        last_date = df.iloc[-1]['date']
        return str(last_date)
    except Exception as e:
        print(f"读取文件 {csv_file} 出错: {e}")
        return None

def analyze_csv_dates(csv_dir):
    """
    分析CSV目录下所有文件的最后日期
    
    Args:
        csv_dir (str): CSV文件目录路径
    """
    csv_dir = Path(csv_dir)
    
    if not csv_dir.exists():
        print(f"目录不存在: {csv_dir}")
        return
    
    # 获取所有CSV文件
    csv_files = list(csv_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"目录 {csv_dir} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    print("="*80)
    
    # 统计不同日期的文件数量
    date_counts = defaultdict(list)
    error_files = []
    
    # 处理每个文件
    for i, csv_file in enumerate(sorted(csv_files), 1):
        last_date = get_last_date_from_csv(csv_file)
        
        if last_date is None:
            error_files.append(csv_file.name)
        else:
            date_counts[last_date].append(csv_file.name)
        
        # 显示进度
        if i % 500 == 0 or i == len(csv_files):
            print(f"已处理: {i}/{len(csv_files)} 文件")
    
    print("\n" + "="*80)
    print("统计结果:")
    print("="*80)
    
    # 按日期排序并显示统计结果
    sorted_dates = sorted(date_counts.keys())
    
    for date in sorted_dates:
        files = date_counts[date]
        print(f"日期 {date}: {len(files)} 个文件")
        
        # 如果文件数量较少，显示文件名
        if len(files) <= 10:
            print(f"  文件: {', '.join(files)}")
        else:
            print(f"  文件: {', '.join(files[:5])} ... (还有{len(files)-5}个)")
        print()
    
    # 显示错误文件
    if error_files:
        print(f"读取出错的文件 ({len(error_files)} 个):")
        for error_file in error_files:
            print(f"  {error_file}")
        print()
    
    # 显示摘要
    print("="*80)
    print("摘要:")
    print(f"总文件数: {len(csv_files)}")
    print(f"成功读取: {len(csv_files) - len(error_files)}")
    print(f"读取出错: {len(error_files)}")
    print(f"不同日期数: {len(date_counts)}")
    
    if sorted_dates:
        print(f"最早日期: {sorted_dates[0]}")
        print(f"最晚日期: {sorted_dates[-1]}")
    
    # 找出最常见的日期
    if date_counts:
        most_common_date = max(date_counts.keys(), key=lambda x: len(date_counts[x]))
        print(f"最常见日期: {most_common_date} ({len(date_counts[most_common_date])} 个文件)")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='查看所有CSV文件的最后日期')
    parser.add_argument(
        '--csv_dir', 
        default='/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/build/market_data',
        help='CSV文件目录路径'
    )
    
    args = parser.parse_args()
    
    print(f"开始分析目录: {args.csv_dir}")
    analyze_csv_dates(args.csv_dir)

if __name__ == '__main__':
    main()