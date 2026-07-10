#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指定股票代码分析工具
支持通过命令行参数指定要分析的股票代码
"""

import sys
import os
import argparse
from daily_stock_data_analyzer import DailyStockDataAnalyzer, analyze_specific_stocks

def main():
    """主函数 - 支持命令行参数"""
    parser = argparse.ArgumentParser(description='分析指定股票代码的技术指标')
    parser.add_argument('stock_codes', nargs='*', help='要分析的股票代码列表（例如：sh.600000 sz.000001）')
    parser.add_argument('--file', '-f', help='包含股票代码的文件路径，每行一个代码')
    parser.add_argument('--output', '-o', default='specific_stock_analysis_report.csv', 
                       help='输出报告文件名')
    parser.add_argument('--limit', '-l', type=int, default=50, 
                       help='当不指定股票代码时，分析前N只股票')
    
    args = parser.parse_args()
    
    stock_codes = []
    
    # 从文件读取股票代码
    if args.file:
        if os.path.exists(args.file):
            with open(args.file, 'r', encoding='utf-8') as f:
                stock_codes = [line.strip() for line in f if line.strip()]
            print(f"从文件 {args.file} 读取了 {len(stock_codes)} 个股票代码")
        else:
            print(f"文件不存在: {args.file}")
            return
    
    # 添加命令行参数中的股票代码
    if args.stock_codes:
        stock_codes.extend(args.stock_codes)
    
    # 去除重复
    stock_codes = list(set(stock_codes))
    
    if not stock_codes:
        print("未指定股票代码，将分析前50只股票")
        # 使用默认分析
        analyzer = DailyStockDataAnalyzer()
        analyzer.load_stock_data(limit=args.limit)
        analysis_df = analyzer.analyze_all_stocks()
        
        if not analysis_df.empty:
            top_stocks = analyzer.get_top_performers(10)
            print("\n技术评分最高的10只股票:")
            print(top_stocks[['stock_code', 'close_price', 'price_change_pct', 'technical_score', 'trend']])
            
            analyzer.export_analysis_report(args.output)
            print(f"分析报告已导出至: {args.output}")
        
        return
    
    print(f"开始分析 {len(stock_codes)} 只指定股票: {', '.join(stock_codes)}")
    
    # 分析指定股票
    analyze_specific_stocks(stock_codes)

if __name__ == "__main__":
    main()