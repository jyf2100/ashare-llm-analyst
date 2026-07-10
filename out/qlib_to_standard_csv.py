#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qlib数据转标准CSV格式转换器

功能：
1. 将Qlib二进制数据转换为标准的股票CSV格式
2. 输出格式与常见股票数据格式一致
3. 列名：date,open,high,low,close,volume,change,factor
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import struct
from datetime import datetime, timedelta
import argparse
import logging
from typing import Dict, List, Optional

class QlibToStandardCSV:
    """Qlib数据转标准CSV格式转换器"""
    
    def __init__(self, qlib_data_path, output_path, verbose=True):
        """
        初始化转换器
        
        Args:
            qlib_data_path (str): Qlib数据根目录路径
            output_path (str): 输出CSV文件目录路径
            verbose (bool): 是否输出详细日志
        """
        self.qlib_data_path = Path(qlib_data_path)
        self.output_path = Path(output_path)
        self.features_path = self.qlib_data_path / "features"
        self.verbose = verbose
        
        # 设置日志
        self._setup_logging()
        
        self.logger.info(f"初始化转换器")
        self.logger.info(f"Qlib数据路径: {self.qlib_data_path}")
        self.logger.info(f"输出路径: {self.output_path}")
        self.logger.info(f"特征数据路径: {self.features_path}")
        
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"输出目录已创建/确认存在: {self.output_path}")
        
        # 读取交易日历
        self.calendar = self._load_calendar()
        
    def _setup_logging(self):
        """
        设置日志配置
        """
        self.logger = logging.getLogger('QlibConverter')
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # 清除已有的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def _load_calendar(self):
        """
        加载交易日历
        
        Returns:
            list: 交易日期列表
        """
        calendar_file = self.qlib_data_path / "calendars" / "day.txt"
        self.logger.info(f"尝试加载交易日历: {calendar_file}")
        
        if calendar_file.exists():
            try:
                with open(calendar_file, 'r') as f:
                    dates = [line.strip() for line in f.readlines()]
                dates = sorted(dates)
                self.logger.info(f"成功加载交易日历，共 {len(dates)} 个交易日")
                if dates:
                    self.logger.info(f"日期范围: {dates[0]} 到 {dates[-1]}")
                return dates
            except Exception as e:
                self.logger.error(f"读取交易日历文件时出错: {e}")
                return []
        else:
            self.logger.warning(f"未找到交易日历文件: {calendar_file}")
            self.logger.info("将使用生成的日期序列")
            return []
    
    def _read_bin_file(self, file_path: Path) -> np.array:
        """
        读取二进制数据文件
        
        Args:
            file_path (Path): 二进制文件路径
            
        Returns:
            np.array: 数据数组
        """
        if not file_path.exists():
            self.logger.debug(f"文件不存在: {file_path}")
            return np.array([])
            
        try:
            file_size = file_path.stat().st_size
            self.logger.debug(f"读取文件: {file_path.name}, 大小: {file_size} bytes")
            
            with open(file_path, 'rb') as f:
                data = f.read()
                
            # Qlib的二进制文件通常是float32格式
            values = np.frombuffer(data, dtype=np.float32)
            self.logger.debug(f"成功读取 {len(values)} 个数据点")
            
            if len(values) > 0:
                self.logger.debug(f"数据范围: {values.min():.6f} 到 {values.max():.6f}")
                # 检查异常值
                nan_count = np.isnan(values).sum()
                inf_count = np.isinf(values).sum()
                if nan_count > 0:
                    self.logger.warning(f"发现 {nan_count} 个NaN值")
                if inf_count > 0:
                    self.logger.warning(f"发现 {inf_count} 个无穷值")
            
            return values
        except Exception as e:
            self.logger.error(f"读取文件 {file_path} 时出错: {e}")
            return np.array([])
    
    def convert_stock_data(self, stock_code: str) -> bool:
        """
        转换单个股票的数据为标准CSV格式
        
        Args:
            stock_code (str): 股票代码（如 sh600000）
            
        Returns:
            bool: 转换是否成功
        """
        self.logger.info(f"="*50)
        self.logger.info(f"开始转换股票: {stock_code}")
        
        stock_path = self.features_path / stock_code
        if not stock_path.exists():
            self.logger.error(f"股票 {stock_code} 的数据目录不存在: {stock_path}")
            return False
            
        self.logger.info(f"股票数据目录: {stock_path}")
        
        # 读取各种数据文件
        data_files = {
            'open': stock_path / "open.day.bin",
            'high': stock_path / "high.day.bin", 
            'low': stock_path / "low.day.bin",
            'close': stock_path / "close.day.bin",
            'volume': stock_path / "volume.day.bin",
            'change': stock_path / "change.day.bin",
            'factor': stock_path / "factor.day.bin"
        }
        
        self.logger.info(f"准备读取 {len(data_files)} 个数据文件")
        
        # 读取数据
        data_dict = {}
        max_length = 0
        file_status = {}
        
        for field, file_path in data_files.items():
            self.logger.debug(f"读取字段: {field}")
            values = self._read_bin_file(file_path)
            data_dict[field] = values
            file_status[field] = {
                'exists': file_path.exists(),
                'length': len(values),
                'has_data': len(values) > 0
            }
            
            if len(values) > max_length:
                max_length = len(values)
        
        # 输出文件状态摘要
        self.logger.info("文件读取状态摘要:")
        for field, status in file_status.items():
            status_str = f"  {field}: 存在={status['exists']}, 长度={status['length']}, 有数据={status['has_data']}"
            if status['has_data']:
                self.logger.info(status_str)
            else:
                self.logger.warning(status_str)
        
        self.logger.info(f"最大数据长度: {max_length}")
                
        if max_length == 0:
            self.logger.error(f"股票 {stock_code} 没有有效数据")
            return False
            
        # 创建DataFrame
        self.logger.info("开始构建DataFrame")
        df_data = {}
        
        # 使用交易日历作为日期索引
        if len(self.calendar) >= max_length:
            dates = self.calendar[-max_length:]
            self.logger.info(f"使用交易日历，取最后 {max_length} 个交易日")
        else:
            # 如果没有交易日历，生成日期序列
            end_date = datetime.now()
            start_date = end_date - timedelta(days=max_length * 2)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            dates = [d.strftime('%Y-%m-%d') for d in dates[-max_length:]]
            self.logger.warning(f"交易日历不足，生成日期序列，长度: {len(dates)}")
            
        df_data['date'] = dates
        self.logger.debug(f"日期范围: {dates[0]} 到 {dates[-1]}")
        
        # 添加各字段数据，如果长度不足则用NaN填充
        self.logger.info("处理各字段数据对齐")
        alignment_info = []
        for field, values in data_dict.items():
            if len(values) == max_length:
                df_data[field] = values
                alignment_info.append(f"{field}: 完全匹配({len(values)})")
            elif len(values) > 0:
                # 如果数据长度不一致，从末尾对齐
                padded_values = np.full(max_length, np.nan)
                padded_values[-len(values):] = values
                df_data[field] = padded_values
                alignment_info.append(f"{field}: 末尾对齐({len(values)}->{max_length})")
            else:
                df_data[field] = np.full(max_length, np.nan)
                alignment_info.append(f"{field}: 全NaN填充({max_length})")
        
        for info in alignment_info:
            self.logger.debug(f"  {info}")
        
        # 创建DataFrame
        df = pd.DataFrame(df_data)
        self.logger.info(f"初始DataFrame创建完成，形状: {df.shape}")
        
        # 确保列的顺序和名称标准化
        standard_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'change', 'factor']
        df = df.reindex(columns=standard_columns)
        self.logger.debug(f"列顺序标准化完成: {list(df.columns)}")
        
        # 数据质量检查
        self.logger.info("进行数据质量检查")
        initial_rows = len(df)
        
        # 过滤掉全为NaN的行
        df_before_nan_filter = len(df)
        df = df.dropna(subset=['open', 'close'], how='all')
        nan_filtered = df_before_nan_filter - len(df)
        if nan_filtered > 0:
            self.logger.info(f"过滤掉 {nan_filtered} 行全NaN数据")
        
        # 过滤掉价格为0的无效数据行
        df_before_zero_filter = len(df)
        df = df[(df['open'] > 0) | (df['close'] > 0)]
        zero_filtered = df_before_zero_filter - len(df)
        if zero_filtered > 0:
            self.logger.info(f"过滤掉 {zero_filtered} 行零价格数据")
        
        final_rows = len(df)
        self.logger.info(f"数据过滤完成: {initial_rows} -> {final_rows} 行")
        
        if df.empty:
            self.logger.error(f"股票 {stock_code} 过滤后没有有效数据")
            return False
        
        # 数据统计信息
        self.logger.info("数据统计信息:")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns and not df[col].isna().all():
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    self.logger.info(f"  {col}: 有效数据 {len(valid_data)} 个, 范围 {valid_data.min():.6f} - {valid_data.max():.6f}")
            
        # 转换股票代码格式（sh600000 -> 600000, sz000001 -> 000001）
        clean_code = stock_code.replace('sh', '').replace('sz', '')
        output_file = self.output_path / f"{clean_code}.csv"
        self.logger.info(f"准备保存到文件: {output_file}")
        
        # 保存为CSV，使用标准格式
        try:
            df.to_csv(output_file, index=False, float_format='%.6f')
            file_size = output_file.stat().st_size
            self.logger.info(f"文件保存成功: {output_file}")
            self.logger.info(f"文件大小: {file_size} bytes")
            self.logger.info(f"数据行数: {len(df)}, 日期范围: {df['date'].iloc[0]} 到 {df['date'].iloc[-1]}")
            self.logger.info(f"股票 {stock_code} 转换完成")
            return True
        except Exception as e:
            self.logger.error(f"保存文件时出错: {e}")
            return False
    
    def convert_all_stocks(self, limit: Optional[int] = None) -> Dict[str, int]:
        """
        转换所有股票数据
        
        Args:
            limit (int, optional): 限制转换的股票数量
            
        Returns:
            Dict[str, int]: 转换结果统计
        """
        self.logger.info("="*60)
        self.logger.info("开始批量转换所有股票数据")
        
        if not self.features_path.exists():
            self.logger.error(f"特征数据目录不存在: {self.features_path}")
            return {'total': 0, 'success': 0, 'failed': 0}
            
        stock_dirs = [d for d in self.features_path.iterdir() if d.is_dir()]
        self.logger.info(f"扫描到 {len(stock_dirs)} 个股票数据目录")
        
        if limit:
            original_count = len(stock_dirs)
            stock_dirs = stock_dirs[:limit]
            self.logger.info(f"应用数量限制: {original_count} -> {len(stock_dirs)} 个股票")
        
        # 显示将要转换的股票列表
        stock_codes = [d.name for d in stock_dirs]
        self.logger.info(f"将要转换的股票: {', '.join(stock_codes[:10])}{'...' if len(stock_codes) > 10 else ''}")
            
        success_count = 0
        failed_stocks = []
        start_time = datetime.now()
        
        for i, stock_dir in enumerate(stock_dirs, 1):
            stock_code = stock_dir.name
            self.logger.info(f"\n进度: {i}/{len(stock_dirs)} ({i/len(stock_dirs)*100:.1f}%)")
            
            try:
                if self.convert_stock_data(stock_code):
                    success_count += 1
                else:
                    failed_stocks.append(stock_code)
            except Exception as e:
                self.logger.error(f"转换股票 {stock_code} 时发生异常: {e}")
                failed_stocks.append(stock_code)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # 输出转换结果摘要
        self.logger.info("\n" + "="*60)
        self.logger.info("批量转换完成！")
        self.logger.info(f"总耗时: {duration}")
        self.logger.info(f"转换结果: 成功 {success_count}/{len(stock_dirs)} 个股票")
        self.logger.info(f"成功率: {success_count/len(stock_dirs)*100:.1f}%")
        
        if failed_stocks:
            self.logger.warning(f"转换失败的股票 ({len(failed_stocks)} 个): {', '.join(failed_stocks[:10])}{'...' if len(failed_stocks) > 10 else ''}")
        
        return {
            'total': len(stock_dirs),
            'success': success_count,
            'failed': len(failed_stocks),
            'duration_seconds': duration.total_seconds()
        }
    
    def convert_specific_stocks(self, stock_codes: List[str]) -> Dict[str, int]:
        """
        转换指定的股票数据
        
        Args:
            stock_codes (list): 股票代码列表
            
        Returns:
            Dict[str, int]: 转换结果统计
        """
        self.logger.info("="*60)
        self.logger.info(f"开始转换指定的 {len(stock_codes)} 个股票")
        self.logger.info(f"股票列表: {', '.join(stock_codes)}")
        
        success_count = 0
        failed_stocks = []
        start_time = datetime.now()
        
        for i, stock_code in enumerate(stock_codes, 1):
            self.logger.info(f"\n进度: {i}/{len(stock_codes)} ({i/len(stock_codes)*100:.1f}%)")
            
            try:
                if self.convert_stock_data(stock_code):
                    success_count += 1
                else:
                    failed_stocks.append(stock_code)
            except Exception as e:
                self.logger.error(f"转换股票 {stock_code} 时发生异常: {e}")
                failed_stocks.append(stock_code)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # 输出转换结果摘要
        self.logger.info("\n" + "="*60)
        self.logger.info("指定股票转换完成！")
        self.logger.info(f"总耗时: {duration}")
        self.logger.info(f"转换结果: 成功 {success_count}/{len(stock_codes)} 个股票")
        self.logger.info(f"成功率: {success_count/len(stock_codes)*100:.1f}%")
        
        if failed_stocks:
            self.logger.warning(f"转换失败的股票: {', '.join(failed_stocks)}")
        
        return {
            'total': len(stock_codes),
            'success': success_count,
            'failed': len(failed_stocks),
            'duration_seconds': duration.total_seconds()
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Qlib数据转标准CSV格式转换器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python qlib_to_standard_csv.py --sample                    # 转换示例股票
  python qlib_to_standard_csv.py --stocks sh600000 sz000001  # 转换指定股票
  python qlib_to_standard_csv.py --limit 10                  # 转换前10个股票
  python qlib_to_standard_csv.py --verbose                   # 详细日志模式
  python qlib_to_standard_csv.py --quiet                     # 静默模式
        """
    )
    parser.add_argument('--qlib_path', type=str, 
                       default='/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/build/qlib_data/cn_data',
                       help='Qlib数据目录路径')
    parser.add_argument('--output_path', type=str,
                       default='/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/build/market_data',
                       help='输出CSV文件目录路径')
    parser.add_argument('--stocks', type=str, nargs='*',
                       help='指定要转换的股票代码列表，如: sh600000 sh600036')
    parser.add_argument('--limit', type=int,
                       help='限制转换的股票数量（用于测试）')
    parser.add_argument('--sample', action='store_true',
                       help='仅转换示例股票（sh600000）')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='输出详细日志信息')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='静默模式，只输出错误信息')
    
    args = parser.parse_args()
    
    # 确定日志级别
    verbose = args.verbose and not args.quiet
    
    # 程序启动信息
    start_time = datetime.now()
    print("=" * 80)
    print("Qlib数据转标准CSV格式转换器")
    print(f"启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"输入路径: {args.qlib_path}")
    print(f"输出路径: {args.output_path}")
    print(f"输出格式: date,open,high,low,close,volume,change,factor")
    print(f"日志模式: {'详细' if verbose else '静默' if args.quiet else '标准'}")
    
    if args.sample:
        print("运行模式: 示例转换 (sh600000)")
    elif args.stocks:
        print(f"运行模式: 指定股票转换 ({len(args.stocks)} 个股票)")
    elif args.limit:
        print(f"运行模式: 限量转换 (前 {args.limit} 个股票)")
    else:
        print("运行模式: 全量转换")
    
    print("=" * 80)
    
    try:
        # 创建转换器
        converter = QlibToStandardCSV(args.qlib_path, args.output_path, verbose=verbose)
        
        # 执行转换
        if args.sample:
            # 转换示例股票
            result = converter.convert_specific_stocks(['sh600000'])
        elif args.stocks:
            # 转换指定股票
            result = converter.convert_specific_stocks(args.stocks)
        else:
            # 转换所有股票
            result = converter.convert_all_stocks(args.limit)
        
        # 输出最终结果摘要
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("程序执行完成")
        print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {total_duration}")
        print(f"转换结果: 成功 {result['success']}/{result['total']} 个股票")
        print(f"成功率: {result['success']/result['total']*100:.1f}%" if result['total'] > 0 else "成功率: N/A")
        print("=" * 80)
        
        # 根据结果设置退出码
        if result['success'] == result['total']:
            exit_code = 0  # 全部成功
        elif result['success'] > 0:
            exit_code = 1  # 部分成功
        else:
            exit_code = 2  # 全部失败
            
        return exit_code
        
    except KeyboardInterrupt:
        print("\n用户中断程序执行")
        return 130
    except Exception as e:
        print(f"\n程序执行时发生错误: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()