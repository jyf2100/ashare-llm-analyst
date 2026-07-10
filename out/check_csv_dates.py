#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV文件日期检查工具
检查所有导出的CSV文件，将最后一行日期不是2025-08-05的文件移动到备份目录
"""

import os
import shutil
import pandas as pd
from datetime import datetime
import argparse
import logging
from pathlib import Path


class CSVDateChecker:
    """CSV文件日期检查器"""
    
    def __init__(self, csv_dir, backup_dir, target_date="2025-08-05", verbose=False):
        """
        初始化CSV日期检查器
        
        Args:
            csv_dir (str): CSV文件目录
            backup_dir (str): 备份目录
            target_date (str): 目标日期，格式YYYY-MM-DD
            verbose (bool): 是否显示详细日志
        """
        self.csv_dir = Path(csv_dir)
        self.backup_dir = Path(backup_dir)
        self.target_date = target_date
        self.verbose = verbose
        
        # 设置日志
        self._setup_logging()
        
        # 创建备份目录
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.total_files = 0
        self.checked_files = 0
        self.moved_files = 0
        self.error_files = 0
        self.moved_file_list = []
        self.error_file_list = []
    
    def _setup_logging(self):
        """设置日志配置"""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_last_date_from_csv(self, csv_file):
        """
        获取CSV文件的最后一行日期
        
        Args:
            csv_file (Path): CSV文件路径
            
        Returns:
            str: 最后一行的日期，如果出错返回None
        """
        try:
            # 读取CSV文件的最后几行
            df = pd.read_csv(csv_file, usecols=['date'])
            if df.empty:
                self.logger.warning(f"文件 {csv_file.name} 为空")
                return None
            
            # 获取最后一行的日期
            last_date = df.iloc[-1]['date']
            
            # 验证日期格式
            try:
                datetime.strptime(last_date, '%Y-%m-%d')
                return last_date
            except ValueError:
                self.logger.warning(f"文件 {csv_file.name} 的日期格式不正确: {last_date}")
                return None
                
        except Exception as e:
            self.logger.error(f"读取文件 {csv_file.name} 时出错: {str(e)}")
            return None
    
    def check_and_move_file(self, csv_file):
        """
        检查单个CSV文件并在需要时移动到备份目录
        
        Args:
            csv_file (Path): CSV文件路径
            
        Returns:
            bool: 是否移动了文件
        """
        self.checked_files += 1
        
        # 获取最后一行日期
        last_date = self.get_last_date_from_csv(csv_file)
        
        if last_date is None:
            self.error_files += 1
            self.error_file_list.append(csv_file.name)
            return False
        
        if self.verbose:
            self.logger.debug(f"文件 {csv_file.name} 的最后日期: {last_date}")
        
        # 检查日期是否匹配目标日期
        if last_date != self.target_date:
            try:
                # 移动文件到备份目录
                backup_file = self.backup_dir / csv_file.name
                shutil.move(str(csv_file), str(backup_file))
                
                self.moved_files += 1
                self.moved_file_list.append(csv_file.name)
                self.logger.info(f"移动文件: {csv_file.name} (最后日期: {last_date}) -> {backup_file}")
                return True
                
            except Exception as e:
                self.logger.error(f"移动文件 {csv_file.name} 时出错: {str(e)}")
                self.error_files += 1
                self.error_file_list.append(csv_file.name)
                return False
        
        return False
    
    def check_all_files(self):
        """
        检查所有CSV文件
        """
        self.logger.info(f"开始检查CSV文件，目录: {self.csv_dir}")
        self.logger.info(f"目标日期: {self.target_date}")
        self.logger.info(f"备份目录: {self.backup_dir}")
        
        # 获取所有CSV文件
        csv_files = list(self.csv_dir.glob('*.csv'))
        self.total_files = len(csv_files)
        
        if self.total_files == 0:
            self.logger.warning(f"在目录 {self.csv_dir} 中未找到CSV文件")
            return
        
        self.logger.info(f"找到 {self.total_files} 个CSV文件")
        
        # 检查每个文件
        for i, csv_file in enumerate(csv_files, 1):
            if self.verbose:
                self.logger.debug(f"检查进度: {i}/{self.total_files} - {csv_file.name}")
            
            self.check_and_move_file(csv_file)
        
        # 输出统计信息
        self.print_summary()
    
    def print_summary(self):
        """
        打印检查结果摘要
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("检查完成 - 统计摘要")
        self.logger.info("="*60)
        self.logger.info(f"总文件数: {self.total_files}")
        self.logger.info(f"已检查文件数: {self.checked_files}")
        self.logger.info(f"移动到备份目录的文件数: {self.moved_files}")
        self.logger.info(f"错误文件数: {self.error_files}")
        self.logger.info(f"保留在原目录的文件数: {self.total_files - self.moved_files - self.error_files}")
        
        if self.moved_files > 0:
            self.logger.info(f"\n移动的文件列表 ({self.moved_files}个):")
            for filename in self.moved_file_list[:10]:  # 只显示前10个
                self.logger.info(f"  - {filename}")
            if len(self.moved_file_list) > 10:
                self.logger.info(f"  ... 还有 {len(self.moved_file_list) - 10} 个文件")
        
        if self.error_files > 0:
            self.logger.info(f"\n错误文件列表 ({self.error_files}个):")
            for filename in self.error_file_list:
                self.logger.info(f"  - {filename}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='检查CSV文件的最后日期，将不符合条件的文件移动到备份目录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例用法:
  python check_csv_dates.py
  python check_csv_dates.py --csv-dir ./data --backup-dir ./backup
  python check_csv_dates.py --target-date 2025-08-06 --verbose
        """
    )
    
    parser.add_argument(
        '--csv-dir',
        default='./market_data',
        help='CSV文件目录 (默认: ./market_data)'
    )
    
    parser.add_argument(
        '--backup-dir',
        default='./backup_csv',
        help='备份目录 (默认: ./backup_csv)'
    )
    
    parser.add_argument(
        '--target-date',
        default='2025-08-05',
        help='目标日期，格式YYYY-MM-DD (默认: 2025-08-05)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='试运行模式，只检查不移动文件'
    )
    
    args = parser.parse_args()
    
    try:
        # 验证目标日期格式
        datetime.strptime(args.target_date, '%Y-%m-%d')
    except ValueError:
        print(f"错误: 目标日期格式不正确: {args.target_date}，应为YYYY-MM-DD格式")
        return 1
    
    # 检查CSV目录是否存在
    if not os.path.exists(args.csv_dir):
        print(f"错误: CSV目录不存在: {args.csv_dir}")
        return 1
    
    if args.dry_run:
        print("试运行模式 - 只检查不移动文件")
    
    # 创建检查器并执行检查
    checker = CSVDateChecker(
        csv_dir=args.csv_dir,
        backup_dir=args.backup_dir,
        target_date=args.target_date,
        verbose=args.verbose
    )
    
    if args.dry_run:
        # 试运行模式：只检查不移动
        csv_files = list(Path(args.csv_dir).glob('*.csv'))
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        need_move = []
        for csv_file in csv_files:
            last_date = checker.get_last_date_from_csv(csv_file)
            if last_date and last_date != args.target_date:
                need_move.append((csv_file.name, last_date))
        
        print(f"需要移动的文件数: {len(need_move)}")
        if need_move:
            print("需要移动的文件:")
            for filename, last_date in need_move[:10]:
                print(f"  - {filename} (最后日期: {last_date})")
            if len(need_move) > 10:
                print(f"  ... 还有 {len(need_move) - 10} 个文件")
    else:
        checker.check_all_files()
    
    return 0


if __name__ == '__main__':
    exit(main())