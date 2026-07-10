#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qlib数据转换示例脚本

演示如何将Qlib的二进制数据转换为CSV格式
"""

from qlib_to_csv_converter import QlibDataConverter
import os

def main():
    """主函数 - 演示数据转换过程"""
    
    # 设置路径
    qlib_data_path = "/mnt/disk01/workspaces/worksummary/RD-Agent/build/qlib_data/cn_data"
    output_path = "/mnt/disk01/workspaces/worksummary/RD-Agent/build/market_data"
    
    print("=== Qlib数据转换示例 ===")
    print(f"数据源路径: {qlib_data_path}")
    print(f"输出路径: {output_path}")
    
    # 创建转换器
    converter = QlibDataConverter(qlib_data_path, output_path)
    
    # 示例1: 转换单个股票（浦发银行 600000）
    print("\n--- 示例1: 转换单个股票 ---")
    success = converter.convert_stock_data('sh600000')
    if success:
        csv_file = os.path.join(output_path, "600000.csv")
        print(f"✓ 转换成功！CSV文件已保存到: {csv_file}")
        
        # 显示文件信息
        if os.path.exists(csv_file):
            file_size = os.path.getsize(csv_file) / 1024  # KB
            print(f"  文件大小: {file_size:.1f} KB")
    
    # 示例2: 转换多个指定股票
    print("\n--- 示例2: 转换多个指定股票 ---")
    stock_list = ['sh600036', 'sh600519', 'sh000001']  # 招商银行、茅台、平安银行
    converter.convert_specific_stocks(stock_list)
    
    # 示例3: 批量转换（限制数量）
    print("\n--- 示例3: 批量转换前5个股票 ---")
    converter.convert_all_stocks(limit=5)
    
    print("\n=== 转换完成 ===")
    print(f"请查看输出目录: {output_path}")
    
    # 列出生成的CSV文件
    if os.path.exists(output_path):
        csv_files = [f for f in os.listdir(output_path) if f.endswith('.csv')]
        print(f"\n生成的CSV文件数量: {len(csv_files)}")
        if csv_files:
            print("文件列表:")
            for i, file in enumerate(sorted(csv_files)[:10], 1):  # 显示前10个
                print(f"  {i}. {file}")
            if len(csv_files) > 10:
                print(f"  ... 还有 {len(csv_files) - 10} 个文件")

if __name__ == "__main__":
    main()