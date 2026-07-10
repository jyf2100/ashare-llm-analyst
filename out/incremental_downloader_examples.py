#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量股票数据下载器使用示例
展示各种使用场景和参数组合
"""

import subprocess
import sys
import os
from datetime import datetime, timedelta

def run_command(cmd):
    """执行命令并显示结果"""
    print(f"\n执行命令: {cmd}")
    print("=" * 60)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("错误信息:")
        print(result.stderr)
    print("=" * 60)
    return result.returncode == 0

def example_1_backfill_30_days():
    """示例1: 往前补充30天数据"""
    print("\n📈 示例1: 往前补充30天数据")
    print("适用场景: 现有数据不够历史深度，需要往前补充更多历史数据")
    
    cmd = "python /mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/01-incremental_stock_data_downloader.py --backfill-days 30 --test"
    return run_command(cmd)

def example_2_forward_fill_to_date():
    """示例2: 往后补充到指定日期"""
    print("\n📅 示例2: 往后补充到指定日期")
    print("适用场景: 数据更新滞后，需要补充到最新日期")
    
    # 使用今天的日期
    today = datetime.now().strftime('%Y-%m-%d')
    cmd = f"python /mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/01-incremental_stock_data_downloader.py --forward-fill-date {today} --test"
    return run_command(cmd)

def example_3_both_directions():
    """示例3: 同时往前和往后补充数据"""
    print("\n🔄 示例3: 同时往前补充15天和往后补充到今天")
    print("适用场景: 数据有缺口，需要双向补充")
    
    today = datetime.now().strftime('%Y-%m-%d')
    cmd = f"python /mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/01-incremental_stock_data_downloader.py --backfill-days 15 --forward-fill-date {today} --test"
    return run_command(cmd)

def example_4_specific_stocks():
    """示例4: 指定特定股票进行补充"""
    print("\n🎯 示例4: 指定特定股票进行数据补充")
    print("适用场景: 只需要更新特定几只股票的数据")
    
    today = datetime.now().strftime('%Y-%m-%d')
    cmd = f"python /mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/01-incremental_stock_data_downloader.py --forward-fill-date {today} --stock-codes sh.600000 sh.600036 sz.000001"
    return run_command(cmd)

def example_5_limited_stocks():
    """示例5: 限制处理股票数量"""
    print("\n📊 示例5: 限制处理股票数量")
    print("适用场景: 分批处理大量股票，避免一次性处理过多")
    
    cmd = "python /mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/01-incremental_stock_data_downloader.py --backfill-days 10 --max-stocks 20"
    return run_command(cmd)

def example_6_production_update():
    """示例6: 生产环境日常更新"""
    print("\n🏭 示例6: 生产环境日常更新")
    print("适用场景: 每日定时任务，更新所有股票到最新日期")
    
    today = datetime.now().strftime('%Y-%m-%d')
    cmd = f"python /mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/01-incremental_stock_data_downloader.py --forward-fill-date {today}"
    print(f"生产环境命令: {cmd}")
    print("注意: 这个命令会处理所有现有股票，可能需要较长时间")
    print("建议在服务器上使用 nohup 或 screen 运行")
    return True

def show_help():
    """显示帮助信息"""
    print("\n❓ 查看完整帮助信息")
    cmd = "python /mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/01-incremental_stock_data_downloader.py --help"
    return run_command(cmd)

def check_data_status():
    """检查数据状态"""
    print("\n📋 检查现有数据状态")
    
    data_dir = "/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/market_data/"
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return False
    
    import glob
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"发现 {len(csv_files)} 个股票数据文件")
    
    if csv_files:
        print("\n前10个文件示例:")
        for i, file in enumerate(csv_files[:10]):
            filename = os.path.basename(file)
            size = os.path.getsize(file)
            print(f"  {i+1}. {filename} ({size} bytes)")
        
        if len(csv_files) > 10:
            print(f"  ... 还有 {len(csv_files) - 10} 个文件")
    
    return True

def main():
    """主函数"""
    print("🚀 增量股票数据下载器使用示例")
    print("=" * 80)
    
    # 检查数据状态
    check_data_status()
    
    # 显示帮助
    show_help()
    
    print("\n" + "=" * 80)
    print("📚 使用示例演示")
    print("=" * 80)
    
    examples = [
        ("1", "往前补充30天数据 (测试模式)", example_1_backfill_30_days),
        ("2", "往后补充到今天 (测试模式)", example_2_forward_fill_to_date),
        ("3", "双向补充数据 (测试模式)", example_3_both_directions),
        ("4", "指定股票补充", example_4_specific_stocks),
        ("5", "限制股票数量", example_5_limited_stocks),
        ("6", "生产环境示例", example_6_production_update),
    ]
    
    print("\n可用示例:")
    for num, desc, _ in examples:
        print(f"  {num}. {desc}")
    
    print("\n选择要运行的示例 (输入数字，或 'all' 运行所有测试示例，或 'q' 退出):")
    
    while True:
        choice = input("> ").strip().lower()
        
        if choice == 'q':
            print("退出示例程序")
            break
        elif choice == 'all':
            print("\n运行所有测试示例...")
            for num, desc, func in examples[:5]:  # 排除生产环境示例
                print(f"\n运行示例 {num}: {desc}")
                try:
                    func()
                except KeyboardInterrupt:
                    print("\n用户中断")
                    break
                except Exception as e:
                    print(f"示例 {num} 运行失败: {str(e)}")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            idx = int(choice) - 1
            num, desc, func = examples[idx]
            print(f"\n运行示例 {num}: {desc}")
            try:
                func()
            except KeyboardInterrupt:
                print("\n用户中断")
            except Exception as e:
                print(f"示例运行失败: {str(e)}")
        else:
            print("无效选择，请输入 1-6 的数字，'all' 或 'q'")
    
    print("\n" + "=" * 80)
    print("📖 常用命令参考")
    print("=" * 80)
    
    commands = [
        ("往前补充50天数据", "--backfill-days 50"),
        ("更新到今天", f"--forward-fill-date {datetime.now().strftime('%Y-%m-%d')}"),
        ("双向补充", f"--backfill-days 30 --forward-fill-date {datetime.now().strftime('%Y-%m-%d')}"),
        ("指定股票", "--stock-codes sh.600000 sz.000001 --forward-fill-date 2024-12-31"),
        ("限制数量", "--backfill-days 20 --max-stocks 100"),
        ("测试模式", "--backfill-days 10 --test"),
    ]
    
    for desc, params in commands:
        print(f"\n{desc}:")
        print(f"  python 01-incremental_stock_data_downloader.py {params}")
    
    print("\n" + "=" * 80)
    print("⚠️  注意事项")
    print("=" * 80)
    print("""
1. 首次运行建议使用 --test 参数进行测试
2. 大批量更新建议分批进行，使用 --max-stocks 限制数量
3. 生产环境建议使用 nohup 或 screen 运行长时间任务
4. 网络不稳定时可能出现部分失败，可重新运行补充
5. 数据会自动去重，重复运行不会产生重复数据
6. 建议在非交易时间进行大批量更新
""")

if __name__ == "__main__":
    main()