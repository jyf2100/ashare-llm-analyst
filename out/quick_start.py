#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本
提供最常用的股票分析功能组合
"""

import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入整合系统
from integrated_stock_system import IntegratedStockSystem

def quick_analysis_small_scale():
    """
    小规模快速分析（推荐新手使用）
    - 100只股票
    - 150天历史数据
    - 30天RPS
    - 优化版计算器
    """
    print("\n🚀 小规模快速分析模式")
    print("适合: 新手用户、快速测试")
    print("配置: 100只股票, 150天数据, 30天RPS")
    
    system = IntegratedStockSystem()
    
    # 设置组件（小规模优化配置）
    system.setup_downloader(use_incremental=True)
    system.setup_rps_calculator(use_batch=False)
    system.setup_stock_selector()
    
    # 执行完整流程
    result = system.run_complete_workflow(
        target_count=100,
        days=150,
        rps_period=30,
        analysis_days=20,
        force_update=False
    )
    
    return result

def quick_analysis_medium_scale():
    """
    中等规模分析（推荐日常使用）
    - 500只股票
    - 300天历史数据
    - 50天RPS
    - 优化版计算器
    """
    print("\n🚀 中等规模分析模式")
    print("适合: 日常使用、专业分析")
    print("配置: 500只股票, 300天数据, 50天RPS")
    
    system = IntegratedStockSystem()
    
    # 设置组件（中等规模配置）
    system.setup_downloader(use_incremental=True)
    system.setup_rps_calculator(use_batch=False)
    system.setup_stock_selector()
    
    # 执行完整流程
    result = system.run_complete_workflow(
        target_count=500,
        days=300,
        rps_period=50,
        analysis_days=30,
        force_update=False
    )
    
    return result

def quick_analysis_large_scale():
    """
    大规模分析（推荐专业用户）
    - 1000只股票
    - 500天历史数据
    - 50天RPS
    - 分批计算器
    """
    print("\n🚀 大规模分析模式")
    print("适合: 专业用户、全面分析")
    print("配置: 1000只股票, 500天数据, 50天RPS, 分批处理")
    
    system = IntegratedStockSystem()
    
    # 设置组件（大规模配置）
    system.setup_downloader(use_incremental=True)
    system.setup_rps_calculator(use_batch=True, batch_size=500, memory_limit_gb=6)
    system.setup_stock_selector()
    
    # 执行完整流程
    result = system.run_complete_workflow(
        target_count=1000,
        days=500,
        rps_period=50,
        analysis_days=30,
        force_update=False
    )
    
    return result

def quick_update_analysis():
    """
    快速更新分析
    - 基于现有数据
    - 仅更新过期数据
    - 重新计算RPS和选股
    """
    print("\n🔄 快速更新分析模式")
    print("适合: 数据更新、日常维护")
    
    system = IntegratedStockSystem()
    
    # 检查现有数据
    csv_files = [f for f in os.listdir(system.data_dir) if f.endswith('.csv')]
    stock_count = len(csv_files)
    
    if stock_count == 0:
        print("❌ 没有找到现有数据，请先运行完整分析")
        return None
    
    print(f"📊 发现 {stock_count} 只股票的现有数据")
    
    # 设置组件（根据数据量自动选择）
    system.setup_downloader(use_incremental=True)
    system.setup_rps_calculator(use_batch=stock_count > 800)
    system.setup_stock_selector()
    
    # 增量更新数据
    print("\n=== 步骤1: 增量更新数据 ===")
    download_result = system.download_data(
        target_count=stock_count,
        days=300,
        force_update=False  # 仅更新过期数据
    )
    
    # 重新计算RPS
    print("\n=== 步骤2: 重新计算RPS ===")
    rps_success = system.calculate_rps(rps_period=50)
    
    # 重新选股分析
    print("\n=== 步骤3: 重新选股分析 ===")
    analysis_results = system.analyze_stocks(analysis_days=30)
    
    # 获取最新推荐
    print("\n=== 步骤4: 获取最新推荐 ===")
    recommendations = system.get_recommendations(
        strategy='advanced_strategy',
        top_n=20
    )
    
    return {
        'download_result': download_result,
        'rps_success': rps_success,
        'analysis_results': analysis_results,
        'recommendations': recommendations
    }

def quick_recommendation_only():
    """
    仅获取推荐（基于现有数据）
    - 不下载新数据
    - 不重新计算RPS
    - 仅基于现有数据进行选股分析
    """
    print("\n⚡ 快速推荐模式")
    print("适合: 快速查看推荐、基于现有数据")
    
    system = IntegratedStockSystem()
    system.setup_stock_selector()
    
    # 检查数据
    csv_files = [f for f in os.listdir(system.data_dir) if f.endswith('.csv')]
    rps_files = [f for f in os.listdir(system.data_dir) if 'rps' in f and f.endswith('.pkl')]
    
    if not csv_files:
        print("❌ 没有找到股票数据，请先运行完整分析")
        return None
    
    if not rps_files:
        print("⚠️ 没有找到RPS数据，推荐结果可能不准确")
    
    print(f"📊 基于 {len(csv_files)} 只股票的现有数据")
    
    # 执行选股分析
    analysis_results = system.analyze_stocks(analysis_days=30)
    
    if analysis_results is None:
        return None
    
    # 获取多种策略的推荐
    recommendations = {}
    strategies = ['basic_strategy', 'advanced_strategy', 'momentum_strategy']
    
    for strategy in strategies:
        print(f"\n=== {strategy} 推荐 ===")
        recs = system.get_recommendations(
            strategy=strategy,
            top_n=15,
            min_selection_rate=25
        )
        recommendations[strategy] = recs
    
    return {
        'analysis_results': analysis_results,
        'recommendations': recommendations
    }

def show_system_status():
    """
    显示系统状态和数据概况
    """
    print("\n📊 系统状态检查")
    
    data_dir = "market_data"
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 检查股票数据
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"📈 股票数据文件: {len(csv_files)} 个")
    
    if csv_files:
        # 检查数据时间范围
        sample_file = os.path.join(data_dir, csv_files[0])
        try:
            import pandas as pd
            df = pd.read_csv(sample_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                print(f"📅 数据时间范围: {df['date'].min().date()} 到 {df['date'].max().date()}")
                print(f"📊 平均数据行数: {len(df)} 行")
        except:
            pass
    
    # 检查RPS数据
    rps_files = [f for f in os.listdir(data_dir) if 'rps' in f and f.endswith('.pkl')]
    print(f"🎯 RPS数据文件: {len(rps_files)} 个")
    
    if rps_files:
        latest_rps = max(rps_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
        rps_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(data_dir, latest_rps)))
        print(f"🕒 最新RPS文件: {latest_rps} ({rps_time.strftime('%Y-%m-%d %H:%M')})")
    
    # 检查分析结果
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    analysis_files = [f for f in json_files if 'analysis' in f]
    print(f"📋 分析结果文件: {len(analysis_files)} 个")
    
    if analysis_files:
        latest_analysis = max(analysis_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
        analysis_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(data_dir, latest_analysis)))
        print(f"📊 最新分析结果: {latest_analysis} ({analysis_time.strftime('%Y-%m-%d %H:%M')})")
    
    # 磁盘使用情况
    total_size = 0
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
    
    print(f"💾 数据目录大小: {total_size / 1024 / 1024:.1f} MB")
    
    # 推荐操作
    print("\n💡 推荐操作:")
    if len(csv_files) == 0:
        print("   - 运行小规模快速分析开始使用")
    elif len(rps_files) == 0:
        print("   - 计算RPS指标")
    elif len(analysis_files) == 0:
        print("   - 执行选股分析")
    else:
        print("   - 运行快速更新分析获取最新推荐")
        print("   - 或直接获取基于现有数据的推荐")

def main():
    """
    主菜单
    """
    print("\n" + "="*60)
    print("⚡ 股票分析系统 - 快速启动")
    print("="*60)
    
    while True:
        print("\n🎯 快速模式选择:")
        print("1. 小规模快速分析 (100只股票, 适合新手)")
        print("2. 中等规模分析 (500只股票, 推荐日常使用)")
        print("3. 大规模分析 (1000只股票, 专业用户)")
        print("4. 快速更新分析 (基于现有数据更新)")
        print("5. 仅获取推荐 (基于现有数据)")
        print("6. 系统状态检查")
        print("7. 进入完整功能模式")
        print("0. 退出")
        
        choice = input("\n请选择模式 (0-7): ").strip()
        
        try:
            if choice == '0':
                print("👋 感谢使用！")
                break
            
            elif choice == '1':
                quick_analysis_small_scale()
            
            elif choice == '2':
                quick_analysis_medium_scale()
            
            elif choice == '3':
                quick_analysis_large_scale()
            
            elif choice == '4':
                quick_update_analysis()
            
            elif choice == '5':
                quick_recommendation_only()
            
            elif choice == '6':
                show_system_status()
            
            elif choice == '7':
                print("\n🔄 切换到完整功能模式...")
                from integrated_stock_system import main as full_main
                full_main()
                break
            
            else:
                print("❌ 无效选择，请重新输入")
        
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 操作失败: {str(e)}")
            print("请检查网络连接和数据完整性")
            
            # 提供故障排除建议
            print("\n🔧 故障排除建议:")
            print("   - 检查网络连接")
            print("   - 确认akshare库已正确安装")
            print("   - 尝试运行系统状态检查")
            print("   - 如果是内存问题，尝试小规模分析")

if __name__ == "__main__":
    main()