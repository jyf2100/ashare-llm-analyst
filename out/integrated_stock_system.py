#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合的股票系统
集成最新的选股、下载、计算RPS功能
提供统一的接口和工作流程
"""

import os
import sys
import json
import time
from datetime import datetime
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# 加载环境变量
load_dotenv(override=True)

# 导入核心模块
from optimized_market_downloader import OptimizedMarketDownloader
from incremental_market_downloader import IncrementalMarketDownloader
from optimized_rps_calculator import OptimizedRPSCalculator
from batch_rps_calculator import BatchRPSCalculator
from optimized_stock_selection import OptimizedStockSelector

class IntegratedStockSystem:
    """
    整合的股票系统
    提供完整的数据下载、RPS计算、选股分析功能
    """
    
    def __init__(self, data_dir="market_data", proxy_config=None):
        """
        初始化整合系统
        
        参数:
        data_dir: str, 数据目录
        proxy_config: dict, 代理配置
        """
        self.data_dir = data_dir
        
        # 从环境变量读取代理配置
        if proxy_config is None and os.getenv('HTTP_PROXY') and os.getenv('HTTPS_PROXY'):
            proxy_config = {
                'http_proxy': os.getenv('HTTP_PROXY'),
                'https_proxy': os.getenv('HTTPS_PROXY')
            }
        
        self.proxy_config = proxy_config
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 初始化各个组件
        self.downloader = None
        self.rps_calculator = None
        self.stock_selector = None
        
        print("=== 整合股票系统初始化完成 ===")
    
    def setup_downloader(self, use_incremental=True):
        """
        设置数据下载器
        
        参数:
        use_incremental: bool, 是否使用增量下载器
        """
        if use_incremental:
            self.downloader = IncrementalMarketDownloader(
                data_dir=self.data_dir,
                proxy_config=self.proxy_config
            )
            print("✅ 已设置增量数据下载器")
        else:
            self.downloader = OptimizedMarketDownloader(
                data_dir=self.data_dir,
                proxy_config=self.proxy_config
            )
            print("✅ 已设置优化数据下载器")
    
    def setup_rps_calculator(self, use_batch=False, batch_size=1000, memory_limit_gb=4):
        """
        设置RPS计算器
        
        参数:
        use_batch: bool, 是否使用分批计算器
        batch_size: int, 批次大小
        memory_limit_gb: float, 内存限制
        """
        if use_batch:
            self.rps_calculator = BatchRPSCalculator(
                data_dir=self.data_dir,
                batch_size=batch_size,
                memory_limit_gb=memory_limit_gb
            )
            print(f"✅ 已设置分批RPS计算器 (批次大小: {batch_size})")
        else:
            self.rps_calculator = OptimizedRPSCalculator(
                data_dir=self.data_dir
            )
            print("✅ 已设置优化RPS计算器")
    
    def setup_stock_selector(self):
        """
        设置选股分析器
        """
        self.stock_selector = OptimizedStockSelector(
            data_dir=self.data_dir
        )
        print("✅ 已设置选股分析器")
    
    def download_data(self, target_count=500, days=300, force_update=False, batch_size=50):
        """
        下载股票数据
        
        参数:
        target_count: int, 目标股票数量
        days: int, 历史数据天数
        force_update: bool, 是否强制更新
        batch_size: int, 批次大小
        
        返回:
        dict: 下载结果
        """
        if self.downloader is None:
            self.setup_downloader(use_incremental=True)
        
        print(f"\n=== 开始下载股票数据 ===")
        print(f"目标数量: {target_count}只股票")
        print(f"历史天数: {days}天")
        print(f"强制更新: {force_update}")
        
        start_time = time.time()
        
        if isinstance(self.downloader, IncrementalMarketDownloader):
            result = self.downloader.incremental_batch_download(
                target_count=target_count,
                days=days,
                batch_size=batch_size,
                force_update=force_update
            )
        else:
            result = self.downloader.batch_download(
                target_count=target_count,
                days=days,
                batch_size=batch_size
            )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n=== 数据下载完成 ===")
        print(f"成功下载: {result.get('success_count', 0)}只股票")
        print(f"失败数量: {result.get('failed_count', 0)}只股票")
        print(f"总耗时: {elapsed_time:.2f}秒")
        
        return result
    
    def calculate_rps(self, rps_period=50, max_workers=None):
        """
        计算RPS指标
        
        参数:
        rps_period: int, RPS计算周期
        max_workers: int, 最大线程数（仅用于增量下载器）
        
        返回:
        bool: 计算是否成功
        """
        if self.rps_calculator is None:
            # 自动选择RPS计算器
            stock_count = len([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
            use_batch = stock_count > 1000
            self.setup_rps_calculator(use_batch=use_batch)
        
        print(f"\n=== 开始计算RPS指标 ===")
        print(f"RPS周期: {rps_period}天")
        
        start_time = time.time()
        
        try:
            if isinstance(self.rps_calculator, BatchRPSCalculator):
                result = self.rps_calculator.calculate_large_scale_rps(rps_period=rps_period)
            else:
                result = self.rps_calculator.calculate_rps_optimized(rps_period=rps_period)
            
            elapsed_time = time.time() - start_time
            
            print(f"\n=== RPS计算完成 ===")
            print(f"总耗时: {elapsed_time:.2f}秒")
            
            return True
            
        except Exception as e:
            print(f"❌ RPS计算失败: {str(e)}")
            
            # 如果使用增量下载器，尝试其内置的RPS计算
            if isinstance(self.downloader, IncrementalMarketDownloader):
                print("🔄 尝试使用增量下载器的内置RPS计算...")
                try:
                    self.downloader.calculate_rps(rps_period=rps_period, max_workers=max_workers)
                    print("✅ 使用内置RPS计算成功")
                    return True
                except Exception as e2:
                    print(f"❌ 内置RPS计算也失败: {str(e2)}")
            
            return False
    
    def analyze_stocks(self, analysis_days=30, strategies=None):
        """
        执行选股分析
        
        参数:
        analysis_days: int, 分析天数
        strategies: list, 分析策略列表
        
        返回:
        dict: 分析结果
        """
        if self.stock_selector is None:
            self.setup_stock_selector()
        
        if strategies is None:
            strategies = ['basic_strategy', 'advanced_strategy', 'momentum_strategy']
        
        print(f"\n=== 开始选股分析 ===")
        print(f"分析天数: {analysis_days}天")
        print(f"分析策略: {', '.join(strategies)}")
        
        start_time = time.time()
        
        # 加载数据
        if not self.stock_selector.load_data(use_optimized=True):
            print("❌ 数据加载失败")
            return None
        
        # 执行分析
        results = self.stock_selector.analyze_all_stocks(analysis_days=analysis_days)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n=== 选股分析完成 ===")
        print(f"总耗时: {elapsed_time:.2f}秒")
        
        # 打印分析报告
        self.stock_selector.print_analysis_report()
        
        return results
    
    def get_recommendations(self, strategy='advanced_strategy', top_n=20, min_selection_rate=30):
        """
        获取股票推荐
        
        参数:
        strategy: str, 选股策略
        top_n: int, 推荐数量
        min_selection_rate: float, 最小选中率
        
        返回:
        list: 推荐股票列表
        """
        if self.stock_selector is None:
            print("❌ 请先执行选股分析")
            return []
        
        recommendations = self.stock_selector.get_top_stocks(
            strategy=strategy,
            top_n=top_n,
            min_selection_rate=min_selection_rate
        )
        
        print(f"\n=== 股票推荐 ({strategy}) ===")
        print(f"推荐数量: {len(recommendations)}只股票")
        
        for i, stock in enumerate(recommendations[:10], 1):
            print(f"{i:2d}. {stock['stock_code']} - 选中率: {stock['selection_rate']:.1f}%")
        
        if len(recommendations) > 10:
            print(f"... 还有 {len(recommendations) - 10} 只股票")
        
        return recommendations
    
    def run_complete_workflow(self, target_count=500, days=300, rps_period=50, 
                            analysis_days=30, force_update=False):
        """
        运行完整的工作流程
        
        参数:
        target_count: int, 目标股票数量
        days: int, 历史数据天数
        rps_period: int, RPS计算周期
        analysis_days: int, 分析天数
        force_update: bool, 是否强制更新数据
        
        返回:
        dict: 完整的分析结果
        """
        print("\n" + "="*60)
        print("🚀 开始执行完整的股票分析工作流程")
        print("="*60)
        
        workflow_start = time.time()
        
        # 步骤1: 下载数据
        download_result = self.download_data(
            target_count=target_count,
            days=days,
            force_update=force_update
        )
        
        if download_result.get('success_count', 0) == 0:
            print("❌ 数据下载失败，无法继续")
            return None
        
        # 步骤2: 计算RPS
        rps_success = self.calculate_rps(rps_period=rps_period)
        
        if not rps_success:
            print("⚠️ RPS计算失败，但继续进行选股分析")
        
        # 步骤3: 选股分析
        analysis_results = self.analyze_stocks(analysis_days=analysis_days)
        
        if analysis_results is None:
            print("❌ 选股分析失败")
            return None
        
        # 步骤4: 获取推荐
        recommendations = {}
        for strategy in ['basic_strategy', 'advanced_strategy', 'momentum_strategy']:
            recommendations[strategy] = self.get_recommendations(
                strategy=strategy,
                top_n=20,
                min_selection_rate=30
            )
        
        workflow_time = time.time() - workflow_start
        
        # 生成完整报告
        complete_result = {
            'timestamp': datetime.now().isoformat(),
            'workflow_time': workflow_time,
            'download_result': download_result,
            'rps_success': rps_success,
            'analysis_results': analysis_results,
            'recommendations': recommendations,
            'parameters': {
                'target_count': target_count,
                'days': days,
                'rps_period': rps_period,
                'analysis_days': analysis_days,
                'force_update': force_update
            }
        }
        
        # 保存结果
        result_file = os.path.join(self.data_dir, f"complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(complete_result, f, ensure_ascii=False, indent=2, default=str)
        
        print("\n" + "="*60)
        print("🎉 完整工作流程执行完成")
        print(f"⏱️ 总耗时: {workflow_time:.2f}秒")
        print(f"📄 结果已保存到: {result_file}")
        print("="*60)
        
        return complete_result

def main():
    """
    主函数 - 提供交互式菜单
    """
    print("\n" + "="*60)
    print("📈 整合股票分析系统")
    print("="*60)
    
    # 初始化系统
    system = IntegratedStockSystem()
    
    while True:
        print("\n请选择操作:")
        print("1. 运行完整工作流程（推荐）")
        print("2. 仅下载数据")
        print("3. 仅计算RPS")
        print("4. 仅选股分析")
        print("5. 获取股票推荐")
        print("6. 查看系统状态")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-6): ").strip()
        
        try:
            if choice == '0':
                print("👋 感谢使用！")
                break
            
            elif choice == '1':
                print("\n=== 完整工作流程配置 ===")
                target_count = int(input("目标股票数量 [500]: ") or "500")
                days = int(input("历史数据天数 [300]: ") or "300")
                rps_period = int(input("RPS计算周期 [50]: ") or "50")
                analysis_days = int(input("分析天数 [30]: ") or "30")
                force_update = input("是否强制更新数据 [n]: ").lower().startswith('y')
                
                system.run_complete_workflow(
                    target_count=target_count,
                    days=days,
                    rps_period=rps_period,
                    analysis_days=analysis_days,
                    force_update=force_update
                )
            
            elif choice == '2':
                target_count = int(input("目标股票数量 [500]: ") or "500")
                days = int(input("历史数据天数 [300]: ") or "300")
                force_update = input("是否强制更新 [n]: ").lower().startswith('y')
                
                system.download_data(
                    target_count=target_count,
                    days=days,
                    force_update=force_update
                )
            
            elif choice == '3':
                rps_period = int(input("RPS计算周期 [50]: ") or "50")
                system.calculate_rps(rps_period=rps_period)
            
            elif choice == '4':
                analysis_days = int(input("分析天数 [30]: ") or "30")
                system.analyze_stocks(analysis_days=analysis_days)
            
            elif choice == '5':
                print("\n可用策略: basic_strategy, advanced_strategy, momentum_strategy")
                strategy = input("选择策略 [advanced_strategy]: ") or "advanced_strategy"
                top_n = int(input("推荐数量 [20]: ") or "20")
                
                system.get_recommendations(
                    strategy=strategy,
                    top_n=top_n
                )
            
            elif choice == '6':
                print("\n=== 系统状态 ===")
                csv_files = [f for f in os.listdir(system.data_dir) if f.endswith('.csv')]
                print(f"数据目录: {system.data_dir}")
                print(f"股票数据文件: {len(csv_files)}个")
                
                rps_files = [f for f in os.listdir(system.data_dir) if 'rps' in f and f.endswith('.pkl')]
                print(f"RPS数据文件: {len(rps_files)}个")
                
                if rps_files:
                    print(f"最新RPS文件: {max(rps_files)}")
                
                print(f"下载器: {'已设置' if system.downloader else '未设置'}")
                print(f"RPS计算器: {'已设置' if system.rps_calculator else '未设置'}")
                print(f"选股分析器: {'已设置' if system.stock_selector else '未设置'}")
            
            else:
                print("❌ 无效选择，请重新输入")
        
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 操作失败: {str(e)}")
            print("请检查输入并重试")

if __name__ == "__main__":
    main()