#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IP限制解决方案
当遇到IP被限制时的解决方案和替代数据源
"""

import time
import random
import requests
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

class IPRestrictionSolver:
    """IP限制解决方案类"""
    
    def __init__(self, proxy_config: Optional[Dict] = None):
        """
        初始化IP限制解决器
        
        Args:
            proxy_config: 代理配置字典
        """
        # 从环境变量读取代理配置
        default_proxy = {
            'http': os.getenv('HTTP_PROXY', 'http://172.32.147.190:7890'),
            'https': os.getenv('HTTPS_PROXY', 'http://172.32.147.190:7890')
        }
        self.proxy_config = proxy_config or default_proxy
        self.setup_proxy()
        self.backup_data_dir = 'backup_data'
        os.makedirs(self.backup_data_dir, exist_ok=True)
        
    def setup_proxy(self):
        """设置代理环境变量"""
        if self.proxy_config:
            os.environ['http_proxy'] = self.proxy_config.get('http', '')
            os.environ['https_proxy'] = self.proxy_config.get('https', '')
            print(f"✅ 代理已设置: {self.proxy_config.get('http')}")
    
    def test_network_connectivity(self) -> Dict[str, bool]:
        """测试网络连接状态"""
        results = {
            'basic_internet': False,
            'proxy_working': False,
            'akshare_api': False,
            'alternative_sources': False
        }
        
        # 测试基本网络连接
        try:
            response = requests.get('http://www.baidu.com', timeout=10)
            results['basic_internet'] = response.status_code == 200
            print("✅ 基本网络连接正常")
        except Exception as e:
            print(f"❌ 基本网络连接失败: {e}")
        
        # 测试代理是否工作
        try:
            proxies = {
                'http': self.proxy_config.get('http'),
                'https': self.proxy_config.get('https')
            }
            response = requests.get('http://httpbin.org/ip', proxies=proxies, timeout=10)
            if response.status_code == 200:
                results['proxy_working'] = True
                print("✅ 代理连接正常")
                print(f"当前IP: {response.json().get('origin')}")
        except Exception as e:
            print(f"❌ 代理连接失败: {e}")
        
        # 测试AKShare API
        try:
            # 尝试获取简单数据
            df = ak.stock_zh_a_spot_em()
            if len(df) > 0:
                results['akshare_api'] = True
                print("✅ AKShare API正常")
        except Exception as e:
            print(f"❌ AKShare API失败: {e}")
        
        return results
    
    def get_alternative_stock_list(self) -> pd.DataFrame:
        """获取备用股票列表"""
        print("🔄 尝试获取备用股票列表...")
        
        # 方法1: 使用本地缓存
        cache_file = os.path.join(self.backup_data_dir, 'stock_list_cache.csv')
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                print(f"✅ 从缓存获取到 {len(df)} 只股票")
                return df
            except Exception as e:
                print(f"❌ 缓存读取失败: {e}")
        
        # 方法2: 手动构建常见股票列表
        common_stocks = [
            '000001', '000002', '000858', '000895', '000938',
            '002415', '002594', '002714', '300059', '300122',
            '600000', '600036', '600519', '600887', '601318',
            '601398', '601857', '601988', '603259', '603986'
        ]
        
        df = pd.DataFrame({
            'code': common_stocks,
            'name': [f'股票{code}' for code in common_stocks]
        })
        
        # 保存到缓存
        df.to_csv(cache_file, index=False)
        print(f"✅ 使用备用股票列表 {len(df)} 只股票")
        return df
    
    def download_with_retry_strategy(self, stock_code: str, period: int = 300) -> Optional[pd.DataFrame]:
        """使用重试策略下载单只股票数据"""
        retry_strategies = [
            {'delay': 1, 'method': 'normal'},
            {'delay': 3, 'method': 'with_random_delay'},
            {'delay': 5, 'method': 'with_longer_delay'},
            {'delay': 10, 'method': 'final_attempt'}
        ]
        
        for i, strategy in enumerate(retry_strategies):
            try:
                # 添加随机延迟避免频率限制
                if strategy['method'] == 'with_random_delay':
                    time.sleep(random.uniform(1, 3))
                else:
                    time.sleep(strategy['delay'])
                
                # 尝试下载数据
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=period)).strftime('%Y%m%d')
                
                df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                      start_date=start_date, end_date=end_date, adjust="qfq")
                
                if len(df) > 0:
                    print(f"✅ {stock_code} 下载成功 (尝试 {i+1}/{len(retry_strategies)})")
                    return df
                    
            except Exception as e:
                print(f"❌ {stock_code} 尝试 {i+1} 失败: {str(e)[:100]}")
                if i == len(retry_strategies) - 1:
                    print(f"❌ {stock_code} 所有重试策略失败")
        
        return None
    
    def use_cached_data(self) -> Dict[str, pd.DataFrame]:
        """使用缓存数据进行分析"""
        print("🔄 尝试使用缓存数据...")
        cached_data = {}
        
        # 查找所有缓存文件
        market_data_dir = 'market_data'
        if os.path.exists(market_data_dir):
            for file in os.listdir(market_data_dir):
                if file.endswith('.csv'):
                    stock_code = file.replace('.csv', '')
                    try:
                        df = pd.read_csv(os.path.join(market_data_dir, file))
                        cached_data[stock_code] = df
                    except Exception as e:
                        print(f"❌ 读取缓存文件 {file} 失败: {e}")
        
        print(f"✅ 找到 {len(cached_data)} 个缓存数据文件")
        return cached_data
    
    def generate_ip_restriction_report(self) -> str:
        """生成IP限制问题报告"""
        report = []
        report.append("=== IP限制问题诊断报告 ===")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 网络连接测试
        connectivity = self.test_network_connectivity()
        report.append("网络连接状态:")
        for key, status in connectivity.items():
            status_str = "✅ 正常" if status else "❌ 异常"
            report.append(f"  {key}: {status_str}")
        report.append("")
        
        # 缓存数据统计
        cached_data = self.use_cached_data()
        report.append(f"可用缓存数据: {len(cached_data)} 只股票")
        report.append("")
        
        # 解决方案建议
        report.append("建议解决方案:")
        if not connectivity['akshare_api']:
            report.append("1. 🔄 更换代理服务器")
            report.append("2. ⏰ 等待一段时间后重试 (建议1-2小时)")
            report.append("3. 📊 使用现有缓存数据进行分析")
            report.append("4. 🌐 考虑使用其他数据源")
        else:
            report.append("1. ✅ API正常，可以继续使用")
        
        report.append("")
        report.append("技术建议:")
        report.append("- 降低请求频率 (增加延迟到5-10秒)")
        report.append("- 使用更小的批次大小 (5-10只股票)")
        report.append("- 在非交易时间进行数据下载")
        report.append("- 定期备份已下载的数据")
        
        return "\n".join(report)
    
    def emergency_download_mode(self, stock_list: List[str], max_stocks: int = 20) -> Dict[str, pd.DataFrame]:
        """紧急下载模式 - 使用最保守的策略"""
        print(f"🚨 启动紧急下载模式，目标: {min(len(stock_list), max_stocks)} 只股票")
        
        successful_downloads = {}
        failed_downloads = []
        
        for i, stock_code in enumerate(stock_list[:max_stocks]):
            print(f"\n📊 下载进度: {i+1}/{min(len(stock_list), max_stocks)} - {stock_code}")
            
            # 超长延迟避免限制
            if i > 0:
                delay = random.uniform(10, 20)  # 10-20秒随机延迟
                print(f"⏰ 等待 {delay:.1f} 秒...")
                time.sleep(delay)
            
            df = self.download_with_retry_strategy(stock_code, period=100)  # 减少数据量
            
            if df is not None:
                successful_downloads[stock_code] = df
                # 立即保存到本地
                df.to_csv(f'market_data/{stock_code}.csv', index=False)
                print(f"💾 {stock_code} 数据已保存")
            else:
                failed_downloads.append(stock_code)
            
            # 每5只股票报告一次进度
            if (i + 1) % 5 == 0:
                success_rate = len(successful_downloads) / (i + 1) * 100
                print(f"\n📈 阶段性报告: 成功率 {success_rate:.1f}% ({len(successful_downloads)}/{i+1})")
                
                # 如果成功率太低，建议停止
                if success_rate < 20 and i > 10:
                    print("⚠️  成功率过低，建议停止下载并检查网络状况")
                    break
        
        print(f"\n=== 紧急下载完成 ===")
        print(f"✅ 成功: {len(successful_downloads)} 只股票")
        print(f"❌ 失败: {len(failed_downloads)} 只股票")
        
        if failed_downloads:
            print(f"失败股票: {failed_downloads[:10]}{'...' if len(failed_downloads) > 10 else ''}")
        
        return successful_downloads

def main():
    """主函数 - IP限制解决方案"""
    print("=== IP限制问题解决方案 ===")
    
    solver = IPRestrictionSolver()
    
    print("\n请选择操作:")
    print("1. 网络连接诊断")
    print("2. 生成问题报告")
    print("3. 使用缓存数据分析")
    print("4. 紧急下载模式 (保守策略)")
    print("5. 查看解决建议")
    
    try:
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == '1':
            print("\n🔍 开始网络连接诊断...")
            connectivity = solver.test_network_connectivity()
            print("\n诊断完成!")
            
        elif choice == '2':
            print("\n📋 生成问题报告...")
            report = solver.generate_ip_restriction_report()
            print(report)
            
            # 保存报告
            with open('ip_restriction_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            print("\n💾 报告已保存到 ip_restriction_report.txt")
            
        elif choice == '3':
            print("\n📊 使用缓存数据...")
            cached_data = solver.use_cached_data()
            if cached_data:
                print(f"✅ 可以使用 {len(cached_data)} 只股票的缓存数据进行分析")
                print("建议运行: python optimized_rps_calculator.py")
            else:
                print("❌ 没有找到可用的缓存数据")
                
        elif choice == '4':
            print("\n🚨 启动紧急下载模式...")
            stock_list = solver.get_alternative_stock_list()['code'].tolist()
            max_stocks = int(input("请输入最大下载数量 (建议10-20): ") or "10")
            
            os.makedirs('market_data', exist_ok=True)
            successful_data = solver.emergency_download_mode(stock_list, max_stocks)
            
            if successful_data:
                print(f"\n✅ 紧急下载完成，获得 {len(successful_data)} 只股票数据")
                print("可以继续进行RPS计算和选股分析")
            else:
                print("\n❌ 紧急下载失败，建议稍后重试")
                
        elif choice == '5':
            print("\n💡 IP限制解决建议:")
            print("")
            print("立即可行的解决方案:")
            print("1. 🔄 更换代理服务器或VPN")
            print("2. ⏰ 等待1-2小时后重试")
            print("3. 📊 使用现有缓存数据")
            print("4. 🌙 在非交易时间下载 (晚上或周末)")
            print("")
            print("长期解决方案:")
            print("1. 🏢 申请专业数据接口")
            print("2. 💰 购买付费数据服务")
            print("3. 🔄 建立多个数据源备份")
            print("4. ⚡ 使用分布式下载策略")
            
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n❌ 操作失败: {e}")

if __name__ == "__main__":
    main()