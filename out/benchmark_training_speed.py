#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练速度基准测试脚本
用于比较原始训练脚本和优化版本的性能差异
"""

import os
import time
import sys
import subprocess
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def run_command_with_timeout(command, timeout=3600):
    """运行命令并记录执行时间"""
    print(f"执行命令: {command}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'success': result.returncode == 0,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'duration': timeout,
            'stdout': '',
            'stderr': 'Command timed out'
        }
    except Exception as e:
        return {
            'success': False,
            'duration': 0,
            'stdout': '',
            'stderr': str(e)
        }

def get_system_info():
    """获取系统信息"""
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'python_version': platform.python_version()
    }

def check_training_data():
    """检查训练数据是否存在"""
    training_data_dir = "training_data"
    
    if not os.path.exists(training_data_dir):
        print(f"❌ 训练数据目录不存在: {training_data_dir}")
        return False
    
    # 检查特征文件
    feature_files = [f for f in os.listdir(training_data_dir) if f.startswith('features_') and f.endswith('.npy')]
    if not feature_files:
        print(f"❌ 未找到特征文件")
        return False
    
    # 检查标签文件
    label_files = [f for f in os.listdir(training_data_dir) if f.startswith('labels_') and f.endswith('.npy')]
    if not label_files:
        print(f"❌ 未找到标签文件")
        return False
    
    print(f"✅ 找到 {len(feature_files)} 个特征文件和 {len(label_files)} 个标签文件")
    return True

def create_test_data():
    """创建测试数据（如果不存在真实数据）"""
    print("创建测试数据...")
    
    training_data_dir = "training_data"
    os.makedirs(training_data_dir, exist_ok=True)
    
    # 创建模拟数据
    n_samples = 10000
    n_features = 50
    
    # 生成特征数据
    X = np.random.randn(n_samples, n_features)
    timestamp = datetime.now().strftime('%Y%m%d')
    
    feature_file = os.path.join(training_data_dir, f'features_{timestamp}.npy')
    np.save(feature_file, X)
    
    # 生成标签数据
    labels = {
        'return_5d_gt_5pct': np.random.binomial(1, 0.3, n_samples),
        'return_10d_gt_5pct': np.random.binomial(1, 0.25, n_samples),
        'multi_class_5d': np.random.randint(0, 3, n_samples)
    }
    
    for label_name, y in labels.items():
        label_file = os.path.join(training_data_dir, f'labels_{label_name}_{timestamp}.npy')
        np.save(label_file, y)
    
    # 创建配置文件
    import yaml
    config = {
        'feature_columns': [f'feature_{i}' for i in range(n_features)],
        'label_columns': list(labels.keys())
    }
    
    config_file = os.path.join(training_data_dir, f'config_{timestamp}.yaml')
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 测试数据创建完成: {n_samples} 样本, {n_features} 特征")

def run_benchmark():
    """运行基准测试"""
    print("="*60)
    print("机器学习训练速度基准测试")
    print("="*60)
    
    # 获取系统信息
    system_info = get_system_info()
    print(f"\n系统信息:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # 检查训练数据
    print(f"\n检查训练数据...")
    if not check_training_data():
        print("创建测试数据...")
        create_test_data()
    
    # 测试配置
    test_configs = [
        {
            'name': '原始脚本',
            'command': 'python 02.02-train_ml_model.py',
            'description': '原始训练脚本（如果存在）'
        },
        {
            'name': '优化版-快速模式',
            'command': 'python 02.02-train_ml_model_optimized.py --fast',
            'description': '优化版本，快速模式 + 并行训练'
        },
        {
            'name': '优化版-标准模式',
            'command': 'python 02.02-train_ml_model_optimized.py --standard',
            'description': '优化版本，标准模式 + 并行训练'
        },
        {
            'name': '优化版-快速串行',
            'command': 'python 02.02-train_ml_model_optimized.py --fast-serial',
            'description': '优化版本，快速模式 + 串行训练'
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*40}")
        print(f"测试: {config['name']}")
        print(f"描述: {config['description']}")
        print(f"{'='*40}")
        
        # 检查脚本是否存在
        script_name = config['command'].split()[1]
        if not os.path.exists(script_name):
            print(f"❌ 脚本不存在: {script_name}")
            results.append({
                'name': config['name'],
                'success': False,
                'duration': 0,
                'error': 'Script not found'
            })
            continue
        
        # 运行测试
        result = run_command_with_timeout(config['command'], timeout=1800)  # 30分钟超时
        
        if result['success']:
            print(f"✅ 测试完成")
            print(f"⏱️  执行时间: {result['duration']:.2f} 秒 ({result['duration']/60:.2f} 分钟)")
        else:
            print(f"❌ 测试失败")
            print(f"错误信息: {result['stderr'][:200]}...")
        
        results.append({
            'name': config['name'],
            'success': result['success'],
            'duration': result['duration'],
            'duration_minutes': result['duration'] / 60,
            'error': result['stderr'] if not result['success'] else None
        })
    
    # 生成报告
    generate_report(results, system_info)

def generate_report(results, system_info):
    """生成测试报告"""
    print(f"\n{'='*60}")
    print("基准测试报告")
    print(f"{'='*60}")
    
    # 创建结果表格
    df = pd.DataFrame(results)
    successful_results = df[df['success'] == True]
    
    if len(successful_results) > 0:
        print(f"\n✅ 成功的测试:")
        print(f"{'测试名称':<20} {'执行时间(秒)':<15} {'执行时间(分钟)':<15} {'相对速度':<10}")
        print("-" * 70)
        
        # 计算相对速度（以最慢的为基准）
        max_duration = successful_results['duration'].max()
        
        for _, row in successful_results.iterrows():
            relative_speed = max_duration / row['duration'] if row['duration'] > 0 else 0
            print(f"{row['name']:<20} {row['duration']:<15.2f} {row['duration_minutes']:<15.2f} {relative_speed:<10.2f}x")
        
        # 速度提升分析
        if len(successful_results) > 1:
            fastest = successful_results.loc[successful_results['duration'].idxmin()]
            slowest = successful_results.loc[successful_results['duration'].idxmax()]
            speedup = slowest['duration'] / fastest['duration']
            
            print(f"\n📊 性能分析:")
            print(f"  最快: {fastest['name']} ({fastest['duration']:.2f}秒)")
            print(f"  最慢: {slowest['name']} ({slowest['duration']:.2f}秒)")
            print(f"  最大速度提升: {speedup:.2f}x")
    
    # 失败的测试
    failed_results = df[df['success'] == False]
    if len(failed_results) > 0:
        print(f"\n❌ 失败的测试:")
        for _, row in failed_results.iterrows():
            print(f"  {row['name']}: {row['error']}")
    
    # 保存详细报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'benchmark_report_{timestamp}.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("机器学习训练速度基准测试报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("系统信息:\n")
        for key, value in system_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("测试结果:\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        if len(successful_results) > 1:
            f.write("性能分析:\n")
            fastest = successful_results.loc[successful_results['duration'].idxmin()]
            slowest = successful_results.loc[successful_results['duration'].idxmax()]
            speedup = slowest['duration'] / fastest['duration']
            f.write(f"  最快: {fastest['name']} ({fastest['duration']:.2f}秒)\n")
            f.write(f"  最慢: {slowest['name']} ({slowest['duration']:.2f}秒)\n")
            f.write(f"  最大速度提升: {speedup:.2f}x\n")
    
    print(f"\n📄 详细报告已保存到: {report_file}")

def run_single_test(script_name, mode=None):
    """运行单个测试"""
    if mode:
        command = f"python {script_name} --{mode}"
    else:
        command = f"python {script_name}"
    
    print(f"运行单个测试: {command}")
    
    if not os.path.exists(script_name):
        print(f"❌ 脚本不存在: {script_name}")
        return
    
    if not check_training_data():
        print("创建测试数据...")
        create_test_data()
    
    result = run_command_with_timeout(command, timeout=1800)
    
    if result['success']:
        print(f"✅ 测试完成")
        print(f"⏱️  执行时间: {result['duration']:.2f} 秒 ({result['duration']/60:.2f} 分钟)")
    else:
        print(f"❌ 测试失败")
        print(f"错误信息: {result['stderr']}")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("训练速度基准测试工具")
            print("\n用法:")
            print("  python benchmark_training_speed.py                    # 运行完整基准测试")
            print("  python benchmark_training_speed.py --single original  # 测试原始脚本")
            print("  python benchmark_training_speed.py --single optimized fast  # 测试优化版快速模式")
            print("  python benchmark_training_speed.py --create-data      # 只创建测试数据")
            return
        
        elif sys.argv[1] == "--single":
            if len(sys.argv) < 3:
                print("❌ 请指定脚本类型: original 或 optimized")
                return
            
            script_type = sys.argv[2]
            mode = sys.argv[3] if len(sys.argv) > 3 else None
            
            if script_type == "original":
                run_single_test("02.02-train_ml_model.py")
            elif script_type == "optimized":
                run_single_test("02.02-train_ml_model_optimized.py", mode)
            else:
                print("❌ 无效的脚本类型，请使用 'original' 或 'optimized'")
            return
        
        elif sys.argv[1] == "--create-data":
            create_test_data()
            return
    
    # 运行完整基准测试
    run_benchmark()

if __name__ == "__main__":
    main()