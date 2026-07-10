#!/usr/bin/env python
"""
测试运行器

运行所有数据管道单元测试
"""

import os
import sys
import subprocess

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("运行数据管道单元测试")
    print("=" * 70)

    test_files = [
        "test_providers.py",
        "test_downloaders.py",
        "test_rps_calculator.py",
        "test_feature_engineering.py",
        "test_model_trainer.py",
        "test_pipeline.py",
    ]

    total_passed = 0
    total_failed = 0

    for test_file in test_files:
        print(f"\n运行 {test_file}...")
        print("-" * 70)

        result = subprocess.run(
            ["python", test_file],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print(f"✅ {test_file} 通过")
            total_passed += 1
        else:
            print(f"❌ {test_file} 失败")
            total_failed += 1

    # 打印汇总
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    print(f"总计: {len(test_files)} 个测试文件")
    print(f"通过: {total_passed}")
    print(f"失败: {total_failed}")

    if total_failed == 0:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print(f"\n⚠️  {total_failed} 个测试文件失败")
        return 1


def run_specific_test(test_name):
    """运行指定测试"""
    test_file = f"test_{test_name}.py"
    test_path = os.path.join(os.path.dirname(__file__), test_file)

    if not os.path.exists(test_path):
        print(f"测试文件不存在: {test_file}")
        return 1

    print(f"运行 {test_file}...")
    result = subprocess.run(["python", test_path])

    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 运行指定测试
        test_name = sys.argv[1]
        sys.exit(run_specific_test(test_name))
    else:
        # 运行所有测试
        sys.exit(run_all_tests())
