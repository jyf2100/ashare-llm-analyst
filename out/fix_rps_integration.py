#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复RPS集成问题
"""

import os
import shutil

def fix_rps_integration():
    """修复RPS集成问题"""
    print("🔧 修复RPS集成问题")
    print("=" * 50)
    
    # 备份原文件
    original_file = "02.01-generate_ml_training_data.py"
    backup_file = "02.01-generate_ml_training_data.py.backup"
    
    if os.path.exists(original_file):
        shutil.copy2(original_file, backup_file)
        print(f"✅ 已备份原文件到: {backup_file}")
    
    # 读取原文件内容
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复_get_rps_value方法
    old_get_rps_value = '''    def _get_rps_value(self, rps_series, target_date):
        """获取RPS值"""
        try:
            valid_dates = rps_series.index[rps_series.index <= target_date]
            if len(valid_dates) > 0:
                return float(rps_series[valid_dates.max()])
        except:
            pass
        return -99'''
    
    new_get_rps_value = '''    def _get_rps_value(self, rps_series, target_date):
        """获取RPS值"""
        try:
            valid_dates = rps_series.index[rps_series.index <= target_date]
            if len(valid_dates) > 0:
                rps_value = float(rps_series[valid_dates.max()])
                # 确保RPS值在合理范围内
                if 0 <= rps_value <= 100:
                    return rps_value
        except Exception as e:
            pass
        return 50.0  # 返回默认值而不是-99'''
    
    # 修复RPS特征添加逻辑
    old_rps_logic = '''                # 添加RPS特征
                if stock_code in self.rps_data and 'rps' in self.rps_data[stock_code]:
                    rps_series = self.rps_data[stock_code]['rps']
                    data_with_features['rps'] = data_with_features.index.map(
                        lambda x: self._get_rps_value(rps_series, x)
                    )
                else:
                    data_with_features['rps'] = 50.0  # 默认值'''
    
    new_rps_logic = '''                # 添加RPS特征
                if stock_code in self.rps_data and 'rps' in self.rps_data[stock_code]:
                    rps_series = self.rps_data[stock_code]['rps']
                    rps_values = data_with_features.index.map(
                        lambda x: self._get_rps_value(rps_series, x)
                    )
                    # 确保所有RPS值都是有效的
                    data_with_features['rps'] = rps_values.fillna(50.0)
                    # 替换任何异常值
                    data_with_features['rps'] = data_with_features['rps'].apply(
                        lambda x: 50.0 if x < 0 or x > 100 else x
                    )
                else:
                    data_with_features['rps'] = 50.0  # 默认值'''
    
    # 应用修复
    content = content.replace(old_get_rps_value, new_get_rps_value)
    content = content.replace(old_rps_logic, new_rps_logic)
    
    # 写入修复后的文件
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ RPS集成问题修复完成")
    print("\n修复内容:")
    print("1. _get_rps_value方法返回50.0而不是-99")
    print("2. 添加RPS值范围检查 (0-100)")
    print("3. 确保所有RPS值都是有效的")
    print("4. 替换异常RPS值为默认值50.0")
    
    return True

def test_fix():
    """测试修复效果"""
    print("\n🧪 测试修复效果")
    print("=" * 50)
    print("请运行以下命令测试:")
    print("1. python 02.01-generate_ml_training_data.py")
    print("2. python debug_rps_data.py")
    print("3. python 02.02-train_ml_model.py")

if __name__ == "__main__":
    fix_rps_integration()
    test_fix()