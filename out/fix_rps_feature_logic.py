#!/usr/bin/env python3
"""
修复训练数据生成器中的RPS特征添加逻辑

问题：当前代码检查 self.rps_data[stock_code]['rps']，但实际RPS数据结构是
直接包含日期和RPS值的列表，不是包含'rps'键的字典。

修复：
1. 修正RPS数据访问逻辑
2. 正确处理RPS时间序列数据
3. 添加调试日志
"""

import os
import shutil
from datetime import datetime

def fix_rps_feature_logic():
    """修复RPS特征添加逻辑"""
    
    file_path = '02.01-generate_ml_training_data.py'
    
    # 备份原文件
    backup_path = f'{file_path}.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy2(file_path, backup_path)
    print(f"✅ 已备份原文件到: {backup_path}")
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复RPS特征添加逻辑
    old_rps_logic = '''                # 添加RPS特征
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
    
    new_rps_logic = '''                # 添加RPS特征
                if stock_code in self.rps_data:
                    # RPS数据是直接的时间序列，不需要访问'rps'键
                    rps_series = self.rps_data[stock_code]
                    rps_values = data_with_features.index.map(
                        lambda x: self._get_rps_value(rps_series, x)
                    )
                    # 确保所有RPS值都是有效的
                    data_with_features['rps'] = rps_values.fillna(50.0)
                    # 替换任何异常值
                    data_with_features['rps'] = data_with_features['rps'].apply(
                        lambda x: 50.0 if x < 0 or x > 100 else x
                    )
                    logger.info(f"股票 {stock_code}: 成功添加RPS特征，平均值={data_with_features['rps'].mean():.2f}")
                else:
                    data_with_features['rps'] = 50.0  # 默认值
                    logger.warning(f"股票 {stock_code}: 未找到RPS数据，使用默认值50.0")'''
    
    # 执行替换
    if old_rps_logic in content:
        content = content.replace(old_rps_logic, new_rps_logic)
        print("✅ 已修复RPS特征添加逻辑")
    else:
        print("⚠️  未找到预期的RPS逻辑代码，可能已经被修改")
        return False
    
    # 写入修复后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n🔧 修复内容:")
    print("1. 移除了对'rps'键的检查")
    print("2. 直接使用self.rps_data[stock_code]作为RPS时间序列")
    print("3. 添加了详细的调试日志")
    print("4. 保持了原有的数据验证和默认值逻辑")
    
    print("\n📋 后续步骤:")
    print("1. 重新运行训练数据生成: python 02.01-generate_ml_training_data.py")
    print("2. 检查RPS数据质量: python debug_rps_data.py")
    print("3. 如果问题仍然存在，检查RPS数据结构")
    
    return True

if __name__ == "__main__":
    print("RPS特征逻辑修复工具")
    print("=" * 40)
    
    if fix_rps_feature_logic():
        print("\n✅ RPS特征逻辑修复完成!")
    else:
        print("\n❌ RPS特征逻辑修复失败!")