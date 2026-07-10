#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复RPS数据加载逻辑
"""

import os
import shutil

def fix_rps_data_loading():
    """修复RPS数据加载逻辑"""
    print("🔧 修复RPS数据加载逻辑")
    print("=" * 50)
    
    # 备份原文件
    original_file = "02.01-generate_ml_training_data.py"
    backup_file = "02.01-generate_ml_training_data.py.backup2"
    
    if os.path.exists(original_file):
        shutil.copy2(original_file, backup_file)
        print(f"✅ 已备份原文件到: {backup_file}")
    
    # 读取原文件内容
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换load_rps_data方法
    old_load_rps_method = '''    def load_rps_data(self):
        """加载RPS数据"""
        if not os.path.exists(self.rps_data_dir):
            logger.warning(f"RPS数据目录不存在: {self.rps_data_dir}")
            return False
        
        rps_files = [f for f in os.listdir(self.rps_data_dir) if f.endswith('.pkl')]
        if not rps_files:
            logger.warning(f"RPS数据目录中没有.pkl文件: {self.rps_data_dir}")
            return False
        
        self.rps_data = {}
        
        for rps_file in rps_files:
            try:
                file_path = os.path.join(self.rps_data_dir, rps_file)
                with open(file_path, 'rb') as f:
                    raw_rps_data = pickle.load(f)
                
                # 处理RPS数据结构
                for stock_code, rps_list in raw_rps_data.items():
                    if stock_code not in self.rps_data:
                        self.rps_data[stock_code] = {}
                    
                    if isinstance(rps_list, list) and len(rps_list) > 0:
                        rps_dict = {}
                        for item in rps_list:
                            if isinstance(item, dict) and 'date' in item and 'rps' in item:
                                rps_dict[item['date']] = float(item['rps'])
                        
                        if rps_dict:
                            rps_series = pd.Series(rps_dict)
                            rps_series.index = pd.to_datetime(rps_series.index)
                            self.rps_data[stock_code]['rps'] = rps_series.sort_index()
                
                logger.info(f"加载RPS文件: {rps_file}")
                break  # 只加载第一个有效文件
                
            except Exception as e:
                logger.warning(f"加载RPS文件 {rps_file} 失败: {e}")
        
        return len(self.rps_data) > 0'''
    
    new_load_rps_method = '''    def load_rps_data(self, rps_period='RPS20'):
        """加载RPS数据
        
        Args:
            rps_period: RPS周期，可选 'RPS20', 'RPS60', 'RPS120'
        """
        if not os.path.exists(self.rps_data_dir):
            logger.warning(f"RPS数据目录不存在: {self.rps_data_dir}")
            return False
        
        rps_files = [f for f in os.listdir(self.rps_data_dir) if f.endswith('.pkl')]
        if not rps_files:
            logger.warning(f"RPS数据目录中没有.pkl文件: {self.rps_data_dir}")
            return False
        
        self.rps_data = {}
        
        for rps_file in rps_files:
            try:
                file_path = os.path.join(self.rps_data_dir, rps_file)
                with open(file_path, 'rb') as f:
                    raw_rps_data = pickle.load(f)
                
                logger.info(f"加载RPS文件: {rps_file}")
                logger.info(f"RPS文件顶级键: {list(raw_rps_data.keys())}")
                
                # 检查是否是多周期结构
                if rps_period in raw_rps_data:
                    period_data = raw_rps_data[rps_period]
                    logger.info(f"使用RPS周期: {rps_period}")
                    logger.info(f"该周期包含 {len(period_data)} 只股票")
                    
                    # 处理该周期的RPS数据
                    for stock_code, rps_list in period_data.items():
                        if stock_code not in self.rps_data:
                            self.rps_data[stock_code] = {}
                        
                        if isinstance(rps_list, list) and len(rps_list) > 0:
                            rps_dict = {}
                            for item in rps_list:
                                if isinstance(item, dict) and 'date' in item and 'rps' in item:
                                    rps_dict[item['date']] = float(item['rps'])
                            
                            if rps_dict:
                                rps_series = pd.Series(rps_dict)
                                rps_series.index = pd.to_datetime(rps_series.index)
                                self.rps_data[stock_code]['rps'] = rps_series.sort_index()
                    
                    logger.info(f"成功加载 {len(self.rps_data)} 只股票的RPS数据")
                    break  # 成功加载后退出
                    
                else:
                    # 如果不是多周期结构，尝试原来的处理方式
                    logger.warning(f"未找到RPS周期 {rps_period}，尝试原始格式")
                    for stock_code, rps_list in raw_rps_data.items():
                        if stock_code not in self.rps_data:
                            self.rps_data[stock_code] = {}
                        
                        if isinstance(rps_list, list) and len(rps_list) > 0:
                            rps_dict = {}
                            for item in rps_list:
                                if isinstance(item, dict) and 'date' in item and 'rps' in item:
                                    rps_dict[item['date']] = float(item['rps'])
                            
                            if rps_dict:
                                rps_series = pd.Series(rps_dict)
                                rps_series.index = pd.to_datetime(rps_series.index)
                                self.rps_data[stock_code]['rps'] = rps_series.sort_index()
                    break
                
            except Exception as e:
                logger.warning(f"加载RPS文件 {rps_file} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        success = len(self.rps_data) > 0
        if success:
            logger.info(f"RPS数据加载成功，共 {len(self.rps_data)} 只股票")
            # 显示一些统计信息
            sample_stocks = list(self.rps_data.keys())[:3]
            for stock in sample_stocks:
                if 'rps' in self.rps_data[stock]:
                    rps_series = self.rps_data[stock]['rps']
                    logger.info(f"  {stock}: {len(rps_series)} 个RPS值, 范围 {rps_series.min():.2f}-{rps_series.max():.2f}")
        else:
            logger.error("RPS数据加载失败")
        
        return success'''
    
    # 应用修复
    content = content.replace(old_load_rps_method, new_load_rps_method)
    
    # 写入修复后的文件
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ RPS数据加载逻辑修复完成")
    print("\n修复内容:")
    print("1. 支持多周期RPS数据结构 (RPS20, RPS60, RPS120)")
    print("2. 默认使用RPS20周期")
    print("3. 增加详细的日志输出")
    print("4. 增加错误处理和统计信息")
    print("5. 向后兼容原始RPS数据格式")
    
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
    fix_rps_data_loading()
    test_fix()