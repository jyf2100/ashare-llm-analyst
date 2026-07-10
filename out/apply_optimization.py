#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用策略优化参数
将优化后的参数应用到实际的策略配置中
"""

import yaml
import json
import shutil
from datetime import datetime
from typing import Dict, Any
import os

class OptimizationApplier:
    """优化参数应用器"""
    
    def __init__(self, 
                 optimized_config_file: str = "optimized_config.yaml",
                 target_strategy_file: str = "-02-optimized_stock_selection_v2.py"):
        self.optimized_config_file = optimized_config_file
        self.target_strategy_file = target_strategy_file
        self.backup_suffix = f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def load_optimized_config(self) -> Dict[str, Any]:
        """加载优化配置"""
        try:
            with open(self.optimized_config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到优化配置文件: {self.optimized_config_file}")
    
    def backup_original_file(self) -> str:
        """备份原始文件"""
        backup_file = self.target_strategy_file + self.backup_suffix
        shutil.copy2(self.target_strategy_file, backup_file)
        print(f"✅ 原始文件已备份至: {backup_file}")
        return backup_file
    
    def generate_updated_config_section(self, optimized_config: Dict[str, Any]) -> str:
        """生成更新后的配置代码段"""
        strategies = optimized_config.get('strategies', {})
        
        config_code = []
        config_code.append("        return {")
        config_code.append("            'strategies': {")
        
        for strategy_name, strategy_config in strategies.items():
            params = strategy_config.get('parameters', {})
            weight = strategy_config.get('weight', 0.25)
            
            config_code.append(f"                '{strategy_name}': {{")
            config_code.append(f"                    'weight': {weight:.3f},")
            config_code.append("                    'parameters': {")
            
            for param_name, param_value in params.items():
                if isinstance(param_value, str):
                    config_code.append(f"                        '{param_name}': '{param_value}',")
                elif isinstance(param_value, bool):
                    config_code.append(f"                        '{param_name}': {str(param_value)},")
                else:
                    config_code.append(f"                        '{param_name}': {param_value},")
            
            config_code.append("                    }")
            config_code.append("                },")
        
        config_code.append("            },")
        
        # 添加其他配置部分
        config_code.extend([
            "            'performance': {",
            "                'max_workers': 4,",
            "                'chunk_size': 100,",
            "                'use_parallel': True,",
            "                'cache_enabled': True",
            "            },",
            "            'analysis': {",
            "                'analysis_days': 30,",
            "                'min_selection_rate': 20,",
            "                'top_n_stocks': 20",
            "            },",
            "            'risk': {",
            "                'max_correlation': 0.8,",
            "                'max_drawdown_threshold': 0.2,",
            "                'var_confidence': 0.05",
            "            }",
            "        }"
        ])
        
        return "\n".join(config_code)
    
    def update_strategy_file(self, optimized_config: Dict[str, Any]) -> bool:
        """更新策略文件"""
        try:
            # 读取原始文件
            with open(self.target_strategy_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 生成新的配置代码
            new_config_code = self.generate_updated_config_section(optimized_config)
            
            # 查找并替换_load_default_config方法中的return语句
            import re
            
            # 匹配_load_default_config方法中的return字典
            pattern = r'(def _load_default_config\(self\) -> Dict\[str, Any\]:\s*"""加载默认配置"""\s*return )\{[^}]*(?:\{[^}]*\}[^}]*)*\}'
            
            # 查找匹配位置
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                print("❌ 未找到_load_default_config方法，尝试手动定位...")
                return self._manual_update_strategy_file(content, new_config_code)
            
            # 替换配置
            new_content = content[:match.start()] + match.group(1) + new_config_code.replace('        return ', '') + content[match.end():]
            
            # 写入更新后的文件
            with open(self.target_strategy_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print("✅ 策略文件已成功更新")
            return True
            
        except Exception as e:
            print(f"❌ 更新策略文件失败: {e}")
            return False
    
    def _manual_update_strategy_file(self, content: str, new_config_code: str) -> bool:
        """手动更新策略文件（备用方法）"""
        try:
            # 查找_load_default_config方法的开始
            start_marker = "def _load_default_config(self) -> Dict[str, Any]:"
            start_pos = content.find(start_marker)
            
            if start_pos == -1:
                print("❌ 无法找到_load_default_config方法")
                return False
            
            # 查找return语句
            return_pos = content.find("return {", start_pos)
            if return_pos == -1:
                print("❌ 无法找到return语句")
                return False
            
            # 查找对应的结束大括号
            brace_count = 0
            end_pos = return_pos + 7  # "return {"的长度
            
            while end_pos < len(content):
                if content[end_pos] == '{':
                    brace_count += 1
                elif content[end_pos] == '}':
                    if brace_count == 0:
                        end_pos += 1
                        break
                    brace_count -= 1
                end_pos += 1
            
            # 替换配置部分
            new_content = (content[:return_pos] + 
                          "return " + new_config_code.replace('        return ', '') + 
                          content[end_pos:])
            
            # 写入文件
            with open(self.target_strategy_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print("✅ 策略文件已手动更新")
            return True
            
        except Exception as e:
            print(f"❌ 手动更新失败: {e}")
            return False
    
    def create_config_yaml(self, optimized_config: Dict[str, Any]) -> str:
        """创建独立的配置YAML文件"""
        config_file = "config.yaml"
        
        # 提取策略配置
        yaml_config = {
            'strategies': optimized_config.get('strategies', {}),
            'performance': {
                'max_workers': 4,
                'chunk_size': 100,
                'use_parallel': True,
                'cache_enabled': True
            },
            'analysis': {
                'analysis_days': 30,
                'min_selection_rate': 20,
                'top_n_stocks': 20
            },
            'risk': {
                'max_correlation': 0.8,
                'max_drawdown_threshold': 0.2,
                'var_confidence': 0.05
            },
            'optimization_info': optimized_config.get('optimization_info', {})
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 配置文件已创建: {config_file}")
        return config_file
    
    def validate_optimization(self) -> bool:
        """验证优化效果"""
        print("\n🔍 验证优化效果...")
        
        try:
            # 尝试导入更新后的模块
            import importlib.util
            spec = importlib.util.spec_from_file_location("optimized_strategy", self.target_strategy_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 检查ConfigManager是否能正常初始化
            config_manager = module.ConfigManager()
            strategies_config = config_manager.get('strategies', {})
            
            print("✅ 策略配置验证通过")
            print(f"   - 策略数量: {len(strategies_config)}")
            
            for strategy_name, config in strategies_config.items():
                weight = config.get('weight', 0)
                param_count = len(config.get('parameters', {}))
                print(f"   - {strategy_name}: 权重={weight:.3f}, 参数={param_count}个")
            
            return True
            
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return False
    
    def apply_optimization(self) -> bool:
        """应用优化"""
        print("🚀 开始应用策略优化...")
        print("="*40)
        
        try:
            # 1. 加载优化配置
            print("1. 加载优化配置...")
            optimized_config = self.load_optimized_config()
            print(f"   ✅ 已加载 {len(optimized_config.get('strategies', {}))} 个策略配置")
            
            # 2. 备份原始文件
            print("\n2. 备份原始文件...")
            backup_file = self.backup_original_file()
            
            # 3. 创建独立配置文件
            print("\n3. 创建配置文件...")
            config_file = self.create_config_yaml(optimized_config)
            
            # 4. 更新策略文件
            print("\n4. 更新策略文件...")
            if not self.update_strategy_file(optimized_config):
                print("⚠️  策略文件更新失败，但配置文件已创建")
                print(f"   可以手动使用配置文件: {config_file}")
                return False
            
            # 5. 验证更新
            print("\n5. 验证更新...")
            if not self.validate_optimization():
                print("⚠️  验证失败，建议检查更新内容")
                return False
            
            print("\n" + "="*40)
            print("✅ 优化应用完成!")
            print(f"\n📁 文件状态:")
            print(f"   - 原始备份: {backup_file}")
            print(f"   - 更新文件: {self.target_strategy_file}")
            print(f"   - 配置文件: {config_file}")
            
            print(f"\n📊 优化摘要:")
            strategies = optimized_config.get('strategies', {})
            for name, config in strategies.items():
                improvement = config.get('optimization', {}).get('improvement_score', 0)
                if improvement > 0:
                    print(f"   - {name}: 改进评分 {improvement:.1f}/1.0")
            
            return True
            
        except Exception as e:
            print(f"❌ 应用优化失败: {e}")
            return False
    
    def rollback(self, backup_file: str = None) -> bool:
        """回滚到备份版本"""
        if not backup_file:
            # 查找最新的备份文件
            import glob
            backup_files = glob.glob(f"{self.target_strategy_file}_backup_*")
            if not backup_files:
                print("❌ 未找到备份文件")
                return False
            backup_file = max(backup_files)  # 最新的备份
        
        try:
            shutil.copy2(backup_file, self.target_strategy_file)
            print(f"✅ 已回滚到备份版本: {backup_file}")
            return True
        except Exception as e:
            print(f"❌ 回滚失败: {e}")
            return False

def main():
    """主函数"""
    print("⚙️  策略优化应用工具")
    print("="*30)
    
    applier = OptimizationApplier()
    
    # 检查优化配置文件是否存在
    if not os.path.exists(applier.optimized_config_file):
        print(f"❌ 找不到优化配置文件: {applier.optimized_config_file}")
        print("请先运行 strategy_optimization.py 生成优化配置")
        return
    
    # 应用优化
    success = applier.apply_optimization()
    
    if success:
        print("\n🎯 下一步建议:")
        print("1. 运行测试验证优化效果")
        print("2. 进行小规模回测")
        print("3. 监控选股质量变化")
        print("4. 如有问题可使用备份文件回滚")
    else:
        print("\n⚠️  应用失败，请检查错误信息")
        print("可以尝试手动应用配置文件中的参数")

if __name__ == "__main__":
    main()