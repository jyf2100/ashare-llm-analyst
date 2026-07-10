#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理independent_selection_results.json文件，提取预测收益率高的股票
并更新.env文件中的STOCKS_CONFIG配置
"""

import json
import os
import time
from typing import Dict, List, Tuple

def load_json_file(file_path: str) -> dict:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return {}

def extract_high_return_stocks(data: dict, min_return: float = 5.0, min_confidence: float = 0.3) -> List[Tuple[str, float, float]]:
    """提取预测收益率和置信度都高的股票
    
    Args:
        data: JSON数据
        min_return: 最小预测收益率阈值
        min_confidence: 最小置信度阈值
    
    Returns:
        List of (stock_code, predicted_return, confidence)
    """
    ml_predictions = data.get('step3_ml_predictions', {})
    high_return_stocks = []
    
    for stock_code, prediction in ml_predictions.items():
        predicted_return = prediction.get('predicted_return_5d', 0)
        confidence = prediction.get('confidence', 0)
        
        if predicted_return >= min_return and confidence >= min_confidence:
            high_return_stocks.append((stock_code, predicted_return, confidence))
    
    # 按预测收益率降序排序
    high_return_stocks.sort(key=lambda x: x[1], reverse=True)
    return high_return_stocks

def load_stock_mapping_from_file(cache_file: str = 'stock_mapping_cache.json') -> Dict[str, str]:
    """从缓存文件加载股票映射"""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # 检查缓存是否过期（1天）
            cache_time = cache_data.get('timestamp', 0)
            current_time = time.time()
            if current_time - cache_time < 24 * 3600:  # 24小时内有效
                print(f"从缓存文件加载股票映射，共 {len(cache_data['mapping'])} 个")
                return cache_data['mapping']
            else:
                print("缓存文件已过期，需要重新获取")
        else:
            print("缓存文件不存在，需要重新获取")
    except Exception as e:
        print(f"读取缓存文件失败: {e}")
    
    return {}

def save_stock_mapping_to_file(mapping: Dict[str, str], cache_file: str = 'stock_mapping_cache.json'):
    """保存股票映射到缓存文件"""
    try:
        cache_data = {
            'timestamp': time.time(),
            'mapping': mapping,
            'total_count': len(mapping)
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"股票映射已保存到缓存文件: {cache_file}")
    except Exception as e:
        print(f"保存缓存文件失败: {e}")

def get_stock_name_mapping() -> Dict[str, str]:
    """获取股票代码到股票名称的映射"""
    cache_file = 'stock_mapping_cache.json'
    
    # 首先尝试从缓存文件加载
    mapping = load_stock_mapping_from_file(cache_file)
    if mapping:
        return mapping
    
    # 缓存无效，尝试从API获取
    try:
        # 尝试从akshare获取实时股票信息
        import akshare as ak
        import time
        print("正在从akshare获取股票名称映射...")
        
        # 获取A股实时行情数据
        df = ak.stock_zh_a_spot_em()
        
        if df is not None and not df.empty:
            # 创建代码到名称的映射
            mapping = {}
            for _, row in df.iterrows():
                code = row['代码']
                name = row['名称']
                
                # 添加不同格式的映射
                if code.startswith('6'):
                    mapping[f'sh.{code}'] = name
                    mapping[f'sh{code}'] = name
                elif code.startswith(('0', '2', '3')):
                    mapping[f'sz.{code}'] = name
                    mapping[f'sz{code}'] = name
                
                # 也添加原始代码映射
                mapping[code] = name
            
            print(f"成功获取 {len(mapping)} 个股票代码映射")
            
            # 保存到缓存文件
            save_stock_mapping_to_file(mapping, cache_file)
            
            return mapping
            
    except ImportError as e:
        raise ImportError(f"akshare库未安装，无法获取股票名称映射: {e}")
    except Exception as e:
        raise RuntimeError(f"从akshare获取股票映射失败: {e}")
    
    # 如果到达这里，说明所有获取映射的方法都失败了
    raise RuntimeError("无法获取股票代码到名称的映射，请检查网络连接或akshare库安装")

def format_stock_config(stocks: List[Tuple[str, float, float]]) -> Dict[str, str]:
    """格式化股票配置
    
    Args:
        stocks: List of (stock_code, predicted_return, confidence)
    
    Returns:
        Dict of {stock_name: stock_code}
    """
    stock_mapping = get_stock_name_mapping()
    config = {}
    
    for stock_code, predicted_return, confidence in stocks:
        # 移除前缀
        clean_code = stock_code.replace('sh.', 'sh').replace('sz.', 'sz')
        stock_name = stock_mapping.get(stock_code, f"股票_{clean_code}")
        config[stock_name] = clean_code
        print(f"添加股票: {stock_name} ({clean_code}) - 预测收益: {predicted_return:.2f}%, 置信度: {confidence:.3f}")
    
    return config

def update_env_file(new_stocks: Dict[str, str], env_path: str = '.env'):
    """更新.env文件中的STOCKS_CONFIG"""
    try:
        # 读取现有的.env文件
        with open(env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 查找STOCKS_CONFIG行
        stocks_config_line_idx = -1
        current_config = {}
        
        for i, line in enumerate(lines):
            if line.strip().startswith('STOCKS_CONFIG='):
                stocks_config_line_idx = i
                # 解析现有配置
                config_str = line.strip().split('=', 1)[1].strip('"\'')
                try:
                    current_config = json.loads(config_str)
                except:
                    current_config = {}
                break
        
        # 合并配置
        merged_config = {**current_config, **new_stocks}
        
        # 更新配置行
        new_config_str = json.dumps(merged_config, ensure_ascii=False, separators=(',', ':'))
        new_line = f'STOCKS_CONFIG="{new_config_str}"\n'
        
        if stocks_config_line_idx >= 0:
            lines[stocks_config_line_idx] = new_line
        else:
            lines.append(new_line)
        
        # 写回文件
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"\n成功更新.env文件，当前STOCKS_CONFIG包含 {len(merged_config)} 只股票:")
        for name, code in merged_config.items():
            print(f"  {name}: {code}")
            
    except Exception as e:
        print(f"更新.env文件失败: {e}")

def main():
    """主函数"""
    # 文件路径
    json_file = 'results/results_20250815/independent_selection_results.json'
    env_file = '.env'
    
    print("=== 处理independent_selection_results.json文件 ===")
    
    # 加载JSON数据
    data = load_json_file(json_file)
    if not data:
        print("无法加载JSON文件")
        return
    
    # 提取高收益股票
    print("\n提取预测收益率 >= 5% 且置信度 >= 0.3 的股票...")
    high_return_stocks = extract_high_return_stocks(data, min_return=5.0, min_confidence=0.3)
    
    if not high_return_stocks:
        print("未找到符合条件的股票")
        return
    
    print(f"\n找到 {len(high_return_stocks)} 只符合条件的股票:")
    for stock_code, predicted_return, confidence in high_return_stocks:
        print(f"  {stock_code}: 预测收益 {predicted_return:.2f}%, 置信度 {confidence:.3f}")
    
    # 格式化股票配置
    print("\n格式化股票配置...")
    new_stocks_config = format_stock_config(high_return_stocks)
    
    # 更新.env文件
    print("\n更新.env文件...")
    update_env_file(new_stocks_config, env_file)
    
    print("\n处理完成！")

if __name__ == '__main__':
    main()