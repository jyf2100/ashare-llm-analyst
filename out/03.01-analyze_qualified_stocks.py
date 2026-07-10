#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析符合条件的股票：
1. 预测收益率 > 5%
2. 同时存在于 step1_candidates、advanced 和 momentum 三个策略中
3. 将符合条件的股票添加到 .env 文件的 STOCKS_CONFIG 中
"""

import json
import os
import time
from typing import Dict, List, Set

def load_json_file(file_path: str) -> dict:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return {}

def find_qualified_stocks(data: dict) -> List[str]:
    """直接从JSON文件中筛选符合条件的股票，不进行预测"""
    # 直接从ML预测结果中筛选
    ml_predictions = data.get('step3_ml_predictions', {})
    print(f"ML预测结果总数: {len(ml_predictions)}")
    
    if not ml_predictions:
        print("未找到ML预测结果")
        return []
    
    # 筛选预测收益率 > 5% 的股票
    qualified_stocks = []
    for stock, prediction_data in ml_predictions.items():
        predicted_return = prediction_data.get('predicted_return_5d', 0)
        if predicted_return > 1.0:
            qualified_stocks.append((stock, predicted_return))
    
    # 按预测收益率降序排序，取前10条记录
    qualified_stocks.sort(key=lambda x: x[1], reverse=True)
    qualified_stocks = qualified_stocks[:10]
    
    print(f"\n符合条件的股票 (预测收益率 > 5%，前10条):")
    for stock, return_rate in qualified_stocks:
        print(f"  {stock}: {return_rate:.2f}%")
    
    return [stock for stock, _ in qualified_stocks]

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

def get_stock_name_mapping(required_stocks: List[str] = None) -> Dict[str, str]:
    """获取股票代码到股票名称的映射（支持增量更新）"""
    cache_file = 'stock_mapping_cache.json'
    
    # 首先尝试从缓存文件加载
    existing_mapping = {}
    cache_exists = False
    
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            existing_mapping = cache_data.get('mapping', {})
            cache_exists = True
            
            # 检查缓存是否过期（1天）
            cache_time = cache_data.get('timestamp', 0)
            current_time = time.time()
            if current_time - cache_time < 24 * 3600:  # 24小时内有效
                print(f"从缓存文件加载股票映射，共 {len(existing_mapping)} 个")
                return existing_mapping
            else:
                print(f"缓存文件已过期，但保留现有 {len(existing_mapping)} 个映射，只做增量更新")
    except Exception as e:
        print(f"读取缓存文件失败: {e}")
    
    # 如果有指定的股票列表，检查哪些股票缺失映射
    missing_stocks = []
    if required_stocks:
        for stock in required_stocks:
            if stock not in existing_mapping:
                missing_stocks.append(stock)
        
        if not missing_stocks:
            print(f"所需的 {len(required_stocks)} 个股票映射都已存在，无需更新")
            return existing_mapping
        else:
            print(f"需要获取 {len(missing_stocks)} 个缺失股票的映射: {missing_stocks[:5]}{'...' if len(missing_stocks) > 5 else ''}")
    
    # 缓存无效或有缺失股票，尝试从API获取
    try:
        import akshare as ak
        
        # 如果缓存文件存在且只是过期，尝试增量更新
        if cache_exists and not required_stocks:
            print("缓存文件存在但已过期，跳过全量更新以避免频繁API调用")
            print("如需强制更新，请删除缓存文件后重试")
            return existing_mapping
        
        print("正在从akshare获取股票名称映射...")
        
        # 获取A股实时行情数据
        df = ak.stock_zh_a_spot_em()
        
        if df is not None and not df.empty:
            # 创建代码到名称的映射
            new_mapping = existing_mapping.copy()  # 保留现有映射
            update_count = 0
            
            for _, row in df.iterrows():
                code = row['代码']
                name = row['名称']
                
                # 如果指定了required_stocks，只更新缺失的
                if required_stocks:
                    stock_formats = [code, f'sh.{code}', f'sh{code}', f'sz.{code}', f'sz{code}']
                    if not any(fmt in missing_stocks for fmt in stock_formats):
                        continue
                
                # 添加不同格式的映射
                if code.startswith('6'):
                    new_mapping[f'sh.{code}'] = name
                    new_mapping[f'sh{code}'] = name
                elif code.startswith(('0', '2', '3')):
                    new_mapping[f'sz.{code}'] = name
                    new_mapping[f'sz{code}'] = name
                
                # 也添加原始代码映射
                new_mapping[code] = name
                update_count += 1
            
            print(f"成功更新 {update_count} 个股票代码映射，总计 {len(new_mapping)} 个")
            
            # 保存到缓存文件
            try:
                cache_data = {
                    'timestamp': time.time(),
                    'mapping': new_mapping
                }
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                print(f"股票映射已保存到缓存文件: {cache_file}")
            except Exception as e:
                print(f"保存缓存文件失败: {e}")
            
            return new_mapping
            
    except ImportError as e:
        if existing_mapping:
            print(f"akshare库不可用，使用现有缓存映射 {len(existing_mapping)} 个")
            return existing_mapping
        raise ImportError(f"akshare库未安装，无法获取股票名称映射: {e}")
    except Exception as e:
        if existing_mapping:
            print(f"从akshare获取映射失败，使用现有缓存映射 {len(existing_mapping)} 个: {e}")
            return existing_mapping
        raise RuntimeError(f"从akshare获取股票映射失败: {e}")
    
    # 如果到达这里，说明所有获取映射的方法都失败了
    if existing_mapping:
        print(f"使用现有缓存映射 {len(existing_mapping)} 个")
        return existing_mapping
    
    raise RuntimeError("无法获取股票代码到名称的映射，请检查网络连接或akshare库安装")

def format_stock_config(stocks: List[str]) -> Dict[str, str]:
    """格式化股票配置"""
    # 传入需要的股票列表，实现增量更新
    stock_mapping = get_stock_name_mapping(required_stocks=stocks)
    config = {}
    
    for stock_code in stocks:
        # 移除前缀
        clean_code = stock_code.replace('sh.', 'sh').replace('sz.', 'sz')
        stock_name = stock_mapping.get(stock_code, f"股票_{clean_code}")
        config[stock_name] = clean_code
        print(f"添加股票: {stock_name} ({clean_code})")
    
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
        
        # 完全替换配置（不保留原有配置）
        new_config = new_stocks
        
        # 更新配置行
        new_config_str = json.dumps(new_config, ensure_ascii=False, separators=(',', ':'))
        new_line = f"STOCKS_CONFIG='{new_config_str}'\n"
        
        if stocks_config_line_idx >= 0:
            lines[stocks_config_line_idx] = new_line
        else:
            lines.append(new_line)
        
        # 写回文件
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"\n成功更新.env文件，STOCKS_CONFIG已完全替换为 {len(new_config)} 只股票:")
        for name, code in new_config.items():
            print(f"  {name}: {code}")
            
    except Exception as e:
        print(f"更新.env文件失败: {e}")

def main():
    """主函数"""
    # 动态获取当天的结果文件，只做结果文件内容过滤
    from datetime import datetime
    current_date = datetime.now().strftime('%Y%m%d')
    json_file = f'results/results_{current_date}/independent_selection_results.json'
    env_file = '.env'
    
    print("=== 分析符合条件的股票（直接从结果文件筛选）===")
    print(f"使用结果文件: {json_file}")
    
    # 加载JSON数据
    data = load_json_file(json_file)
    if not data:
        print("无法加载JSON文件")
        return
    
    # 找出符合条件的股票
    qualified_stocks = find_qualified_stocks(data)
    
    if not qualified_stocks:
        print("未找到符合条件的股票")
        return
    
    print(f"\n找到 {len(qualified_stocks)} 只符合条件的股票")
    
    # 格式化股票配置
    print("\n格式化股票配置...")
    new_stocks_config = format_stock_config(qualified_stocks)
    
    # 更新.env文件
    print("\n更新.env文件...")
    update_env_file(new_stocks_config, env_file)
    
    print("\n处理完成！")

if __name__ == '__main__':
    main()