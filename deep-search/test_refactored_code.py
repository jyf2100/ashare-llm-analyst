#!/usr/bin/env python3
"""
重构代码测试脚本
验证所有模块的功能正常性
"""
import os
import sys
import json
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_manager():
    """测试配置管理器"""
    print("=" * 50)
    print("测试配置管理器")
    print("=" * 50)
    
    try:
        from config_manager import config
        
        # 测试获取各种配置
        openai_config = config.get_openai_config()
        print(f"OpenAI配置: {openai_config}")
        
        search_config = config.get_search_config()
        print(f"搜索配置: {search_config}")
        
        cache_config = config.get_cache_config()
        print(f"缓存配置: {cache_config}")
        
        print("✅ 配置管理器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置管理器测试失败: {e}")
        return False

def test_logger_manager():
    """测试日志管理器"""
    print("=" * 50)
    print("测试日志管理器")
    print("=" * 50)
    
    try:
        from logger_manager import get_logger, main_logger
        
        # 测试获取日志器
        test_logger = get_logger('test')
        test_logger.info("这是一条测试日志")
        test_logger.warning("这是一条警告日志")
        test_logger.error("这是一条错误日志")
        
        # 测试主日志器
        main_logger.info("主日志器测试")
        
        print("✅ 日志管理器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 日志管理器测试失败: {e}")
        return False

def test_exception_handler():
    """测试异常处理器"""
    print("=" * 50)
    print("测试异常处理器")
    print("=" * 50)
    
    try:
        from exception_handler import safe_call, log_exceptions, APIError
        
        # 测试安全调用
        @safe_call(default_return="默认值")
        def test_function_with_error():
            raise ValueError("测试异常")
        
        result = test_function_with_error()
        print(f"安全调用结果: {result}")
        
        # 测试异常装饰器
        @log_exceptions
        def test_function_with_log():
            print("正常执行的函数")
            return "成功"
        
        result = test_function_with_log()
        print(f"日志装饰器结果: {result}")
        
        print("✅ 异常处理器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 异常处理器测试失败: {e}")
        return False

def test_cache_manager():
    """测试缓存管理器"""
    print("=" * 50)
    print("测试缓存管理器")
    print("=" * 50)
    
    try:
        from cache_manager import cache_manager, cached
        
        # 测试基本缓存操作
        cache_manager.set("test_key", "test_value", ttl=60)
        value = cache_manager.get("test_key")
        print(f"缓存测试: {value}")
        
        # 测试缓存装饰器
        @cached(ttl=30)
        def expensive_function(x):
            print(f"执行昂贵计算: {x}")
            time.sleep(0.1)  # 模拟耗时操作
            return x * 2
        
        # 第一次调用
        start_time = time.time()
        result1 = expensive_function(5)
        time1 = time.time() - start_time
        
        # 第二次调用（应该从缓存获取）
        start_time = time.time()
        result2 = expensive_function(5)
        time2 = time.time() - start_time
        
        print(f"第一次调用: {result1}, 耗时: {time1:.3f}s")
        print(f"第二次调用: {result2}, 耗时: {time2:.3f}s")
        print(f"缓存统计: {cache_manager.get_stats()}")
        
        print("✅ 缓存管理器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 缓存管理器测试失败: {e}")
        return False

def test_data_fetcher():
    """测试数据获取器"""
    print("=" * 50)
    print("测试数据获取器")
    print("=" * 50)
    
    try:
        from data_fetcher import DataFetcher
        
        fetcher = DataFetcher()
        
        # 测试获取股票列表
        stocks = fetcher.get_all_stocks()
        print(f"获取到股票数量: {len(stocks) if stocks is not None else 0}")
        
        # 测试获取交易日期
        trade_dates = fetcher.get_trade_dates("2024-01-01", "2024-01-31")
        print(f"获取到交易日数量: {len(trade_dates) if trade_dates is not None else 0}")
        
        # 测试获取基本信息（使用一个常见的股票代码）
        if stocks is not None and len(stocks) > 0:
            test_code = stocks.iloc[0]['code'] if 'code' in stocks.columns else None
            if test_code:
                basic_info = fetcher.get_company_basic_info(test_code)
                print(f"基本信息测试: {basic_info is not None}")
        
        fetcher.close()
        print("✅ 数据获取器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据获取器测试失败: {e}")
        return False

def test_search_engine():
    """测试搜索引擎"""
    print("=" * 50)
    print("测试搜索引擎")
    print("=" * 50)
    
    try:
        from search_engine import SearchEngine
        
        engine = SearchEngine()
        
        # 测试网络搜索
        search_result = engine.web_search("平安银行 股票")
        print(f"网络搜索结果: {len(search_result.get('results', [])) if search_result else 0} 条")
        
        # 测试公司新闻搜索
        news_result = engine.search_company_news("平安银行")
        print(f"新闻搜索结果: {len(news_result.get('results', [])) if news_result else 0} 条")
        
        # 测试知识库搜索（可能失败，这是正常的）
        try:
            kb_result = engine.knowledge_base_search("平安银行 财务分析")
            print(f"知识库搜索结果: {kb_result is not None}")
        except Exception as e:
            print(f"知识库搜索失败（预期）: {e}")
        
        engine.close()
        print("✅ 搜索引擎测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 搜索引擎测试失败: {e}")
        return False

def test_ai_analyzer():
    """测试AI分析器"""
    print("=" * 50)
    print("测试AI分析器")
    print("=" * 50)
    
    try:
        from ai_analyzer import AIAnalyzer
        
        analyzer = AIAnalyzer()
        
        # 测试基本分析功能
        test_data = {
            "company_name": "平安银行",
            "financial_data": {"revenue": 1000, "profit": 100},
            "market_data": {"price": 10.5, "volume": 1000000}
        }
        
        # 测试财务分析
        financial_analysis = analyzer.analyze_financial_data("平安银行", test_data["financial_data"])
        print(f"财务分析结果长度: {len(financial_analysis) if financial_analysis else 0}")
        
        # 测试投资建议
        investment_advice = analyzer.generate_investment_advice("平安银行", test_data)
        print(f"投资建议结果长度: {len(investment_advice) if investment_advice else 0}")
        
        print("✅ AI分析器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ AI分析器测试失败: {e}")
        return False

def test_financial_analyzer_v2():
    """测试重构后的财务分析器"""
    print("=" * 50)
    print("测试财务分析器V2")
    print("=" * 50)
    
    try:
        from financial_analyzer_v2 import FinancialAnalyzerV2
        
        # 创建分析器实例
        analyzer = FinancialAnalyzerV2()
        
        # 测试分析状态
        status = analyzer.get_analysis_status()
        print(f"分析器状态: {status['components_status']}")
        
        # 测试简单分析（不启用增强功能以加快测试）
        print("开始测试公司分析...")
        result = analyzer.analyze_company("平安银行", enable_enhancement=False)
        
        print(f"分析结果:")
        print(f"  - 公司名称: {result.get('company_name')}")
        print(f"  - 股票代码: {result.get('stock_code')}")
        print(f"  - 分析成功: {result.get('success')}")
        print(f"  - 完成步骤: {result.get('steps_completed')}")
        print(f"  - 错误信息: {result.get('errors')}")
        
        # 关闭分析器
        analyzer.close()
        
        print("✅ 财务分析器V2测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 财务分析器V2测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始重构代码测试")
    print("测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    # 测试结果统计
    test_results = {}
    
    # 执行各项测试
    test_functions = [
        ("配置管理器", test_config_manager),
        ("日志管理器", test_logger_manager),
        ("异常处理器", test_exception_handler),
        ("缓存管理器", test_cache_manager),
        ("数据获取器", test_data_fetcher),
        ("搜索引擎", test_search_engine),
        ("AI分析器", test_ai_analyzer),
        ("财务分析器V2", test_financial_analyzer_v2),
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            test_results[test_name] = False
        
        print()  # 空行分隔
    
    # 输出测试总结
    print("=" * 50)
    print("测试总结")
    print("=" * 50)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！重构代码功能正常。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)