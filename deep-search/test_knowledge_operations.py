#!/usr/bin/env python3
"""
知识库操作功能测试脚本
测试 main.py 中新增的知识库操作功能
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
load_dotenv()

from main import FinancialAnalyzer

async def test_knowledge_operations():
    """测试知识库操作功能"""
    print("🚀 开始测试知识库操作功能")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = FinancialAnalyzer()
    
    # 测试1: 获取知识库信息
    print("\n📊 测试1: 获取知识库信息")
    print("-" * 40)
    kb_info = await analyzer.list_knowledge_base_info()
    if kb_info.get('success'):
        kb = kb_info.get('knowledge_base', {})
        print(f"✅ 知识库名称: {kb.get('name', '未知')}")
        print(f"📝 知识库描述: {kb.get('description', '无描述')}")
        print(f"🆔 知识库ID: {kb.get('id', '未知')}")
    else:
        print(f"❌ 获取知识库信息失败: {kb_info.get('error')}")
    
    # 测试2: 查询知识库
    print("\n🔍 测试2: 查询知识库")
    print("-" * 40)
    query_result = await analyzer.query_knowledge("大金重工盈利分析")
    if query_result.get('success'):
        results = query_result.get('results', [])
        print(f"✅ 查询成功，找到 {len(results)} 条结果")
        for i, result in enumerate(results[:2], 1):  # 只显示前2条
            print(f"  {i}. {result.get('content', '')[:100]}...")
    else:
        print(f"❌ 查询失败: {query_result.get('error')}")
    
    # 测试3: 与知识库对话
    print("\n💬 测试3: 与知识库对话")
    print("-" * 40)
    chat_result = await analyzer.chat_with_knowledge("请简单介绍一下大金重工")
    if chat_result.get('success'):
        answer = chat_result.get('answer', '')
        print(f"✅ 对话成功")
        print(f"📝 回答: {answer[:200]}...")
    else:
        print(f"❌ 对话失败: {chat_result.get('error')}")
    
    # 测试4: 添加知识（可选，需要有效URL）
    print("\n📚 测试4: 添加知识（跳过，避免添加测试数据）")
    print("-" * 40)
    print("⏭️ 跳过添加知识测试，避免向知识库添加测试数据")
    
    print("\n" + "=" * 60)
    print("✅ 知识库操作功能测试完成")

async def test_enhanced_analysis():
    """测试增强分析功能"""
    print("\n🚀 开始测试增强分析功能")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = FinancialAnalyzer()
    
    # 测试增强分析（使用简单的股票代码）
    print("\n📈 测试增强分析: 大金重工")
    print("-" * 40)
    
    try:
        # 这里只测试知识库查询部分，不执行完整分析
        test_queries = [
            "大金重工的最新财务状况如何？",
            "风电业的发展趋势是什么？"
        ]
        
        for query in test_queries:
            print(f"🔍 测试查询: {query}")
            result = await analyzer.chat_with_knowledge(query)
            if result.get('success'):
                answer = result.get('answer', '')
                print(f"✅ 查询成功: {answer[:100]}...")
            else:
                print(f"❌ 查询失败: {result.get('error')}")
            
            # 避免频繁请求
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ 增强分析功能测试完成")

async def main():
    """主测试函数"""
    print("🧪 知识库操作功能测试")
    print("=" * 60)
    
    try:
        # 测试基础知识库操作
        await test_knowledge_operations()
        
        # 测试增强分析功能
        await test_enhanced_analysis()
        
    except Exception as e:
        print(f"💥 测试过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 所有测试完成")

if __name__ == "__main__":
    asyncio.run(main())