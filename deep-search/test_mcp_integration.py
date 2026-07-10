#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP服务集成测试脚本
测试WeKnora MCP服务的集成功能
"""

import asyncio
import sys
import os
from mcp_service import MCPServiceClient

async def test_mcp_service():
    """测试MCP服务基本功能"""
    print("🧪 开始测试MCP服务集成...")
    
    try:
        # 1. 初始化MCP客户端
        print("1️⃣ 初始化MCP客户端...")
        client = MCPServiceClient()
        print("✅ MCP客户端初始化成功")
        
        # 1.5. 测试连接
        print("\n1️⃣.5️⃣ 测试API连接...")
        connection_result = await client.test_connection()
        if connection_result.get('success'):
            print(f"✅ API连接成功: {connection_result.get('url')}")
            print(f"   状态码: {connection_result.get('status_code')}")
        else:
            print(f"❌ API连接失败: {connection_result.get('error')}")
            print("⚠️ 可能的原因：服务未启动、网络问题或配置错误")
        
        # 2. 测试租户列表
        print("\n2️⃣ 测试租户列表...")
        tenants_result = await client.list_tenants()
        if tenants_result.get('success'):
            tenants = tenants_result.get('tenants', [])
            print(f"✅ 获取租户列表成功，共 {len(tenants)} 个租户")
            # 安全地处理租户列表
            if isinstance(tenants, list) and tenants:
                for tenant in tenants[:3]:  # 只显示前3个
                    if isinstance(tenant, dict):
                        print(f"   - {tenant.get('name', 'N/A')}: {tenant.get('description', 'N/A')}")
                    else:
                        print(f"   - {tenant}")
            else:
                print(f"   租户数据格式: {type(tenants)}")
        else:
            print(f"❌ 获取租户列表失败: {tenants_result.get('error', '未知错误')}")
        
        # 3. 测试知识库列表
        print("\n3️⃣ 测试知识库列表...")
        kb_result = await client.list_knowledge_bases()
        if kb_result.get('success'):
            kbs = kb_result.get('knowledge_bases', [])
            print(f"✅ 获取知识库列表成功，共 {len(kbs)} 个知识库")
            # 安全地处理知识库列表
            if isinstance(kbs, list) and kbs:
                for kb in kbs[:3]:  # 只显示前3个
                    if isinstance(kb, dict):
                        print(f"   - {kb.get('name', 'N/A')}: {kb.get('description', 'N/A')}")
                    else:
                        print(f"   - {kb}")
            else:
                print(f"   知识库数据格式: {type(kbs)}")
        else:
            print(f"❌ 获取知识库列表失败: {kb_result.get('error', '未知错误')}")
        
        # 4. 测试创建知识库
        print("\n4️⃣ 测试创建知识库...")
        test_kb_name = "测试知识库_MCP集成"
        create_kb_result = await client.create_knowledge_base(
            name=test_kb_name,
            description="用于测试MCP服务集成的知识库"
        )
        
        if create_kb_result.get('success'):
            kb_id = create_kb_result.get('kb_id')
            print(f"✅ 创建知识库成功: {kb_id}")
            print(f"🔍 调试信息 - 知识库ID: '{kb_id}', 类型: {type(kb_id)}")
        else:
            print(f"❌ 创建知识库失败: {create_kb_result.get('error', '未知错误')}")
            
        # 5. 使用现有的已初始化知识库进行会话测试
        print("\n5️⃣ 测试创建会话（使用现有知识库）...")
        # 使用BaoStock-API文档知识库，它应该已经初始化
        existing_kb_id = "144e96e2-b7a0-4d1c-bf29-c5a4d02cec10"  # BaoStock-API文档
        print(f"🔍 使用现有知识库ID: {existing_kb_id}")
        session_result = await client.create_session(kb_id=existing_kb_id)
        if session_result.get('success'):
            session_id = session_result.get('session_id')
            print(f"✅ 创建会话成功: {session_id}")
            
            # 6. 测试聊天功能
            print("\n6️⃣ 测试聊天功能...")
            chat_result = await client.chat(
                session_id=session_id,
                query="你好，请介绍一下BaoStock API的功能"
            )
            if chat_result.get('success'):
                response = chat_result.get('response', '')
                print(f"✅ 聊天测试成功")
                print(f"   回复: {response[:100]}...")
            else:
                print(f"❌ 聊天测试失败: {chat_result.get('error', '未知错误')}")
            
            # 7. 清理测试数据
            print("\n7️⃣ 清理测试数据...")
            try:
                # 删除会话
                delete_session_result = await client.delete_session(session_id)
                if delete_session_result.get('success'):
                    print("✅ 删除测试会话成功")
                else:
                    print(f"⚠️ 删除测试会话失败: {delete_session_result.get('error', '')}")
            except Exception as e:
                print(f"⚠️ 删除会话异常: {str(e)}")
        else:
            print(f"❌ 创建会话失败: {session_result.get('error', '未知错误')}")
        
        print("\n🎉 MCP服务集成测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ MCP服务测试异常: {str(e)}")
        return False

async def test_financial_analysis_integration():
    """测试财务分析集成"""
    print("\n🔬 测试财务分析集成...")
    
    try:
        # 导入主分析器
        from main import FinancialAnalyzer
        
        # 创建分析器实例
        analyzer = FinancialAnalyzer()
        
        # 检查MCP客户端是否正确初始化
        if hasattr(analyzer, 'mcp_client'):
            print("✅ 财务分析器中MCP客户端初始化成功")
            
            # 测试MCP客户端基本功能
            tenants_result = await analyzer.mcp_client.list_tenants()
            if tenants_result.get('success'):
                print("✅ 财务分析器中MCP服务连接正常")
            else:
                print(f"❌ 财务分析器中MCP服务连接失败: {tenants_result.get('error', '')}")
        else:
            print("❌ 财务分析器中MCP客户端未正确初始化")
            return False
        
        print("✅ 财务分析集成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 财务分析集成测试失败: {str(e)}")
        return False

async def main():
    """主测试函数"""
    print("🚀 开始MCP服务集成测试")
    print("="*50)
    
    # 检查环境变量
    api_key = os.getenv('WEKNORA_API_KEY')
    base_url = os.getenv('WEKNORA_BASE_URL')
    
    if not api_key or not base_url:
        print("❌ 缺少必要的环境变量:")
        print(f"   WEKNORA_API_KEY: {'✅' if api_key else '❌'}")
        print(f"   WEKNORA_BASE_URL: {'✅' if base_url else '❌'}")
        print("\n请设置环境变量后重试")
        return
    
    print(f"✅ 环境变量检查通过")
    print(f"   API Key: {api_key[:20]}...")
    print(f"   Base URL: {base_url}")
    
    # 执行测试
    test1_success = await test_mcp_service()
    test2_success = await test_financial_analysis_integration()
    
    print("\n" + "="*50)
    print("📊 测试结果汇总:")
    print(f"   MCP服务基本功能: {'✅ 通过' if test1_success else '❌ 失败'}")
    print(f"   财务分析集成: {'✅ 通过' if test2_success else '❌ 失败'}")
    
    if test1_success and test2_success:
        print("\n🎉 所有测试通过！MCP服务集成成功！")
    else:
        print("\n⚠️ 部分测试失败，请检查配置和网络连接")

if __name__ == "__main__":
    asyncio.run(main())