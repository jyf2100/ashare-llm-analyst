#!/usr/bin/env python3
"""
MCP聊天功能演示脚本
展示如何使用WeKnora MCP服务进行知识库问答
"""

import asyncio
from mcp_service import MCPServiceClient

async def demo_mcp_chat():
    """演示MCP聊天功能"""
    print("🚀 MCP聊天功能演示")
    print("=" * 50)
    
    # 1. 初始化MCP客户端
    client = MCPServiceClient()
    
    # 2. 使用已初始化的BaoStock知识库
    kb_id = "144e96e2-b7a0-4d1c-bf29-c5a4d02cec10"  # BaoStock-API文档
    print(f"📚 使用知识库: {kb_id}")
    
    # 3. 创建聊天会话
    print("\n🔄 创建聊天会话...")
    session_result = await client.create_session(kb_id=kb_id)
    
    if not session_result.get('success'):
        print(f"❌ 创建会话失败: {session_result.get('error')}")
        return
    
    session_id = session_result.get('session_id')
    print(f"✅ 会话创建成功: {session_id}")
    
    # 4. 演示聊天功能
    print("\n💬 开始聊天演示...")
    
    # 注意：实际的聊天需要使用MCP工具
    print("📝 说明：")
    print("   - 会话已成功创建并准备就绪")
    print("   - 知识库已初始化，包含BaoStock API文档")
    print("   - 实际聊天需要使用MCP工具调用")
    print("   - 可以询问关于BaoStock API的任何问题")
    
    # 模拟聊天测试
    chat_result = await client.chat(session_id=session_id, query="请介绍BaoStock API的主要功能")
    
    if chat_result.get('success'):
        print(f"\n✅ 聊天功能测试成功")
        print(f"📄 响应: {chat_result.get('response')}")
    else:
        print(f"\n❌ 聊天功能测试失败: {chat_result.get('error')}")
    
    # 5. 清理资源
    print("\n🧹 清理资源...")
    try:
        delete_result = await client.delete_session(session_id)
        if delete_result.get('success'):
            print("✅ 会话清理成功")
        else:
            print(f"⚠️ 会话清理失败: {delete_result.get('error')}")
    except Exception as e:
        print(f"⚠️ 清理异常: {e}")
    
    print("\n🎉 演示完成！")
    print("\n📋 使用说明：")
    print("   1. 知识库创建后需要初始化才能使用")
    print("   2. 聊天需要使用已经初始化过的知识库")
    print("   3. 实际聊天交互需要通过MCP工具进行")
    print("   4. 当前演示使用的是BaoStock API文档知识库")

if __name__ == "__main__":
    asyncio.run(demo_mcp_chat())