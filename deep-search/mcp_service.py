#!/usr/bin/env python3
"""
MCP服务调用功能模块
集成WeKnora MCP服务，提供知识库管理和智能问答功能
"""
import os
import json
import asyncio
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime

class MCPServiceClient:
    """MCP服务客户端"""
    
    def __init__(self):
        """初始化MCP服务客户端"""
        # WeKnora MCP服务配置
        self.weknora_config = {
            'api_key': os.getenv('WEKNORA_API_KEY', 'sk-D3F7gfPbH3syLw-OP4RhBHYRyyJ2XV7Mo9zYoMYuvyw13mmV'),
            'base_url': os.getenv('WEKNORA_BASE_URL', 'http://172.32.153.184:18000/api/v1'),
            'server_path': '/mnt/disk01/workspaces/worksummary/WeKnora/mcp-server/run_server.py'
        }
        
        # 代理配置
        self.proxies = {
            'http': os.getenv('HTTP_PROXY', 'http://172.32.147.190:7890'),
            'https': os.getenv('HTTPS_PROXY', 'http://172.32.147.190:7890')
        }
        
        # 请求头配置 - 尝试不同的认证方式
        self.headers = {
            'X-API-Key': self.weknora_config["api_key"],  # 尝试X-API-Key
            'Authorization': f'Bearer {self.weknora_config["api_key"]}',  # 保留Bearer
            'Content-Type': 'application/json',
            'User-Agent': 'FinancialAnalyzer/1.0'
        }
        
        # 创建会话
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.proxies.update(self.proxies)
        
        print(f"🔗 初始化MCP服务客户端")
        print(f"📡 WeKnora API: {self.weknora_config['base_url']}")
    
    async def test_connection(self) -> Dict:
        """测试API连接"""
        try:
            # 尝试访问根端点或健康检查端点
            test_urls = [
                f"{self.weknora_config['base_url']}/health",
                f"{self.weknora_config['base_url']}/",
                f"{self.weknora_config['base_url']}/models"  # 尝试模型端点
            ]
            
            for url in test_urls:
                try:
                    print(f"🔍 测试连接: {url}")
                    response = self.session.get(url, timeout=10)
                    print(f"📊 响应状态: {response.status_code}")
                    print(f"📄 响应头: {dict(response.headers)}")
                    
                    if response.status_code in [200, 401, 403]:  # 401/403说明服务在运行，只是认证问题
                        return {
                            'success': True,
                            'status_code': response.status_code,
                            'url': url,
                            'response': response.text[:500]
                        }
                except Exception as e:
                    print(f"⚠️ 连接测试失败 {url}: {str(e)}")
                    continue
            
            return {'success': False, 'error': '所有连接测试都失败'}
            
        except Exception as e:
            return {'success': False, 'error': f'连接测试异常: {str(e)}'}
    
    async def create_tenant(self, name: str, description: str, business: str) -> Dict:
        """创建租户"""
        try:
            url = f"{self.weknora_config['base_url']}/tenants"
            data = {
                'name': name,
                'description': description,
                'business': business,
                'retriever_engines': {
                    'engines': [
                        {
                            'retriever_engine_type': 'hybrid',
                            'retriever_type': 'default'
                        }
                    ]
                }
            }
            
            response = self.session.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 租户创建成功: {name}")
                return {
                    'success': True,
                    'tenant_id': result.get('id'),
                    'data': result
                }
            else:
                error_msg = f"租户创建失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"租户创建异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    async def list_tenants(self) -> Dict:
        """获取租户列表"""
        try:
            url = f"{self.weknora_config['base_url']}/tenants"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 获取租户列表成功，共 {len(result)} 个租户")
                return {
                    'success': True,
                    'tenants': result
                }
            else:
                error_msg = f"获取租户列表失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"获取租户列表异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    async def create_knowledge_base(self, name: str, description: str, 
                                  embedding_model_id: Optional[str] = None,
                                  summary_model_id: Optional[str] = None) -> Dict:
        """创建知识库"""
        try:
            url = f"{self.weknora_config['base_url']}/knowledge-bases"
            data = {
                'name': name,
                'description': description
            }
            
            if embedding_model_id:
                data['embedding_model_id'] = embedding_model_id
            if summary_model_id:
                data['summary_model_id'] = summary_model_id
            
            print(f"🔍 调试 - 会话创建发送数据: {data}")
            print(f"🔍 调试 - 会话创建请求URL: {url}")
            response = self.session.post(url, json=data, timeout=30)
            
            if response.status_code in [200, 201]:  # 支持200和201状态码
                result = response.json()
                kb_data = result.get('data', result)  # 处理不同的响应格式
                print(f"✅ 知识库创建成功: {name}")
                return {
                    'success': True,
                    'kb_id': kb_data.get('id'),
                    'data': kb_data
                }
            else:
                error_msg = f"知识库创建失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"知识库创建异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    async def list_knowledge_bases(self) -> Dict:
        """获取知识库列表"""
        try:
            url = f"{self.weknora_config['base_url']}/knowledge-bases"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 获取知识库列表成功，共 {len(result)} 个知识库")
                return {
                    'success': True,
                    'knowledge_bases': result
                }
            else:
                error_msg = f"获取知识库列表失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"获取知识库列表异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    async def create_knowledge_from_url(self, kb_id: str, url: str, 
                                      enable_multimodel: bool = True) -> Dict:
        """从URL创建知识"""
        try:
            api_url = f"{self.weknora_config['base_url']}/knowledge-bases/{kb_id}/knowledge/url"
            data = {
                'url': url,
                'enable_multimodel': enable_multimodel
            }
            
            response = self.session.post(api_url, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 从URL创建知识成功: {url}")
                return {
                    'success': True,
                    'knowledge_id': result.get('id'),
                    'data': result
                }
            else:
                error_msg = f"从URL创建知识失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"从URL创建知识异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    async def hybrid_search(self, kb_id: str, query: str, 
                          match_count: int = 5,
                          vector_threshold: float = 0.5,
                          keyword_threshold: float = 0.3) -> Dict:
        """混合搜索"""
        try:
            url = f"{self.weknora_config['base_url']}/knowledge-bases/{kb_id}/search/hybrid"
            data = {
                'query': query,
                'match_count': match_count,
                'vector_threshold': vector_threshold,
                'keyword_threshold': keyword_threshold
            }
            
            response = self.session.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 混合搜索成功，找到 {len(result.get('results', []))} 个结果")
                return {
                    'success': True,
                    'results': result.get('results', []),
                    'data': result
                }
            else:
                error_msg = f"混合搜索失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"混合搜索异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    async def create_chat_session(self, kb_id: str, 
                                summary_model_id: Optional[str] = None,
                                enable_rewrite: bool = True,
                                max_rounds: int = 5,
                                fallback_response: str = "抱歉，我无法回答这个问题。") -> Dict:
        """创建聊天会话"""
        try:
            url = f"{self.weknora_config['base_url']}/sessions"
            data = {
                'knowledge_base_id': kb_id,  # 尝试使用snake_case字段名
                'enable_rewrite': enable_rewrite,
                'max_rounds': max_rounds,
                'fallback_response': fallback_response
            }
            
            if summary_model_id:
                data['summary_model_id'] = summary_model_id
            
            print(f"🔍 调试 - 会话创建发送数据: {data}")
            print(f"🔍 调试 - 会话创建请求URL: {url}")
            response = self.session.post(url, json=data, timeout=30)
            
            if response.status_code in [200, 201]:  # 支持200和201状态码
                result = response.json()
                session_data = result.get('data', result)  # 处理不同的响应格式
                print(f"✅ 聊天会话创建成功")
                return {
                    'success': True,
                    'session_id': session_data.get('id'),
                    'data': session_data
                }
            else:
                error_msg = f"聊天会话创建失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"聊天会话创建异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    async def chat(self, session_id: str, query: str) -> Dict:
        """发送聊天消息"""
        try:
            # 使用正确的聊天API端点
            url = f"{self.weknora_config['base_url']}/knowledge-chat/{session_id}"
            print(f"🔍 调试 - 聊天请求URL: {url}")
            
            data = {
                "query": query
            }
            print(f"🔍 调试 - 聊天请求数据: {data}")
            
            response = self.session.post(url, json=data, timeout=60, stream=True)
            print(f"🔍 调试 - 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                print(f"✅ 聊天成功，处理流式响应...")
                
                # 处理流式响应
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data:'):
                            try:
                                import json
                                data_str = line_str[5:].strip()  # 移除 'data:' 前缀
                                if data_str:
                                    data_obj = json.loads(data_str)
                                    content = data_obj.get('content', '')
                                    if content:
                                        full_response += content
                                    
                                    # 如果是最后一条消息，停止处理
                                    if data_obj.get('done', False):
                                        break
                            except json.JSONDecodeError:
                                continue
                
                print(f"🔍 调试 - 完整响应: {full_response}")
                
                return {
                    'success': True,
                    'response': full_response.strip()
                }
            else:
                error_msg = f"聊天失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
            
        except Exception as e:
            error_msg = f"聊天异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    async def enhance_financial_analysis(self, company_name: str, 
                                       financial_data: Dict,
                                       analysis_report: Dict) -> Dict:
        """使用MCP服务增强财务分析"""
        try:
            print(f"🔍 使用MCP服务增强 {company_name} 的财务分析")
            
            # 1. 创建或获取知识库
            kb_result = await self.create_knowledge_base(
                name=f"财务分析_{company_name}_{datetime.now().strftime('%Y%m%d')}",
                description=f"{company_name}财务分析知识库"
            )
            
            if not kb_result['success']:
                return {'success': False, 'error': '知识库创建失败'}
            
            kb_id = kb_result['kb_id']
            
            # 2. 创建聊天会话
            session_result = await self.create_chat_session(kb_id)
            if not session_result['success']:
                return {'success': False, 'error': '聊天会话创建失败'}
            
            session_id = session_result['session_id']
            
            # 3. 构建增强分析查询
            query = self._build_enhancement_query(company_name, financial_data, analysis_report)
            
            # 4. 获取MCP增强分析
            chat_result = await self.chat(session_id, query)
            if not chat_result['success']:
                return {'success': False, 'error': '增强分析失败'}
            
            # 5. 解析和整合结果
            enhanced_analysis = self._parse_enhancement_result(chat_result['response'])
            
            return {
                'success': True,
                'enhanced_analysis': enhanced_analysis,
                'kb_id': kb_id,
                'session_id': session_id,
                'original_response': chat_result['response']
            }
            
        except Exception as e:
            error_msg = f"MCP增强分析异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def _build_enhancement_query(self, company_name: str, 
                               financial_data: Dict, 
                               analysis_report: Dict) -> str:
        """构建增强分析查询"""
        query = f"""
请基于以下信息，对{company_name}进行深度财务分析增强：

## 公司基本信息
公司名称: {company_name}

## 财务数据
{json.dumps(financial_data, ensure_ascii=False, indent=2)}

## 初步分析报告
{json.dumps(analysis_report, ensure_ascii=False, indent=2)}

## 请提供以下增强分析：
1. 深度财务指标解读
2. 行业对比和竞争优势分析
3. 风险因素识别和评估
4. 投资建议和目标价位
5. 未来发展趋势预测

请提供专业、客观、详细的分析意见。
"""
        return query
    
    def _parse_enhancement_result(self, response: str) -> Dict:
        """解析增强分析结果"""
        try:
            # 尝试解析结构化响应
            if response.strip().startswith('{'):
                return json.loads(response)
            
            # 如果是文本响应，进行简单结构化
            return {
                'enhanced_analysis': response,
                'analysis_type': 'text',
                'timestamp': datetime.now().isoformat(),
                'source': 'weknora_mcp'
            }
            
        except Exception as e:
            print(f"⚠️ 解析增强分析结果失败: {e}")
            return {
                'enhanced_analysis': response,
                'analysis_type': 'raw_text',
                'timestamp': datetime.now().isoformat(),
                'source': 'weknora_mcp',
                'parse_error': str(e)
            }
    
    async def delete_session(self, session_id: str) -> Dict:
        """删除聊天会话"""
        try:
            url = f"{self.weknora_config['base_url']}/sessions/{session_id}"
            response = self.session.delete(url, timeout=30)
            
            if response.status_code in [200, 204]:
                print(f"✅ 删除会话成功: {session_id}")
                return {'success': True}
            else:
                error_msg = f"删除会话失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"删除会话异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    async def delete_knowledge_base(self, kb_id: str) -> Dict:
        """删除知识库"""
        try:
            url = f"{self.weknora_config['base_url']}/knowledge-bases/{kb_id}"
            response = self.session.delete(url, timeout=30)
            
            if response.status_code in [200, 204]:
                print(f"✅ 删除知识库成功: {kb_id}")
                return {'success': True}
            else:
                error_msg = f"删除知识库失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"删除知识库异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}

    # 别名方法，保持向后兼容
    async def create_session(self, kb_id: str, **kwargs) -> Dict:
        """create_chat_session的别名方法"""
        return await self.create_chat_session(kb_id, **kwargs)

# 使用示例
async def test_mcp_service():
    """测试MCP服务功能"""
    client = MCPServiceClient()
    
    # 测试租户列表
    tenants = await client.list_tenants()
    print(f"租户列表: {tenants}")
    
    # 测试知识库列表
    kbs = await client.list_knowledge_bases()
    print(f"知识库列表: {kbs}")

if __name__ == "__main__":
    asyncio.run(test_mcp_service())