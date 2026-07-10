#!/usr/bin/env python3
"""
搜索引擎模块
负责网络搜索和知识库查询
"""
import requests
import json
from typing import Dict, Any, List, Optional
from urllib.parse import quote

from config_manager import config
from logger_manager import get_logger
from exception_handler import api_retry, safe_data_processing, APIError
from cache_manager import cache_manager

logger = get_logger('search')

class SearchEngine:
    """搜索引擎"""
    
    def __init__(self) -> None:
        self.session: requests.Session = requests.Session()
        self.session.headers.update(config.get_headers())
        
        # 设置代理
        proxy_config = config.get_proxy_config()
        if proxy_config:
            self.session.proxies.update(proxy_config)
        
        self.search_config: Dict[str, Any] = config.get_search_config()
        self.kb_config: Dict[str, Any] = config.get_knowledge_base_config()
    
    @api_retry(max_retries=3, delay=1.0)
    @safe_data_processing(default_value={})
    def web_search(self, query: str, max_results: int = None) -> Dict[str, Any]:
        """网络搜索"""
        if not max_results:
            max_results = self.search_config['max_results']
        
        cache_key = f"web_search_{query}_{max_results}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 构建搜索参数
            search_params = {
                'query': query,
                'max_results': max_results,
                'include_domains': self.search_config['trusted_domains']
            }
            
            # 使用Tavily API进行搜索
            api_key = self.search_config['api_key']
            url = "https://api.tavily.com/search"
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            payload = {
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": max_results,
                "include_domains": self.search_config['trusted_domains']
            }
            
            response = self.session.post(
                url, 
                json=payload, 
                headers=headers,
                timeout=self.search_config['timeout']
            )
            
            if response.status_code != 200:
                raise APIError(f"搜索API调用失败: {response.status_code}", "tavily", response.status_code)
            
            result = response.json()
            
            # 处理搜索结果
            processed_result = {
                'query': query,
                'answer': result.get('answer', ''),
                'results': [],
                'total_results': len(result.get('results', []))
            }
            
            for item in result.get('results', []):
                processed_result['results'].append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'content': item.get('content', ''),
                    'score': item.get('score', 0.0)
                })
            
            cache_manager.set(cache_key, processed_result)
            logger.info(f"网络搜索成功: {query}, 找到{len(processed_result['results'])}个结果")
            return processed_result
            
        except Exception as e:
            logger.error(f"网络搜索失败: {query}, 错误: {e}")
            raise
    
    @api_retry(max_retries=2, delay=0.5)
    @safe_data_processing(default_value={})
    def knowledge_base_search(self, query: str, kb_id: str = None) -> Dict[str, Any]:
        """知识库搜索"""
        if not kb_id:
            kb_id = self.kb_config['id']
        
        cache_key = f"kb_search_{kb_id}_{query}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 这里应该调用实际的知识库API
            # 由于没有具体的知识库API实现，这里返回模拟结果
            result = {
                'query': query,
                'kb_id': kb_id,
                'results': [],
                'total_results': 0,
                'status': 'no_implementation'
            }
            
            logger.warning(f"知识库搜索功能未实现: {query}")
            return result
            
        except Exception as e:
            logger.error(f"知识库搜索失败: {query}, 错误: {e}")
            raise
    
    def search_trusted_sources(self, query: str, company_name: str = "") -> Dict[str, Any]:
        """搜索可信来源"""
        try:
            # 构建针对性查询
            if company_name:
                enhanced_query = f"{company_name} {query} 财报 年报 季报"
            else:
                enhanced_query = query
            
            # 限制搜索域名为可信财经网站
            trusted_query = f"{enhanced_query} site:({' OR site:'.join(self.search_config['trusted_domains'])})"
            
            result = self.web_search(trusted_query)
            
            # 过滤和排序结果
            filtered_results = []
            for item in result.get('results', []):
                url = item.get('url', '')
                if any(domain in url for domain in self.search_config['trusted_domains']):
                    filtered_results.append(item)
            
            result['results'] = filtered_results[:self.search_config['max_results']]
            result['total_results'] = len(filtered_results)
            
            logger.info(f"可信来源搜索完成: {query}, 找到{len(filtered_results)}个可信结果")
            return result
            
        except Exception as e:
            logger.error(f"可信来源搜索失败: {query}, 错误: {e}")
            return {}
    
    def multi_source_search(self, query: str, sources: List[str] = None) -> Dict[str, Any]:
        """多源搜索"""
        if not sources:
            sources = ['web', 'knowledge_base']
        
        results = {
            'query': query,
            'sources': {},
            'combined_results': []
        }
        
        # Web搜索
        if 'web' in sources:
            try:
                web_result = self.web_search(query)
                results['sources']['web'] = web_result
                
                # 添加到合并结果
                for item in web_result.get('results', []):
                    item['source'] = 'web'
                    results['combined_results'].append(item)
                    
            except Exception as e:
                logger.error(f"Web搜索失败: {e}")
                results['sources']['web'] = {'error': str(e)}
        
        # 知识库搜索
        if 'knowledge_base' in sources:
            try:
                kb_result = self.knowledge_base_search(query)
                results['sources']['knowledge_base'] = kb_result
                
                # 添加到合并结果
                for item in kb_result.get('results', []):
                    item['source'] = 'knowledge_base'
                    results['combined_results'].append(item)
                    
            except Exception as e:
                logger.error(f"知识库搜索失败: {e}")
                results['sources']['knowledge_base'] = {'error': str(e)}
        
        # 按相关性排序
        results['combined_results'].sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"多源搜索完成: {query}, 共{len(results['combined_results'])}个结果")
        return results
    
    def search_company_news(self, company_name: str, days: int = 30) -> Dict[str, Any]:
        """搜索公司新闻"""
        query = f"{company_name} 最新消息 新闻 公告"
        
        # 添加时间限制
        if days <= 7:
            time_filter = "最近一周"
        elif days <= 30:
            time_filter = "最近一个月"
        else:
            time_filter = "最近三个月"
        
        enhanced_query = f"{query} {time_filter}"
        
        return self.search_trusted_sources(enhanced_query, company_name)
    
    def search_industry_analysis(self, industry: str) -> Dict[str, Any]:
        """搜索行业分析"""
        query = f"{industry} 行业分析 发展趋势 市场前景 竞争格局"
        return self.search_trusted_sources(query)
    
    def search_financial_indicators(self, company_name: str, indicator: str) -> Dict[str, Any]:
        """搜索财务指标分析"""
        query = f"{company_name} {indicator} 财务分析 业绩表现"
        return self.search_trusted_sources(query, company_name)
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """获取搜索建议"""
        suggestions = [
            f"{query} 财务分析",
            f"{query} 投资价值",
            f"{query} 风险评估",
            f"{query} 行业地位",
            f"{query} 发展前景"
        ]
        
        return suggestions
    
    def close(self) -> None:
        """关闭搜索引擎"""
        self.session.close()
        logger.info("搜索引擎已关闭")