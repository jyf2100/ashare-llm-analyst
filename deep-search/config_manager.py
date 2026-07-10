#!/usr/bin/env python3
"""
配置管理模块
统一管理财报分析系统的所有配置项
"""
import os
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ConfigManager:
    """配置管理器"""
    
    def __init__(self) -> None:
        """初始化配置管理器"""
        # 加载环境变量
        load_dotenv()
        
        # 初始化配置
        self._config = self._load_default_config()
        
        # 从环境变量更新配置
        self._update_from_env()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            # OpenAI API配置
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY', 'sk-eD9oqSXvWMfQg5hUv487qw'),
                'base_url': os.getenv('OPENAI_BASE_URL', 'http://172.32.153.184:14000/v1'),
                'model': os.getenv('OPENAI_MODEL', 'Qwen3-Next-80B-A3B'),
                'timeout': int(os.getenv('OPENAI_TIMEOUT', '30')),
                'max_retries': int(os.getenv('OPENAI_MAX_RETRIES', '3'))
            },
            
            # 搜索API配置
            'search': {
                'api_key': os.getenv('SEARCH_API_KEY', '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'),
                'timeout': int(os.getenv('SEARCH_TIMEOUT', '20')),
                'max_results': int(os.getenv('SEARCH_MAX_RESULTS', '5')),
                'trusted_domains': [
                    'eastmoney.com', 'cninfo.com.cn', 'sse.com.cn', 'szse.cn',
                    'csrc.gov.cn', 'xinhuanet.com', 'people.com.cn', 'caixin.com',
                    'ftchinese.com', 'wallstreetcn.com', 'cls.cn', 'gelonghui.com'
                ]
            },
            
            # 代理配置
            'proxy': {
                'http': os.getenv('HTTP_PROXY', 'http://172.32.147.190:7890'),
                'https': os.getenv('HTTPS_PROXY', 'http://172.32.147.190:7890'),
                'enabled': os.getenv('PROXY_ENABLED', 'true').lower() == 'true'
            },
            
            # 请求头配置
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
            },
            
            # 知识库配置
            'knowledge_base': {
                'id': os.getenv('WEKNORA_KB_ID', '77c5a766-bbb7-47c4-8027-da3cec4d2a2b'),
                'timeout': int(os.getenv('KB_TIMEOUT', '15')),
                'max_retries': int(os.getenv('KB_MAX_RETRIES', '2'))
            },
            
            # 缓存配置
            'cache': {
                'enabled': os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
                'ttl': int(os.getenv('CACHE_TTL', '3600')),  # 1小时
                'max_size': int(os.getenv('CACHE_MAX_SIZE', '100'))
            },
            
            # 日志配置
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'format': os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                'file': os.getenv('LOG_FILE', 'financial_analyzer.log'),
                'max_size': int(os.getenv('LOG_MAX_SIZE', '10485760')),  # 10MB
                'backup_count': int(os.getenv('LOG_BACKUP_COUNT', '5'))
            },
            
            # 下载配置
            'download': {
                'base_dir': os.getenv('DOWNLOAD_DIR', 'downloads'),
                'max_file_size': int(os.getenv('MAX_FILE_SIZE', '52428800')),  # 50MB
                'allowed_extensions': ['.pdf', '.doc', '.docx', '.xls', '.xlsx']
            },
            
            # 分析配置
            'analysis': {
                'step_delay': float(os.getenv('STEP_DELAY', '0.5')),
                'validation_enabled': os.getenv('VALIDATION_ENABLED', 'true').lower() == 'true',
                'mcp_enhancement_enabled': os.getenv('MCP_ENHANCEMENT_ENABLED', 'true').lower() == 'true'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_openai_config(self) -> Dict[str, Any]:
        """获取OpenAI配置"""
        return self._config['openai']
    
    def get_search_config(self) -> Dict[str, Any]:
        """获取搜索配置"""
        return self._config['search']
    
    def get_proxy_config(self) -> Dict[str, str]:
        """获取代理配置"""
        proxy_config = self._config['proxy']
        if proxy_config['enabled']:
            return {
                'http': proxy_config['http'],
                'https': proxy_config['https']
            }
        return {}
    
    def get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return self._config['headers']
    
    def get_knowledge_base_config(self) -> Dict[str, Any]:
        """获取知识库配置"""
        return self._config['knowledge_base']
    
    def get_cache_config(self) -> Dict[str, Any]:
        """获取缓存配置"""
        return self._config['cache']
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self._config['logging']
    
    def get_download_config(self) -> Dict[str, Any]:
        """获取下载配置"""
        return self._config['download']
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """获取分析配置"""
        return self._config['analysis']
    
    def update_config(self, key: str, value: Any) -> None:
        """更新配置项"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def reload(self) -> None:
        """重新加载配置"""
        load_dotenv(override=True)
        self._config = self._load_default_config()
        self._update_from_env()
    
    def _update_from_env(self) -> None:
        """从环境变量更新配置"""
        # 这里可以添加从环境变量更新配置的逻辑
        pass

# 全局配置实例
config = ConfigManager()