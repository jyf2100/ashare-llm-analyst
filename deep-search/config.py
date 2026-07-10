"""
配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI配置
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    
    # 代理配置
    HTTP_PROXY = os.getenv('HTTP_PROXY', 'http://172.32.147.190:7890')
    HTTPS_PROXY = os.getenv('HTTPS_PROXY', 'http://172.32.147.190:7890')
    
    # 搜索引擎配置
    SEARCH_ENGINE = os.getenv('SEARCH_ENGINE', 'google')
    SEARCH_API_KEY = os.getenv('SEARCH_API_KEY')
    
    # 数据源配置
    FINANCIAL_DATA_SOURCE = os.getenv('FINANCIAL_DATA_SOURCE', 'eastmoney')
    
    # 请求配置
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # 财报分析配置
    ANALYSIS_YEARS = 5  # 分析过去5年数据
    INDUSTRY_COMPARISON_COUNT = 10  # 行业对比公司数量
    
    @classmethod
    def get_proxies(cls):
        """获取代理配置"""
        if cls.HTTP_PROXY and cls.HTTPS_PROXY:
            return {
                'http': cls.HTTP_PROXY,
                'https': cls.HTTPS_PROXY
            }
        return None