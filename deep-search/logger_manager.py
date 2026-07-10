#!/usr/bin/env python3
"""
日志管理模块
统一管理财报分析系统的日志记录
"""
import logging
import logging.handlers
import os
from typing import Dict, Optional, List, Union
from config_manager import config

class LoggerManager:
    """日志管理器"""
    
    def __init__(self) -> None:
        self._loggers: Dict[str, logging.Logger] = {}
        self._setup_root_logger()
    
    def _setup_root_logger(self) -> None:
        """设置根日志记录器"""
        log_config = config.get_logging_config()
        
        # 创建根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_config['level']))
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(log_config['format'])
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 文件处理器（轮转）
        if log_config['file']:
            file_handler = logging.handlers.RotatingFileHandler(
                log_config['file'],
                maxBytes=log_config['max_size'],
                backupCount=log_config['backup_count'],
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, log_config['level']))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志记录器"""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def set_level(self, level: str) -> None:
        """设置日志级别"""
        log_level = getattr(logging, level.upper())
        logging.getLogger().setLevel(log_level)
        
        # 更新所有处理器的级别
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.setLevel(log_level)
    
    def add_file_handler(self, filename: str, level: str = 'DEBUG') -> None:
        """添加文件处理器"""
        log_config = config.get_logging_config()
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename,
            maxBytes=log_config['max_size'],
            backupCount=log_config['backup_count'],
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        
        formatter = logging.Formatter(log_config['format'])
        file_handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(file_handler)
    
    def remove_console_output(self) -> None:
        """移除控制台输出"""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.handlers.RotatingFileHandler):
                root_logger.removeHandler(handler)

# 全局日志管理器实例
logger_manager = LoggerManager()

# 便捷函数
def get_logger(name: str) -> logging.Logger:
    """获取日志记录器的便捷函数"""
    return logger_manager.get_logger(name)

# 预定义的日志记录器
main_logger = get_logger('main')
api_logger = get_logger('api')
search_logger = get_logger('search')
analysis_logger = get_logger('analysis')
kb_logger = get_logger('knowledge_base')
cache_logger = get_logger('cache')