#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票分析报告邮件发送脚本
自动发送当天生成的股票分析报告到指定邮箱
"""

import os
import smtplib
import glob
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_sender.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_email_config():
    """加载邮件配置，支持多个收件人（逗号分隔）"""
    load_dotenv()
    
    raw_recipients = os.getenv('EMAIL_TO', '')
    recipient_emails = [email.strip() for email in raw_recipients.split(',') if email.strip()]
    
    config = {
        'smtp_server': os.getenv('SMTP_SERVER'),
        'smtp_port': int(os.getenv('SMTP_PORT', 587)),
        'email_user': os.getenv('SMTP_USER'),
        'email_password': os.getenv('SMTP_PASS'),
        'recipient_emails': recipient_emails,
        'smtp_tls': os.getenv('SMTP_TLS', 'True').lower() == 'true'
    }
    
    # 验证配置
    missing_configs = [key for key, value in config.items() if not value and key != 'recipient_emails']
    # 专门检查收件人列表
    if not recipient_emails:
        missing_configs.append('recipient_emails')
    if missing_configs:
        raise ValueError(f"缺少邮件配置: {', '.join(missing_configs)}")
    
    return config

def get_today_reports():
    """获取当天的股票分析报告文件"""
    today = datetime.now().strftime('%Y%m%d')
    public_dir = '/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/public'
    
    # 查找当天的报告文件
    pattern = os.path.join(public_dir, f'*_{today}.md')
    report_files = glob.glob(pattern)
    
    logger.info(f"找到 {len(report_files)} 个当天报告文件")
    for file in report_files:
        logger.info(f"报告文件: {os.path.basename(file)}")
    
    return report_files

def create_email_content(report_files):
    """创建邮件内容"""
    today = datetime.now().strftime('%Y年%m月%d日')
    
    # 邮件主题
    subject = f"股票分析报告 - {today}"
    
    # 邮件正文
    body = f"""
亲爱的用户，

附件是 {today} 的股票分析报告，共包含 {len(report_files)} 只股票的分析结果。

报告包含以下股票：
"""
    
    # 添加股票列表
    for file_path in report_files:
        filename = os.path.basename(file_path)
        stock_code = filename.split('_')[0]
        body += f"- {stock_code}\n"
    
    body += """
请查看附件获取详细的技术分析和投资建议。

注意：本报告仅供参考，投资有风险，入市需谨慎。

祝好！
股票分析系统
"""
    
    return subject, body

def send_email(config, report_files):
    """发送邮件"""
    if not report_files:
        logger.warning("没有找到当天的报告文件，跳过邮件发送")
        return False
    
    try:
        # 创建邮件内容
        subject, body = create_email_content(report_files)
        
        # 创建邮件对象
        msg = MIMEMultipart()
        msg['From'] = config['email_user']
        msg['To'] = ", ".join(config['recipient_emails'])
        msg['Subject'] = subject
        
        # 添加邮件正文
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 添加附件
        for file_path in report_files:
            with open(file_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(file_path)}'
                )
                msg.attach(part)
        
        # 连接SMTP服务器并发送邮件
        logger.info(f"连接SMTP服务器: {config['smtp_server']}:{config['smtp_port']}")
        
        if config['smtp_port'] == 465:
            # 使用SSL连接
            server = smtplib.SMTP_SSL(config['smtp_server'], config['smtp_port'])
        else:
            # 使用TLS连接
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            if config.get('smtp_tls', True):
                server.starttls()  # 启用TLS加密
        
        server.login(config['email_user'], config['email_password'])
        
        # 发送邮件
        text = msg.as_string()
        server.sendmail(config['email_user'], config['recipient_emails'], text)
        server.quit()
        
        logger.info(f"邮件发送成功！收件人: {', '.join(config['recipient_emails'])}")
        return True
        
    except Exception as e:
        logger.error(f"邮件发送失败: {str(e)}")
        return False

def main():
    """主函数"""
    try:
        logger.info("开始执行股票分析报告邮件发送任务")
        
        # 加载邮件配置
        config = load_email_config()
        logger.info("邮件配置加载成功")
        
        # 获取当天报告文件
        report_files = get_today_reports()
        
        # 发送邮件
        success = send_email(config, report_files)
        
        if success:
            logger.info("股票分析报告邮件发送任务完成")
        else:
            logger.error("股票分析报告邮件发送任务失败")
            
    except Exception as e:
        logger.error(f"任务执行失败: {str(e)}")
        raise

if __name__ == '__main__':
    main()