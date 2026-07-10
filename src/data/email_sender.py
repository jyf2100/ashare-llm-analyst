"""
邮件发送模块

将生成的分析报告通过邮件发送给指定用户
"""

import glob
import os
import smtplib
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)


class EmailSender(AnalyzerBase):
    """
    邮件发送器

    通过SMTP发送分析报告邮件

    Example:
        >>> sender = EmailSender()
        >>> result = sender.send_reports(recipients=["user@example.com"])
        >>> print(f"发送 {result['sent_count']} 封邮件")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化邮件发送器

        Args:
            config: 配置实例
        """
        super().__init__(config)
        self.reports_dir = self.config.data.reports_dir
        self.smtp_config = self._load_email_config()

    def _load_email_config(self) -> Optional[Dict[str, Any]]:
        """
        从.env加载SMTP配置

        Returns:
            SMTP配置字典，配置缺失返回None
        """
        # 加载.env文件
        for env_path in [".env", "out/.env", "../.env"]:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                break

        # 读取配置
        smtp_server = os.getenv('SMTP_SERVER')
        smtp_port = os.getenv('SMTP_PORT')
        smtp_user = os.getenv('SMTP_USER')
        smtp_pass = os.getenv('SMTP_PASS')
        email_to = os.getenv('EMAIL_TO')
        smtp_tls = os.getenv('SMTP_TLS', 'True')

        if not all([smtp_server, smtp_port, smtp_user, smtp_pass]):
            logger.warning("SMTP配置不完整")
            return None

        # 解析收件人列表
        recipient_emails = []
        if email_to:
            recipient_emails = [email.strip() for email in email_to.split(',') if email.strip()]

        return {
            'smtp_server': smtp_server,
            'smtp_port': int(smtp_port),
            'email_user': smtp_user,
            'email_password': smtp_pass,
            'recipient_emails': recipient_emails,
            'smtp_tls': smtp_tls.lower() == 'true',
        }

    def get_today_reports(self) -> List[str]:
        """
        获取当天生成的报告文件

        Returns:
            报告文件路径列表
        """
        today = datetime.now().strftime('%Y%m%d')

        # 查找当天的报告文件（支持多种格式）
        patterns = [
            os.path.join(self.reports_dir, f'*_{today}*.md'),
            os.path.join(self.reports_dir, f'*_{today}*.html'),
            os.path.join(self.reports_dir, 'public', f'*_{today}*.md'),
        ]

        report_files = []
        for pattern in patterns:
            report_files.extend(glob.glob(pattern))

        logger.info(f"找到 {len(report_files)} 个当天报告文件")
        return report_files

    def create_email_content(
        self,
        report_files: List[str]
    ) -> tuple[str, str]:
        """
        创建邮件内容

        Args:
            report_files: 报告文件列表

        Returns:
            (主题, 正文) 元组
        """
        today = datetime.now().strftime('%Y年%m月%d日')

        # 邮件主题
        subject = f"股票分析报告 - {today}"

        # 邮件正文
        body_lines = [
            "亲爱的用户，",
            "",
            f"附件是 {today} 的股票分析报告，共包含 {len(report_files)} 只股票的分析结果。",
            "",
            "报告包含以下股票：",
        ]

        # 添加股票列表
        for file_path in report_files:
            filename = os.path.basename(file_path)
            stock_code = filename.split('_')[0]
            body_lines.append(f"- {stock_code}")

        body_lines.extend([
            "",
            "请查看附件获取详细的技术分析和投资建议。",
            "",
            "注意：本报告仅供参考，投资有风险，入市需谨慎。",
            "",
            "祝好！",
            "股票分析系统",
        ])

        body = "\n".join(body_lines)

        return subject, body

    def send_email(
        self,
        recipients: List[str],
        report_files: List[str]
    ) -> bool:
        """
        发送邮件

        Args:
            recipients: 收件人列表
            report_files: 报告文件列表

        Returns:
            成功返回True，失败返回False
        """
        if not report_files:
            logger.warning("没有报告文件需要发送")
            return True

        if not recipients:
            logger.warning("没有收件人，跳过邮件发送")
            return False

        try:
            # 创建邮件内容
            subject, body = self.create_email_content(report_files)

            # 创建邮件对象
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['email_user']
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject

            # 添加邮件正文
            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # 添加附件
            for file_path in report_files:
                try:
                    with open(file_path, 'rb') as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {os.path.basename(file_path)}'
                        )
                        msg.attach(part)
                except Exception as e:
                    logger.error(f"添加附件失败 {file_path}: {e}")
                    continue

            # 连接SMTP服务器并发送
            logger.info(f"连接SMTP服务器: {self.smtp_config['smtp_server']}:{self.smtp_config['smtp_port']}")

            if self.smtp_config['smtp_port'] == 465:
                # 使用SSL连接
                server = smtplib.SMTP_SSL(
                    self.smtp_config['smtp_server'],
                    self.smtp_config['smtp_port']
                )
            else:
                # 使用TLS连接
                server = smtplib.SMTP(
                    self.smtp_config['smtp_server'],
                    self.smtp_config['smtp_port']
                )
                if self.smtp_config.get('smtp_tls', True):
                    server.starttls()

            server.login(
                self.smtp_config['email_user'],
                self.smtp_config['email_password']
            )

            # 发送邮件
            text = msg.as_string()
            server.sendmail(
                self.smtp_config['email_user'],
                recipients,
                text
            )
            server.quit()

            logger.info(f"邮件发送成功！收件人: {', '.join(recipients)}")
            return True

        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            return False

    def send_reports(
        self,
        recipients: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        执行邮件发送流程

        Args:
            recipients: 收件人列表，None则使用配置中的收件人

        Returns:
            执行结果字典
        """
        logger.info("开始发送邮件报告")

        # 检查SMTP配置
        if not self.smtp_config:
            logger.error("SMTP配置缺失，无法发送邮件")
            return {
                "success": False,
                "sent_count": 0,
                "error": "config_missing"
            }

        # 获取报告文件
        report_files = self.get_today_reports()

        if not report_files:
            logger.warning("没有找到报告文件")
            return {
                "success": True,
                "sent_count": 0,
                "message": "没有报告文件"
            }

        # 确定收件人
        if recipients is None:
            recipients = self.smtp_config.get('recipient_emails', [])

        if not recipients:
            logger.warning("没有收件人，跳过邮件发送")
            return {
                "success": True,
                "sent_count": 0,
                "message": "没有收件人"
            }

        # 发送邮件
        success = self.send_email(recipients, report_files)

        if success:
            logger.info(f"邮件发送完成: {len(report_files)} 个报告文件")
            return {
                "success": True,
                "sent_count": len(report_files),
                "message": f"发送 {len(report_files)} 个报告"
            }
        else:
            return {
                "success": False,
                "sent_count": 0,
                "error": "send_failed"
            }
