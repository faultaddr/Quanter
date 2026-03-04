"""
邮件通知器

使用SMTP协议发送邮件通知，支持SSL/TLS加密
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from datetime import datetime
from typing import Dict, Any, List, Optional

from ...domain.interfaces.notifier import INotifier
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger

logger = get_logger(__name__)


@registry.register(ComponentType.NOTIFIER, "email")
class EmailNotifier(INotifier):
    """
    邮件通知器

    特点:
    - 支持SMTP服务器发送
    - 支持SSL/TLS加密
    - 支持多个收件人
    - HTML格式邮件

    配置示例:
    {
        "smtp_server": "smtp.qq.com",
        "smtp_port": 465,
        "username": "your@qq.com",
        "password": "授权码",
        "to_emails": ["target@example.com"],
        "from_name": "QuantTool",  # 可选
        "use_ssl": True,  # 可选，默认True
    }
    """

    def __init__(self):
        """初始化邮件通知器"""
        self._initialized = False
        self._config: Dict[str, Any] = {}
        self._smtp_server: Optional[str] = None
        self._smtp_port: int = 465
        self._username: Optional[str] = None
        self._password: Optional[str] = None
        self._to_emails: List[str] = []
        self._from_name: str = "QuantTool"
        self._use_ssl: bool = True

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        初始化通知器

        Args:
            config: 配置字典
                - smtp_server: SMTP服务器地址 (必填)
                - smtp_port: SMTP端口 (默认465)
                - username: 发件邮箱账号 (必填)
                - password: 邮箱授权码 (必填)
                - to_emails: 收件人列表 (必填)
                - from_name: 发件人名称 (可选，默认QuantTool)
                - use_ssl: 是否使用SSL (可选，默认True)

        Raises:
            ValueError: 缺少必要配置项
        """
        required_fields = ['smtp_server', 'username', 'password', 'to_emails']
        missing = [f for f in required_fields if not config.get(f)]
        if missing:
            raise ValueError(f"缺少必要配置: {missing}")

        self._smtp_server = config['smtp_server']
        self._smtp_port = config.get('smtp_port', 465)
        self._username = config['username']
        self._password = config['password']
        self._to_emails = config['to_emails']
        self._from_name = config.get('from_name', 'QuantTool')
        self._use_ssl = config.get('use_ssl', True)

        # 确保to_emails是列表
        if isinstance(self._to_emails, str):
            self._to_emails = [self._to_emails]

        self._config = config
        self._initialized = True
        logger.info(f"EmailNotifier initialized for {self._username}")

    def send_notification(self, message: str, subject: str = None, **kwargs) -> bool:
        """
        发送邮件通知

        Args:
            message: 邮件正文
            subject: 邮件主题
            **kwargs: 额外参数
                - html: 是否为HTML格式 (默认True)
                - to_emails: 覆盖收件人列表

        Returns:
            True 发送成功，False 发送失败
        """
        if not self._initialized:
            logger.error("EmailNotifier not initialized")
            return False

        # 获取收件人
        to_emails = kwargs.get('to_emails', self._to_emails)
        if isinstance(to_emails, str):
            to_emails = [to_emails]

        # 获取主题
        if not subject:
            subject = f"[QuantTool] 信号通知 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # 创建邮件
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = formataddr((self._from_name, self._username))
        msg['To'] = ', '.join(to_emails)

        # 构建HTML内容
        html_content = self._build_html_content(message, subject)
        text_content = self._build_text_content(message, subject)

        msg.attach(MIMEText(text_content, 'plain', 'utf-8'))
        msg.attach(MIMEText(html_content, 'html', 'utf-8'))

        # 发送邮件
        try:
            if self._use_ssl:
                with smtplib.SMTP_SSL(self._smtp_server, self._smtp_port) as server:
                    server.login(self._username, self._password)
                    server.sendmail(self._username, to_emails, msg.as_string())
            else:
                with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                    server.starttls()
                    server.login(self._username, self._password)
                    server.sendmail(self._username, to_emails, msg.as_string())

            logger.info(f"Email sent successfully to {to_emails}")
            return True

        except smtplib.SMTPException as e:
            logger.error(f"Failed to send email: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending email: {e}")
            return False

    def _build_html_content(self, message: str, subject: str) -> str:
        """构建HTML格式邮件内容"""
        # 检测信号类型
        is_buy = '买入' in (subject or '') or 'BUY' in (subject or '').upper()
        is_sell = '卖出' in (subject or '') or 'SELL' in (subject or '').upper()

        # 选择颜色
        if is_buy:
            bg_color = '#28a745'
            icon = '📈'
        elif is_sell:
            bg_color = '#dc3545'
            icon = '📉'
        else:
            bg_color = '#007bff'
            icon = '📊'

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: {bg_color}; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
        .content {{ background: #f8f9fa; padding: 20px; border-radius: 0 0 8px 8px; }}
        .timestamp {{ color: #6c757d; font-size: 12px; }}
        .footer {{ text-align: center; margin-top: 20px; color: #6c757d; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>{icon} {subject or 'QuantTool 通知'}</h2>
            <div class="timestamp">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        <div class="content">
            <pre style="white-space: pre-wrap; font-family: inherit;">{message}</pre>
        </div>
        <div class="footer">
            <p>由 QuantTool 自动发送</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _build_text_content(self, message: str, subject: str) -> str:
        """构建纯文本格式邮件内容"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        separator = '=' * 50

        text = f"""
{separator}
{subject or 'QuantTool 通知'}
时间: {timestamp}
{separator}

{message}

{separator}
由 QuantTool 自动发送
{separator}
"""
        return text

    def get_name(self) -> str:
        """获取通知器名称"""
        return "email"

    def get_description(self) -> str:
        """获取通知器描述"""
        return "Email notifier using SMTP protocol"

    def notify_signal(self, signal_type: str, symbol: str, score: float, details: Dict = None) -> None:
        """
        发送信号通知 (便捷方法)

        Args:
            signal_type: 信号类型 ('buy' 或 'sell')
            symbol: 股票代码
            score: 评分
            details: 详细信息
        """
        is_buy = signal_type.lower() == 'buy'

        message = f"""
{'🟢 买入信号' if is_buy else '🔴 卖出信号'}

股票: {symbol}
评分: {score:.1f}
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        if details:
            message += "\n详情:\n"
            for key, value in details.items():
                if isinstance(value, float):
                    message += f"  - {key}: {value:.2f}\n"
                else:
                    message += f"  - {key}: {value}\n"

        subject = f"[{'买入' if is_buy else '卖出'}] {symbol} 评分{score:.0f}"
        self.send_notification(message.strip(), subject)