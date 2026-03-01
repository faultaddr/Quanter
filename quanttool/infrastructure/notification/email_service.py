"""邮件通知服务."""

import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict, Any
from datetime import date
from jinja2 import Template

from quanttool.core.logging import get_logger

logger = get_logger(__name__)


# 邮件模板
EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #1a1a1a;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #2c3e50;
            margin-top: 30px;
            border-left: 4px solid #4CAF50;
            padding-left: 10px;
        }
        h3 {
            color: #34495e;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: 600;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .positive {
            color: #e74c3c;
            font-weight: 600;
        }
        .negative {
            color: #27ae60;
            font-weight: 600;
        }
        .metric {
            display: inline-block;
            margin: 10px 15px 10px 0;
            padding: 10px 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
        }
        .metric-value {
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
        }
        .status-good {
            color: #27ae60;
        }
        .status-warning {
            color: #f39c12;
        }
        .status-bad {
            color: #e74c3c;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #999;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: Monaco, Consolas, monospace;
            font-size: 13px;
        }
    </style>
</head>
<body>
    {{ content | safe }}

    <div class="footer">
        <p>本报告由 QuantTool 自动生成</p>
        <p>发送时间: {{ send_time }}</p>
    </div>
</body>
</html>
"""


class EmailService:
    """邮件通知服务."""

    def __init__(
        self,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls

    async def send_email(
        self,
        recipients: List[str],
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
    ) -> bool:
        """发送邮件.

        Args:
            recipients: 收件人列表
            subject: 邮件主题
            body_text: 纯文本内容
            body_html: HTML 内容（可选）

        Returns:
            是否发送成功
        """
        if not self.username or not self.password:
            logger.error("邮件配置不完整，无法发送邮件")
            return False

        # 创建邮件
        message = MIMEMultipart("alternative")
        message["From"] = self.username
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject

        # 添加纯文本内容
        message.attach(MIMEText(body_text, "plain", "utf-8"))

        # 添加 HTML 内容（如果有）
        if body_html:
            message.attach(MIMEText(body_html, "html", "utf-8"))

        try:
            # 连接 SMTP 服务器并发送
            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.username,
                password=self.password,
                use_tls=self.use_tls,
            )
            logger.info(f"邮件已发送至: {recipients}")
            return True

        except Exception as e:
            logger.error(f"发送邮件失败: {e}")
            return False

    async def send_daily_report(
        self,
        report_date: date,
        report_content: str,
        recipients: List[str],
    ) -> bool:
        """发送每日报告邮件.

        Args:
            report_date: 报告日期
            report_content: 报告内容（Markdown 格式）
            recipients: 收件人列表

        Returns:
            是否发送成功
        """
        # 转换 Markdown 为 HTML
        html_content = self._markdown_to_html(report_content)

        # 使用模板
        template = Template(EMAIL_TEMPLATE)
        body_html = template.render(
            content=html_content,
            send_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # 邮件主题
        subject = f"QuantTool 每日投资报告 - {report_date.strftime('%Y-%m-%d')}"

        return await self.send_email(
            recipients=recipients,
            subject=subject,
            body_text=report_content,  # 纯文本版本
            body_html=body_html,  # HTML 版本
        )

    def _markdown_to_html(self, markdown_text: str) -> str:
        """简单的 Markdown 到 HTML 转换.

        这里实现基础的转换，复杂场景建议使用 markdown 库。
        """
        import re

        html = markdown_text

        # 转换标题
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # 转换粗体
        html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # 转换行内代码
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

        # 转换表格（简化处理）
        lines = html.split('\n')
        in_table = False
        table_lines = []
        result_lines = []

        for line in lines:
            if '|' in line and not line.startswith('#') and not line.startswith('<h'):
                if not in_table:
                    in_table = True
                    table_lines = ['<table>']

                # 跳过分隔线
                if '---' in line or ':-:' in line or '---:' in line:
                    continue

                # 解析行
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    # 检测是否是表头（第一行）
                    if len(table_lines) == 1:
                        table_lines.append('<tr>' + ''.join(f'<th>{cell}</th>' for cell in cells) + '</tr>')
                    else:
                        table_lines.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
            else:
                if in_table:
                    table_lines.append('</table>')
                    result_lines.append(''.join(table_lines))
                    in_table = False
                    table_lines = []
                result_lines.append(line)

        if in_table:
            table_lines.append('</table>')
            result_lines.append(''.join(table_lines))

        html = '\n'.join(result_lines)

        # 转换列表
        html = re.sub(r'^\- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'(<li>.+</li>\n)+', r'<ul>\g<0></ul>', html)

        # 转换换行
        html = html.replace('\n\n', '</p><p>')
        html = html.replace('\n', '<br>')

        # 包装段落
        if not html.startswith('<'):
            html = f'<p>{html}</p>'

        return html

    async def test_connection(self) -> bool:
        """测试邮件服务器连接.

        Returns:
            连接是否成功
        """
        try:
            await aiosmtplib.connect(
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.username,
                password=self.password,
                use_tls=self.use_tls,
            )
            return True
        except Exception as e:
            logger.error(f"邮件服务器连接测试失败: {e}")
            return False
