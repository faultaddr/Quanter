"""
微信通知器

使用Server酱(Serverchan)推送消息到微信
注册地址: https://sct.ftqq.com/
"""
import urllib.request
import urllib.parse
import json
from datetime import datetime
from typing import Dict, Any, Optional

from ...domain.interfaces.notifier import INotifier
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger

logger = get_logger(__name__)


@registry.register(ComponentType.NOTIFIER, "wechat")
class WechatNotifier(INotifier):
    """
    微信通知器 (Server酱)

    特点:
    - 通过Server酱推送消息到微信
    - 支持Markdown格式
    - 简单配置，只需SendKey

    配置示例:
    {
        "sendkey": "SCTxxx...",
    }

    注册方式:
    1. 访问 https://sct.ftqq.com/
    2. 微信扫码登录
    3. 获取SendKey
    """

    # Server酱API地址
    API_URL = "https://sctapi.ftqq.com/{}.send"

    def __init__(self):
        """初始化微信通知器"""
        self._initialized = False
        self._config: Dict[str, Any] = {}
        self._sendkey: Optional[str] = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        初始化通知器

        Args:
            config: 配置字典
                - sendkey: Server酱SendKey (必填)

        Raises:
            ValueError: 缺少必要配置项
        """
        if not config.get('sendkey'):
            raise ValueError("缺少必要配置: sendkey")

        self._sendkey = config['sendkey']
        self._config = config
        self._initialized = True
        logger.info("WechatNotifier initialized")

    def send_notification(self, message: str, subject: str = None, **kwargs) -> bool:
        """
        发送微信通知

        Args:
            message: 通知消息 (支持Markdown)
            subject: 消息标题
            **kwargs: 额外参数
                - channel: 推送渠道 (可选，默认9表示微信)

        Returns:
            True 发送成功，False 发送失败
        """
        if not self._initialized:
            logger.error("WechatNotifier not initialized")
            return False

        # 获取标题
        if not subject:
            subject = f"QuantTool 通知 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # 构建消息内容
        content = self._build_content(message, subject)

        # 发送请求
        try:
            url = self.API_URL.format(self._sendkey)
            data = {
                'title': subject,
                'desp': content,
            }

            # 添加渠道参数
            channel = kwargs.get('channel', 9)
            data['channel'] = str(channel)

            # 发送POST请求
            req = urllib.request.Request(
                url,
                data=urllib.parse.urlencode(data).encode('utf-8'),
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))

                if result.get('code') == 0 or result.get('code') == 200:
                    logger.info(f"Wechat notification sent successfully: {subject}")
                    return True
                else:
                    logger.error(f"Failed to send wechat notification: {result}")
                    return False

        except urllib.error.URLError as e:
            logger.error(f"Network error sending wechat notification: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending wechat notification: {e}")
            return False

    def _build_content(self, message: str, subject: str) -> str:
        """
        构建Markdown格式内容

        Server酱支持Markdown格式，可以更好地展示信息
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 检测信号类型
        is_buy = '买入' in (subject or '') or 'BUY' in (subject or '').upper()
        is_sell = '卖出' in (subject or '') or 'SELL' in (subject or '').upper()

        # 构建Markdown内容
        lines = [
            f"**时间**: {timestamp}",
            "",
            "---",
            "",
        ]

        # 添加信号图标
        if is_buy:
            lines.insert(0, "### 🟢 买入信号\n")
        elif is_sell:
            lines.insert(0, "### 🔴 卖出信号\n")
        else:
            lines.insert(0, "### 📊 信号通知\n")

        # 添加消息内容
        lines.append(message)

        lines.extend([
            "",
            "---",
            "",
            "*由 QuantTool 自动发送*",
        ])

        return "\n".join(lines)

    def get_name(self) -> str:
        """获取通知器名称"""
        return "wechat"

    def get_description(self) -> str:
        """获取通知器描述"""
        return "WeChat notifier using Serverchan API"

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

        # 构建Markdown格式消息
        message_lines = [
            f"**股票**: {symbol}",
            f"**评分**: {score:.1f}",
            f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if details:
            message_lines.append("")
            message_lines.append("**详情**:")
            for key, value in details.items():
                if isinstance(value, float):
                    message_lines.append(f"- {key}: `{value:.2f}`")
                else:
                    message_lines.append(f"- {key}: `{value}`")

        message = "\n".join(message_lines)
        subject = f"[{'买入' if is_buy else '卖出'}] {symbol} 评分{score:.0f}"
        self.send_notification(message, subject)