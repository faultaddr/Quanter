"""
控制台通知器

在终端输出信号通知，支持彩色显示
"""
from datetime import datetime
from typing import Dict, Any

from ...domain.interfaces.notifier import INotifier
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger

logger = get_logger(__name__)

# ANSI颜色代码
class Colors:
    """终端颜色代码"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'


@registry.register(ComponentType.NOTIFIER, "console")
class ConsoleNotifier(INotifier):
    """
    控制台通知器

    特点:
    - 彩色输出
    - 清晰格式化
    - 时间戳显示
    """

    def __init__(self):
        """初始化控制台通知器"""
        self._initialized = False
        self._config: Dict[str, Any] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        初始化通知器

        Args:
            config: 配置字典
                - use_colors: 是否使用颜色 (默认True)
                - show_timestamp: 是否显示时间戳 (默认True)
        """
        self._config = config or {}
        self._initialized = True
        logger.info("ConsoleNotifier initialized")

    def send_notification(self, message: str, subject: str = None, **kwargs) -> bool:
        """
        发送通知到控制台

        Args:
            message: 通知消息
            subject: 主题 (可选)

        Returns:
            True (总是成功)
        """
        use_colors = self._config.get('use_colors', True)
        show_timestamp = self._config.get('show_timestamp', True)

        # 获取信号方向
        is_buy = '买入' in (subject or '') or 'BUY' in (subject or '').upper()
        is_sell = '卖出' in (subject or '') or 'SELL' in (subject or '').upper()

        # 构建输出
        lines = []

        # 分隔线
        separator = "=" * 60
        if use_colors:
            separator = f"{Colors.CYAN}{separator}{Colors.RESET}"
        lines.append(separator)

        # 时间戳
        if show_timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if use_colors:
                timestamp = f"{Colors.YELLOW}[{timestamp}]{Colors.RESET}"
            else:
                timestamp = f"[{timestamp}]"
            lines.append(timestamp)

        # 主题
        if subject:
            if use_colors:
                if is_buy:
                    subject = f"{Colors.BG_GREEN}{Colors.WHITE} {subject} {Colors.RESET}"
                elif is_sell:
                    subject = f"{Colors.BG_RED}{Colors.WHITE} {subject} {Colors.RESET}"
                else:
                    subject = f"{Colors.BOLD}{Colors.WHITE}{subject}{Colors.RESET}"
            lines.append(subject)

        # 消息内容
        if use_colors:
            # 为消息内容添加颜色
            for line in message.split('\n'):
                if '买入' in line or 'BUY' in line.upper():
                    line = f"{Colors.GREEN}{line}{Colors.RESET}"
                elif '卖出' in line or 'SELL' in line.upper():
                    line = f"{Colors.RED}{line}{Colors.RESET}"
                elif '评分' in line:
                    line = f"{Colors.BLUE}{line}{Colors.RESET}"
                lines.append(line)
        else:
            lines.append(message)

        # 结束分隔线
        lines.append(separator)

        # 输出
        print('\n'.join(lines))

        return True

    def get_name(self) -> str:
        """获取通知器名称"""
        return "console"

    def get_description(self) -> str:
        """获取通知器描述"""
        return "Console notifier with colored output"

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