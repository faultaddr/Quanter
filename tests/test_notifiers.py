"""
通知器单元测试
"""
import pytest
from unittest.mock import patch, MagicMock
import smtplib

from quanttool.infrastructure.notifiers import EmailNotifier, WechatNotifier


class TestEmailNotifier:
    """邮件通知器测试"""

    def test_get_name(self):
        """测试获取名称"""
        notifier = EmailNotifier()
        assert notifier.get_name() == "email"

    def test_get_description(self):
        """测试获取描述"""
        notifier = EmailNotifier()
        assert "Email" in notifier.get_description()

    def test_initialize_missing_config(self):
        """测试缺少必要配置"""
        notifier = EmailNotifier()
        with pytest.raises(ValueError) as exc_info:
            notifier.initialize({})
        assert "缺少必要配置" in str(exc_info.value)

    def test_initialize_partial_config(self):
        """测试部分配置"""
        notifier = EmailNotifier()
        with pytest.raises(ValueError) as exc_info:
            notifier.initialize({"smtp_server": "smtp.qq.com"})
        assert "缺少必要配置" in str(exc_info.value)

    def test_initialize_success(self):
        """测试初始化成功"""
        notifier = EmailNotifier()
        config = {
            "smtp_server": "smtp.qq.com",
            "smtp_port": 465,
            "username": "test@qq.com",
            "password": "test_password",
            "to_emails": ["target@example.com"],
        }
        notifier.initialize(config)
        assert notifier._initialized is True
        assert notifier._smtp_server == "smtp.qq.com"
        assert notifier._smtp_port == 465
        assert notifier._username == "test@qq.com"
        assert notifier._to_emails == ["target@example.com"]

    def test_initialize_with_string_to_emails(self):
        """测试字符串形式的收件人转换为列表"""
        notifier = EmailNotifier()
        config = {
            "smtp_server": "smtp.qq.com",
            "username": "test@qq.com",
            "password": "test_password",
            "to_emails": "single@example.com",
        }
        notifier.initialize(config)
        assert notifier._to_emails == ["single@example.com"]

    def test_send_without_initialize(self):
        """测试未初始化时发送"""
        notifier = EmailNotifier()
        result = notifier.send_notification("test message")
        assert result is False

    @patch('smtplib.SMTP_SSL')
    def test_send_notification_success(self, mock_smtp_ssl):
        """测试发送邮件成功"""
        notifier = EmailNotifier()
        config = {
            "smtp_server": "smtp.qq.com",
            "smtp_port": 465,
            "username": "test@qq.com",
            "password": "test_password",
            "to_emails": ["target@example.com"],
        }
        notifier.initialize(config)

        # 模拟SMTP服务器
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server

        result = notifier.send_notification("测试消息", "测试主题")
        assert result is True
        mock_server.login.assert_called_once_with("test@qq.com", "test_password")
        mock_server.sendmail.assert_called_once()

    @patch('smtplib.SMTP_SSL')
    def test_send_notification_smtp_error(self, mock_smtp_ssl):
        """测试SMTP错误"""
        notifier = EmailNotifier()
        config = {
            "smtp_server": "smtp.qq.com",
            "username": "test@qq.com",
            "password": "test_password",
            "to_emails": ["target@example.com"],
        }
        notifier.initialize(config)

        mock_server = MagicMock()
        mock_server.login.side_effect = smtplib.SMTPException("Login failed")
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server

        result = notifier.send_notification("测试消息")
        assert result is False

    @patch('smtplib.SMTP_SSL')
    def test_notify_signal_buy(self, mock_smtp_ssl):
        """测试买入信号通知"""
        notifier = EmailNotifier()
        config = {
            "smtp_server": "smtp.qq.com",
            "username": "test@qq.com",
            "password": "test_password",
            "to_emails": ["target@example.com"],
        }
        notifier.initialize(config)

        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server

        notifier.notify_signal("buy", "000001", 85.5, {"趋势": 90.0})
        mock_server.sendmail.assert_called_once()

    @patch('smtplib.SMTP_SSL')
    def test_notify_signal_sell(self, mock_smtp_ssl):
        """测试卖出信号通知"""
        notifier = EmailNotifier()
        config = {
            "smtp_server": "smtp.qq.com",
            "username": "test@qq.com",
            "password": "test_password",
            "to_emails": ["target@example.com"],
        }
        notifier.initialize(config)

        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server

        notifier.notify_signal("sell", "600519", 30.0)
        mock_server.sendmail.assert_called_once()

    @patch('smtplib.SMTP')
    def test_send_without_ssl(self, mock_smtp):
        """测试不使用SSL发送"""
        notifier = EmailNotifier()
        config = {
            "smtp_server": "smtp.qq.com",
            "smtp_port": 587,
            "username": "test@qq.com",
            "password": "test_password",
            "to_emails": ["target@example.com"],
            "use_ssl": False,
        }
        notifier.initialize(config)

        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        result = notifier.send_notification("测试消息")
        assert result is True
        mock_server.starttls.assert_called_once()


class TestWechatNotifier:
    """微信通知器测试"""

    def test_get_name(self):
        """测试获取名称"""
        notifier = WechatNotifier()
        assert notifier.get_name() == "wechat"

    def test_get_description(self):
        """测试获取描述"""
        notifier = WechatNotifier()
        assert "WeChat" in notifier.get_description() or "Serverchan" in notifier.get_description()

    def test_initialize_missing_config(self):
        """测试缺少必要配置"""
        notifier = WechatNotifier()
        with pytest.raises(ValueError) as exc_info:
            notifier.initialize({})
        assert "sendkey" in str(exc_info.value)

    def test_initialize_success(self):
        """测试初始化成功"""
        notifier = WechatNotifier()
        config = {"sendkey": "SCTtest123"}
        notifier.initialize(config)
        assert notifier._initialized is True
        assert notifier._sendkey == "SCTtest123"

    def test_send_without_initialize(self):
        """测试未初始化时发送"""
        notifier = WechatNotifier()
        result = notifier.send_notification("test message")
        assert result is False

    @patch('urllib.request.urlopen')
    def test_send_notification_success(self, mock_urlopen):
        """测试发送微信通知成功"""
        notifier = WechatNotifier()
        notifier.initialize({"sendkey": "SCTtest123"})

        # 模拟API响应
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"code": 0, "message": "success"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = notifier.send_notification("测试消息", "测试主题")
        assert result is True
        mock_urlopen.assert_called_once()

    @patch('urllib.request.urlopen')
    def test_send_notification_api_error(self, mock_urlopen):
        """测试API返回错误"""
        notifier = WechatNotifier()
        notifier.initialize({"sendkey": "SCTtest123"})

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"code": 40001, "message": "invalid sendkey"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = notifier.send_notification("测试消息")
        assert result is False

    @patch('urllib.request.urlopen')
    def test_send_notification_network_error(self, mock_urlopen):
        """测试网络错误"""
        import urllib.error
        notifier = WechatNotifier()
        notifier.initialize({"sendkey": "SCTtest123"})

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        result = notifier.send_notification("测试消息")
        assert result is False

    @patch('urllib.request.urlopen')
    def test_notify_signal_buy(self, mock_urlopen):
        """测试买入信号通知"""
        notifier = WechatNotifier()
        notifier.initialize({"sendkey": "SCTtest123"})

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"code": 0}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        notifier.notify_signal("buy", "000001", 85.5, {"趋势": 90.0})
        mock_urlopen.assert_called_once()

    @patch('urllib.request.urlopen')
    def test_notify_signal_sell(self, mock_urlopen):
        """测试卖出信号通知"""
        notifier = WechatNotifier()
        notifier.initialize({"sendkey": "SCTtest123"})

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"code": 0}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        notifier.notify_signal("sell", "600519", 30.0)
        mock_urlopen.assert_called_once()

    def test_build_content_buy_signal(self):
        """测试构建买入信号内容"""
        notifier = WechatNotifier()
        content = notifier._build_content("测试消息", "[买入] 000001")
        assert "买入" in content
        assert "🟢" in content

    def test_build_content_sell_signal(self):
        """测试构建卖出信号内容"""
        notifier = WechatNotifier()
        content = notifier._build_content("测试消息", "[卖出] 600519")
        assert "卖出" in content
        assert "🔴" in content