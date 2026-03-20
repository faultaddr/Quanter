"""
监控命令

提供实时信号监控服务
"""
import asyncio
import os
from datetime import datetime
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ...application.realtime_monitor_service import RealtimeMonitorService, MonitorConfig
from ...infrastructure.data_providers.real_data_provider import RealAShareDataProvider
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger

# 确保通知器模块被导入并注册
from ...infrastructure.notifiers import ConsoleNotifier, EmailNotifier, WechatNotifier

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Realtime signal monitoring commands")


def _get_notifier(name: str, config: dict = None):
    """获取通知器实例"""
    try:
        notifier_cls = registry.get(ComponentType.NOTIFIER, name)
        notifier = notifier_cls()
        if config:
            notifier.initialize(config)
        else:
            notifier.initialize({})
        return notifier
    except ValueError as e:
        console.print(f"[red]Error loading notifier '{name}': {e}[/red]")
        return None


def _load_notifier_config():
    """从配置文件加载通知器配置"""
    from ...config.settings import settings

    notifiers = []

    # 微信通知
    wechat_config = settings.get("notification.wechat", {})
    if wechat_config.get("enabled") and wechat_config.get("sendkey"):
        notifier = _get_notifier("wechat", {"sendkey": wechat_config["sendkey"]})
        if notifier:
            notifiers.append(notifier)
            console.print("[green]✓[/green] WeChat notifier enabled")

    # 邮件通知
    email_config = settings.get("notification.email", {})
    if email_config.get("enabled") and email_config.get("username"):
        notifier = _get_notifier("email", {
            "smtp_server": email_config.get("smtp_server"),
            "smtp_port": email_config.get("smtp_port", 465),
            "username": email_config.get("username"),
            "password": email_config.get("password"),
            "to_emails": email_config.get("to_emails", []),
            "from_name": email_config.get("from_name", "QuantTool"),
            "use_ssl": email_config.get("use_ssl", True),
        })
        if notifier:
            notifiers.append(notifier)
            console.print("[green]✓[/green] Email notifier enabled")

    return notifiers


@app.command("start")
def start_monitor(
    symbols: str = typer.Option("000001,600519", "--symbols", "-s",
                                 help="Comma-separated stock symbols to monitor"),
    interval: int = typer.Option(5, "--interval", "-i",
                                  help="Check interval in minutes"),
    strategy: str = typer.Option("breakout", "--strategy", "-S",
                                  help="Scoring strategy: breakout, trend, momentum, or qlib"),
    buy_threshold: int = typer.Option(50, "--buy", "-b",
                                       help="Buy signal threshold"),
    sell_threshold: int = typer.Option(40, "--sell", "-B",
                                        help="Sell signal threshold"),
    notifiers: str = typer.Option("console", "--notifiers", "-n",
                                   help="Comma-separated notifiers: console,email,wechat"),
    wechat_key: Optional[str] = typer.Option(None, "--wechat-key", "-w",
                                              help="Serverchan SendKey (or set SERVERCHAN_SENDKEY env)"),
    email_to: Optional[str] = typer.Option(None, "--email-to", "-e",
                                            help="Email recipients (comma-separated)"),
    trading_only: bool = typer.Option(True, "--trading-only/--all-day",
                                       help="Only run during trading hours"),
    cooldown: int = typer.Option(15, "--cooldown", "-c",
                                  help="Cooldown period in minutes between signals"),
):
    """
    Start realtime signal monitoring service.

    Examples:
        # Monitor stocks with console output only
        quanttool monitor start -s 000001,600519

        # Monitor with WeChat notifications
        quanttool monitor start -s 000001,600519 -n console,wechat -w SCTxxx

        # Monitor with email notifications
        quanttool monitor start -s 000001,600519 -n console,email -e your@email.com
    """
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(",")]

    # Build notifier list
    notifier_instances = []
    notifier_names = [n.strip() for n in notifiers.split(",")]

    for name in notifier_names:
        if name == "console":
            notifier = _get_notifier("console")
            if notifier:
                notifier_instances.append(notifier)
                console.print("[green]✓[/green] Console notifier enabled")

        elif name == "wechat":
            # Check for SendKey
            sendkey = wechat_key or os.environ.get("SERVERCHAN_SENDKEY", "")
            if not sendkey:
                # Try config file
                from ...config.settings import settings
                sendkey = settings.get("notification.wechat.sendkey", "")

            if sendkey:
                notifier = _get_notifier("wechat", {"sendkey": sendkey})
                if notifier:
                    notifier_instances.append(notifier)
                    console.print("[green]✓[/green] WeChat notifier enabled")
            else:
                console.print("[yellow]⚠[/yellow] WeChat notifier skipped: no SendKey provided")
                console.print("    Set --wechat-key or SERVERCHAN_SENDKEY environment variable")

        elif name == "email":
            # Get email config
            from ...config.settings import settings

            email_config = settings.get("notification.email", {})

            smtp_server = os.environ.get("SMTP_SERVER") or email_config.get("smtp_server")
            username = os.environ.get("SMTP_USERNAME") or email_config.get("username")
            password = os.environ.get("SMTP_PASSWORD") or email_config.get("password")
            to_emails = email_to.split(",") if email_to else email_config.get("to_emails", [])

            if smtp_server and username and password and to_emails:
                notifier = _get_notifier("email", {
                    "smtp_server": smtp_server,
                    "smtp_port": int(os.environ.get("SMTP_PORT", 465)),
                    "username": username,
                    "password": password,
                    "to_emails": to_emails,
                })
                if notifier:
                    notifier_instances.append(notifier)
                    console.print("[green]✓[/green] Email notifier enabled")
            else:
                console.print("[yellow]⚠[/yellow] Email notifier skipped: incomplete config")
                console.print("    Set SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD env vars or --email-to")

    if not notifier_instances:
        console.print("[red]No notifiers configured, adding console notifier[/red]")
        notifier_instances.append(_get_notifier("console"))

    # Create monitor config
    config = MonitorConfig(
        symbols=symbol_list,
        interval_minutes=interval,
        strategy=strategy,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        trading_hours_only=trading_only,
        cooldown_minutes=cooldown,
    )

    # 初始化数据提供者
    data_provider = RealAShareDataProvider(
        primary_source="baostock",  # baostock 不需要 token
        use_fallback=True
    )
    data_provider.initialize()

    # Create and start service
    service = RealtimeMonitorService(
        config=config,
        data_provider=data_provider,
        notifiers=notifier_instances,
    )

    # Print config summary
    console.print(Panel(
        f"[bold]Monitoring Configuration[/bold]\n"
        f"Symbols: {', '.join(symbol_list)}\n"
        f"Strategy: {strategy}\n"
        f"Interval: {interval} min\n"
        f"Buy threshold: {buy_threshold}\n"
        f"Sell threshold: {sell_threshold}\n"
        f"Notifiers: {', '.join(n.get_name() for n in notifier_instances)}\n"
        f"Trading hours only: {trading_only}",
        title="[bold blue]QuantTool Monitor[/bold blue]",
        border_style="blue",
    ))

    console.print("\n[bold green]Starting monitor service...[/bold green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping monitor service...[/yellow]")
        asyncio.run(service.stop())
        console.print("[green]Monitor stopped.[/green]")


@app.command("status")
def show_status():
    """Show current monitor configuration."""
    from ...config.settings import settings

    console.print("\n[bold]Current Notification Configuration[/bold]\n")

    # WeChat
    wechat_config = settings.get("notification.wechat", {})
    sendkey = os.environ.get("SERVERCHAN_SENDKEY") or wechat_config.get("sendkey", "")
    console.print(f"WeChat: {'[green]✓ Configured[/green]' if sendkey else '[yellow]Not configured[/yellow]'}")

    # Email
    email_config = settings.get("notification.email", {})
    smtp_server = os.environ.get("SMTP_SERVER") or email_config.get("smtp_server", "")
    username = os.environ.get("SMTP_USERNAME") or email_config.get("username", "")
    configured = bool(smtp_server and username)
    console.print(f"Email: {'[green]✓ Configured[/green]' if configured else '[yellow]Not configured[/yellow]'}")

    # Available notifiers
    console.print(f"\n[bold]Available Notifiers[/bold]")
    for name in registry.list_available(ComponentType.NOTIFIER):
        console.print(f"  - {name}")


@app.command("test")
def test_notifier(
    notifier: str = typer.Argument(..., help="Notifier to test: console, email, wechat"),
    message: str = typer.Option("Test notification from QuantTool", "--message", "-m",
                                 help="Test message to send"),
):
    """
    Test a notifier by sending a test message.

    Examples:
        quanttool monitor test console
        quanttool monitor test wechat -m "Hello from QuantTool"
        quanttool monitor test email -m "Test email"
    """
    if notifier == "wechat":
        sendkey = os.environ.get("SERVERCHAN_SENDKEY", "")
        if not sendkey:
            from ...config.settings import settings
            sendkey = settings.get("notification.wechat.sendkey", "")

        if not sendkey:
            console.print("[red]Error: No WeChat SendKey configured[/red]")
            console.print("Set SERVERCHAN_SENDKEY environment variable or configure in config file")
            raise typer.Exit(1)

        n = _get_notifier("wechat", {"sendkey": sendkey})

    elif notifier == "email":
        from ...config.settings import settings
        email_config = settings.get("notification.email", {})

        smtp_server = os.environ.get("SMTP_SERVER") or email_config.get("smtp_server")
        username = os.environ.get("SMTP_USERNAME") or email_config.get("username")
        password = os.environ.get("SMTP_PASSWORD") or email_config.get("password")
        to_emails = email_config.get("to_emails", [])

        if not all([smtp_server, username, password, to_emails]):
            console.print("[red]Error: Email not fully configured[/red]")
            console.print("Required: SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD, to_emails in config")
            raise typer.Exit(1)

        n = _get_notifier("email", {
            "smtp_server": smtp_server,
            "smtp_port": int(os.environ.get("SMTP_PORT", 465)),
            "username": username,
            "password": password,
            "to_emails": to_emails,
        })

    elif notifier == "console":
        n = _get_notifier("console")

    else:
        console.print(f"[red]Unknown notifier: {notifier}[/red]")
        raise typer.Exit(1)

    if n:
        subject = f"[Test] QuantTool Notification Test - {datetime.now().strftime('%H:%M:%S')}"
        success = n.send_notification(message, subject)
        if success:
            console.print(f"[green]✓ Test notification sent via {notifier}[/green]")
        else:
            console.print(f"[red]✗ Failed to send notification via {notifier}[/red]")
            raise typer.Exit(1)