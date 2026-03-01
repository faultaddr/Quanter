"""Report CLI commands."""

import typer
from typing import Optional
from datetime import datetime, date as dt_date

app = typer.Typer()


@app.command()
def daily(
    report_date: Optional[str] = typer.Argument(None, help="报告日期 (YYYY-MM-DD)，默认为今天"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出文件路径"),
    email: bool = typer.Option(False, "--email", "-e", help="同时发送邮件"),
):
    """生成每日投资报告."""
    import asyncio
    from quanttool.reports.daily_report_generator import DailyReportGenerator

    generator = DailyReportGenerator()

    # 解析日期
    target_date = dt_date.today()
    if report_date:
        target_date = datetime.strptime(report_date, "%Y-%m-%d").date()

    typer.echo(f"生成 {target_date} 的每日报告...")

    # 生成报告
    report = generator.generate_daily_report(target_date)

    # 保存到文件
    if output:
        filepath = output
    else:
        filepath = generator.generate_and_save_report(target_date)

    typer.echo(f"✓ 报告已保存: {filepath}")

    # 显示报告内容
    typer.echo("\n" + "="*60)
    typer.echo(report[:2000] + "..." if len(report) > 2000 else report)
    typer.echo("="*60)

    # 发送邮件
    if email:
        typer.echo("\n发送邮件...")

        async def send():
            success = await generator.generate_and_send_report(target_date)
            if success:
                typer.echo("✓ 邮件已发送")
            else:
                typer.echo("✗ 邮件发送失败")

        asyncio.run(send())


@app.command()
def history(
    days: int = typer.Option(30, "--days", "-d", help="查看最近 N 天的报告"),
    show_details: bool = typer.Option(False, "--details", help="显示详细信息"),
):
    """查看历史报告列表."""
    from pathlib import Path

    reports_dir = Path("./reports")

    if not reports_dir.exists():
        typer.echo("暂无报告目录")
        return

    # 查找报告文件
    report_files = sorted(reports_dir.glob("daily_report_*.md"), reverse=True)

    if not report_files:
        typer.echo("暂无历史报告")
        return

    typer.echo(f"最近 {min(days, len(report_files))} 份报告:\n")

    for i, filepath in enumerate(report_files[:days], 1):
        # 从文件名提取日期
        date_str = filepath.stem.replace("daily_report_", "")
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        file_size = filepath.stat().st_size / 1024  # KB

        typer.echo(f"{i}. {formatted_date} - {filepath.name} ({file_size:.1f} KB)")

        if show_details:
            # 读取并显示报告摘要
            content = filepath.read_text(encoding='utf-8')
            lines = content.split('\n')
            # 找到 Top 5 部分
            for j, line in enumerate(lines):
                if 'Top 5' in line:
                    typer.echo("\n   推荐股票:")
                    for k in range(j+2, min(j+7, len(lines))):
                        if lines[k].startswith('|') and '代码' not in lines[k] and '---' not in lines[k]:
                            parts = lines[k].split('|')
                            if len(parts) >= 4:
                                typer.echo(f"     - {parts[2].strip()} ({parts[3].strip()})")
                    break
            typer.echo()


@app.command()
def send(
    report_date: Optional[str] = typer.Argument(None, help="报告日期 (YYYY-MM-DD)，默认为今天"),
    test: bool = typer.Option(False, "--test", "-t", help="发送测试邮件"),
):
    """发送报告邮件."""
    import asyncio
    from quanttool.reports.daily_report_generator import DailyReportGenerator

    generator = DailyReportGenerator()

    target_date = dt_date.today()
    if report_date:
        target_date = datetime.strptime(report_date, "%Y-%m-%d").date()

    if test:
        typer.echo("发送测试邮件...")
        # 生成一个简单的测试报告
        test_report = f"""
# QuantTool 测试邮件

发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

这是一封测试邮件，用于验证邮件配置是否正确。

如果您收到此邮件，说明邮件发送功能工作正常。
        """.strip()

        async def send_test():
            from quanttool.infrastructure.stores.meta_db import MetaDB
            db = MetaDB()
            config = db.get_email_config()

            if not config:
                typer.echo("✗ 未配置邮件，请先配置")
                return

            from quanttool.infrastructure.notification.email_service import EmailService
            email_service = EmailService(
                smtp_host=config.get('smtp_host', 'smtp.gmail.com'),
                smtp_port=config.get('smtp_port', 587),
                username=config.get('username', ''),
                password=config.get('password', ''),
            )

            success = await email_service.send_daily_report(
                report_date=target_date,
                report_content=test_report,
                recipients=config.get('to_addrs', []),
            )

            if success:
                typer.echo("✓ 测试邮件已发送")
            else:
                typer.echo("✗ 发送失败")

        asyncio.run(send_test())
    else:
        typer.echo(f"发送 {target_date} 的报告邮件...")

        async def send():
            success = await generator.generate_and_send_report(target_date)
            if success:
                typer.echo("✓ 邮件已发送")
            else:
                typer.echo("✗ 发送失败")

        asyncio.run(send())


@app.command()
def config(
    smtp_host: Optional[str] = typer.Option(None, "--smtp-host", help="SMTP 服务器"),
    smtp_port: Optional[int] = typer.Option(None, "--smtp-port", help="SMTP 端口"),
    username: Optional[str] = typer.Option(None, "--username", "-u", help="邮箱账号"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="邮箱密码/授权码"),
    recipient: Optional[str] = typer.Option(None, "--recipient", "-r", help="收件人邮箱"),
    test: bool = typer.Option(False, "--test", "-t", help="测试邮件配置"),
):
    """配置邮件发送."""
    from quanttool.infrastructure.stores.meta_db import MetaDB

    db = MetaDB()

    # 获取现有配置
    existing_config = db.get_email_config() or {}

    # 更新配置
    new_config = {
        "id": existing_config.get("id"),
        "smtp_host": smtp_host or existing_config.get("smtp_host", "smtp.gmail.com"),
        "smtp_port": smtp_port or existing_config.get("smtp_port", 587),
        "username": username or existing_config.get("username", ""),
        "password": password or existing_config.get("password", ""),
        "from_addr": username or existing_config.get("from_addr", ""),
        "to_addrs": [recipient] if recipient else existing_config.get("to_addrs", []),
        "enabled": True,
    }

    # 保存配置
    config_id = db.save_email_config(new_config)
    typer.echo(f"✓ 邮件配置已保存")

    if test:
        import asyncio
        from quanttool.infrastructure.notification.email_service import EmailService

        async def test_connection():
            email_service = EmailService(
                smtp_host=new_config["smtp_host"],
                smtp_port=new_config["smtp_port"],
                username=new_config["username"],
                password=new_config["password"],
            )

            success = await email_service.test_connection()
            if success:
                typer.echo("✓ 邮件服务器连接测试通过")
            else:
                typer.echo("✗ 连接测试失败")

        asyncio.run(test_connection())


@app.command()
def pdf(
    input_file: str = typer.Argument(..., help="Markdown 文件路径"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出 PDF 文件路径"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="PDF 标题"),
    open_file: bool = typer.Option(False, "--open", help="转换后打开 PDF"),
):
    """将 Markdown 文件转换为 PDF.

    使用 pandoc + Chrome headless 进行转换，支持中文。
    """
    from pathlib import Path
    from quanttool.infrastructure.utils.md_to_pdf import MarkdownToPDFConverter

    input_path = Path(input_file)
    if not input_path.exists():
        typer.echo(f"✗ 文件不存在: {input_file}")
        raise typer.Exit(1)

    typer.echo(f"正在转换: {input_file}")

    try:
        converter = MarkdownToPDFConverter()
        output_path = converter.convert(
            input_path=str(input_path),
            output_path=output,
            title=title,
            open_after=open_file,
        )
        typer.echo(f"✓ PDF 已生成: {output_path}")
    except RuntimeError as e:
        typer.echo(f"✗ 转换失败: {e}")
        raise typer.Exit(1)
