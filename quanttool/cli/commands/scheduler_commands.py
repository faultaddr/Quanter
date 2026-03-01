"""Scheduler CLI commands."""

import typer
from typing import Optional
from pathlib import Path

app = typer.Typer()


@app.command()
def start(
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run as daemon in background"),
    pid_file: Optional[str] = typer.Option(None, "--pid-file", help="PID file path"),
):
    """启动定时任务调度器."""
    from quanttool.daemon.scheduler_daemon import SchedulerDaemon

    daemon_obj = SchedulerDaemon()

    if daemon:
        typer.echo("启动调度器守护进程...")
        daemon_obj.start_background()
    else:
        typer.echo("启动调度器（前台运行，按 Ctrl+C 停止）...")
        try:
            daemon_obj.run()
        except KeyboardInterrupt:
            typer.echo("\n调度器已停止")


@app.command()
def stop():
    """停止定时任务调度器."""
    import os
    import signal

    pid_file = Path("/tmp/quanttool_scheduler.pid")

    if not pid_file.exists():
        typer.echo("调度器未在运行")
        return

    pid = int(pid_file.read_text().strip())

    try:
        os.kill(pid, signal.SIGTERM)
        typer.echo(f"已向调度器 (PID: {pid}) 发送停止信号")
        pid_file.unlink()
    except ProcessLookupError:
        typer.echo("调度器进程已不存在")
        pid_file.unlink()
    except Exception as e:
        typer.echo(f"停止调度器失败: {e}")


@app.command()
def status():
    """查看调度器状态."""
    import psutil

    pid_file = Path("/tmp/quanttool_scheduler.pid")

    if not pid_file.exists():
        typer.echo("调度器状态: 未运行")
        return

    pid = int(pid_file.read_text().strip())

    try:
        process = psutil.Process(pid)
        if process.is_running():
            typer.echo(f"调度器状态: 运行中")
            typer.echo(f"PID: {pid}")
            typer.echo(f"启动时间: {process.create_time()}")
            typer.echo(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        else:
            typer.echo("调度器状态: 未运行（残留 PID 文件）")
            pid_file.unlink()
    except psutil.NoSuchProcess:
        typer.echo("调度器状态: 未运行（残留 PID 文件）")
        pid_file.unlink()


@app.command()
def run_now(
    task_type: str = typer.Argument(..., help="任务类型: scan, update, report"),
):
    """立即执行一次指定任务."""
    import asyncio
    from quanttool.infrastructure.scheduler.task_scheduler import DailyTaskScheduler

    scheduler = DailyTaskScheduler()

    async def run():
        if task_type == "scan":
            typer.echo("执行每日 scan...")
            await scheduler.run_daily_scan()
        elif task_type == "update":
            typer.echo("更新组合净值...")
            await scheduler.run_portfolio_update()
        elif task_type == "report":
            typer.echo("生成每日报告...")
            await scheduler.run_report_generation()
        else:
            typer.echo(f"未知任务类型: {task_type}")
            return

    asyncio.run(run())
    typer.echo("任务执行完成")


@app.command()
def config(
    scan_time: Optional[str] = typer.Option(None, "--scan-time", help="Scan 时间 (HH:MM)"),
    update_time: Optional[str] = typer.Option(None, "--update-time", help="更新净值时间 (HH:MM)"),
    report_time: Optional[str] = typer.Option(None, "--report-time", help="报告时间 (HH:MM)"),
):
    """配置定时任务时间."""
    # TODO: 实现配置保存逻辑
    typer.echo("配置功能待实现")
    if scan_time:
        typer.echo(f"Scan 时间: {scan_time}")
    if update_time:
        typer.echo(f"更新净值时间: {update_time}")
    if report_time:
        typer.echo(f"报告时间: {report_time}")
