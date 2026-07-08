"""Main CLI for QuantTool."""

import sys
import os
from datetime import datetime
from typing import Optional

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import typer
from .commands.data_commands import app as data_app
from .commands.backtest_commands import app as backtest_app
from .commands.analysis_commands import app as analysis_app
from .commands.scheduler_commands import app as scheduler_app
from .commands.portfolio_commands import app as portfolio_app
from .commands.report_commands import app as report_app
from .commands.monitor_commands import app as monitor_app
from .commands.qlib_commands import app as qlib_app
from .commands.enhanced_commands import app as enhanced_app

app = typer.Typer()

# Add subcommands
app.add_typer(data_app, name="data", help="Data-related commands")
app.add_typer(backtest_app, name="backtest", help="Backtesting commands")
app.add_typer(analysis_app, name="analysis", help="Analysis commands (use 'analyze' for quick stock analysis)")
app.add_typer(scheduler_app, name="scheduler", help="Scheduler daemon commands")
app.add_typer(portfolio_app, name="portfolio", help="Portfolio backtest commands")
app.add_typer(report_app, name="report", help="Daily report commands")
app.add_typer(monitor_app, name="monitor", help="Realtime signal monitoring")
app.add_typer(qlib_app, name="qlib", help="Qlib ML model backtesting (23 models)")
app.add_typer(enhanced_app, name="enhanced", help="增强功能：筹码分析、K线形态、经典策略、综合选股、批量处理")


def _echo_context_summary(context) -> None:
    """Print a concise unified-context score summary."""
    typer.echo("\n=== 三系统评分摘要 ===")
    typer.echo(f"经典评分: {context.classic_score.score:.1f}分")
    if context.trend_score.passed_hard_filter:
        typer.echo(
            f"趋势评分: {context.trend_score.final_score:.1f}分 "
            f"(时机: {context.trend_score.timing_type})"
        )
    else:
        typer.echo(
            f"趋势评分: 未通过过滤 ({context.trend_score.hard_filter_reason})"
        )
    if context.breakout_score.passed_filter:
        typer.echo(f"突破评分: {context.breakout_score.final_score:.1f}分")
    else:
        typer.echo(
            f"突破评分: 未通过筛选 ({context.breakout_score.filter_reason})"
        )
    typer.echo(f"\n最终推荐: {context.final_recommendation.get_action_display()}")
    typer.echo("-" * 50)

@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze (e.g., 601777, 000001.SZ)"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file to save the analysis report")
):
    """Analyze a stock with the unified analysis context."""
    typer.echo(f"正在分析股票：{symbol}")
    typer.echo(f"分析周期：{days} 天")
    typer.echo("-" * 50)

    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        context, report = analyzer.analyze_stock_with_context(symbol, days)
    except Exception as exc:
        import click

        raise click.ClickException(str(exc)) from exc

    _echo_context_summary(context)
    typer.echo(report)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(report)
        typer.echo(f"\n分析报告已保存至：{output}")


@app.command()
def hello(name: str = typer.Argument(..., help="Name to greet")):
    """Say hello to someone."""
    typer.echo(f"Hello {name}! Welcome to QuantTool.")


@app.command()
def plugins():
    """List available plugins."""
    from ..core.registry import registry, ComponentType

    typer.echo("Available Data Providers:")
    for provider in registry.list_available(ComponentType.DATA_PROVIDER):
        typer.echo(f"  - {provider}")

    typer.echo("\nAvailable Strategies:")
    for strategy in registry.list_available(ComponentType.STRATEGY):
        typer.echo(f"  - {strategy}")

    typer.echo("\nAvailable Factors:")
    for factor in registry.list_available(ComponentType.FACTOR):
        typer.echo(f"  - {factor}")


if __name__ == "__main__":
    app()
