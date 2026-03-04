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

from quanttool.factors.stock_analyzer import StockAnalyzer

app = typer.Typer()

# Add subcommands
app.add_typer(data_app, name="data", help="Data-related commands")
app.add_typer(backtest_app, name="backtest", help="Backtesting commands")
app.add_typer(analysis_app, name="analysis", help="Analysis commands (use 'analyze' for quick stock analysis)")
app.add_typer(scheduler_app, name="scheduler", help="Scheduler daemon commands")
app.add_typer(portfolio_app, name="portfolio", help="Portfolio backtest commands")
app.add_typer(report_app, name="report", help="Daily report commands")
app.add_typer(monitor_app, name="monitor", help="Realtime signal monitoring")


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze (e.g., 601777, 000001.SZ)"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file to save the analysis report")
):
    """Analyze a stock with technical indicators and trading strategies."""
    typer.echo(f"正在分析股票：{symbol}")
    typer.echo(f"分析周期：{days} 天")
    typer.echo("-" * 50)

    # Create analyzer instance
    analyzer = StockAnalyzer()

    # Run analysis
    report = analyzer.analyze_stock(symbol, days)

    # Print report
    typer.echo(report)

    # Save to file if requested
    if output:
        with open(output, 'w', encoding='utf-8') as f:
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
