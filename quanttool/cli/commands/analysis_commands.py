"""Commands for stock analysis."""
import typer
import sys
import os
from datetime import datetime
from typing import Optional

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from quanttool.factors.stock_analyzer import StockAnalyzer

app = typer.Typer()


@app.command(name="single")
def analyze_single(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze (e.g., 601777, 000001.SZ)"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file to save the analysis report")
):
    """Analyze a single stock with technical indicators and trading strategies."""
    _run_analysis(symbol, days, output)


def _run_analysis(symbol: str, days: int, output: Optional[str]):
    """Internal function to run the analysis."""
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
def scan(
    market: str = typer.Option("all", "--market", "-m", help="Market to scan: sh, sz, or all"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of top stocks to return (default: 10)")
):
    """Scan the market for potential opportunities based on technical indicators."""
    typer.echo(f"正在扫描 {market} 市场，筛选前 {top_n} 个机会")
    typer.echo(f"分析周期：{days} 天")
    typer.echo("注意：这是简化版本，完整扫描需要获取股票列表。")

    # This is a placeholder - in a real implementation, you'd get a list of all stocks
    # and run the analysis on each one, then rank them based on certain criteria
    typer.echo("此功能需要完整的股票代码列表和较长的处理时间。")
    typer.echo("如需分析单个股票，请使用 'analyze' 命令。")


if __name__ == "__main__":
    app()