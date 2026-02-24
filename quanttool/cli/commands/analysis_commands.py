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


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze (e.g., 601777, 000001.SZ)"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file to save the analysis report")
):
    """Analyze a stock with technical indicators and trading strategies."""
    typer.echo(f"Analyzing stock: {symbol}")
    typer.echo(f"Analysis period: {days} days")
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
        typer.echo(f"\nAnalysis report saved to {output}")


@app.command()
def scan(
    market: str = typer.Option("all", "--market", "-m", help="Market to scan: sh, sz, or all"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of top stocks to return (default: 10)")
):
    """Scan the market for potential opportunities based on technical indicators."""
    typer.echo(f"Scanning {market} market for top {top_n} opportunities")
    typer.echo(f"Analysis period: {days} days")
    typer.echo("Note: This is a simplified version. Full scanning would require access to a list of symbols.")

    # This is a placeholder - in a real implementation, you'd get a list of all stocks
    # and run the analysis on each one, then rank them based on certain criteria
    typer.echo("This functionality requires a complete list of stock symbols and extended processing time.")
    typer.echo("For a single stock analysis, please use the 'analyze' command.")


if __name__ == "__main__":
    app()