"""Backtest commands for QuantTool CLI."""

import typer
from typing import List, Optional
from datetime import datetime
from ...application.backtest_service import BacktestService
from ...application.data_service import DataService
from ...infrastructure.data_providers.historical.tushare_provider import TuShareProvider
from ...infrastructure.stores.parquet_store import ParquetStore


app = typer.Typer()


@app.command()
def run(
    strategy: str = typer.Option(..., "--strategy", "-s", help="Strategy to backtest"),
    symbols: List[str] = typer.Option(..., "--symbol", "-sym", help="Symbols to trade"),
    start_date: datetime = typer.Option(
        ..., "--start", "-sd", formats=["%Y-%m-%d"], help="Start date (YYYY-MM-DD)"
    ),
    end_date: datetime = typer.Option(
        ..., "--end", "-ed", formats=["%Y-%m-%d"], help="End date (YYYY-MM-DD)"
    ),
    timeframe: str = typer.Option(
        "10m", "--timeframe", "-tf", help="Timeframe (1m, 5m, 10m, 1d, etc.)"
    ),
    initial_cash: float = typer.Option(
        100000.0, "--cash", "-c", help="Initial cash amount"
    ),
    commission_rate: float = typer.Option(
        0.0003, "--commission", "-comm", help="Commission rate per trade"
    ),
    provider: str = typer.Option(
        "tushare", "--provider", "-p", help="Data provider to use"
    ),
):
    """Run a backtest."""
    # Initialize services
    backtest_service = BacktestService()

    # Run backtest
    result = backtest_service.run_backtest(
        strategy_name=strategy,
        strategy_params={},  # In a full implementation, we'd allow parameter specification
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        data_provider=provider,
    )

    # Print results
    typer.echo(f"Backtest completed!")
    typer.echo(f"Strategy: {strategy}")
    typer.echo(f"Symbols: {symbols}")
    typer.echo(f"Period: {start_date} to {end_date}")
    typer.echo(f"Timeframe: {timeframe}")
    typer.echo(f"Initial Capital: ${initial_cash:,.2f}")
    typer.echo(f"Final Capital: ${result.final_capital:,.2f}")
    typer.echo(f"Total Return: {result.total_return:.2%}")
    typer.echo(f"Annual Return: {result.annual_return:.2%}")
    typer.echo(f"Win Rate: {result.win_rate:.2%}")
    typer.echo(f"Total Trades: {result.total_trades}")
    typer.echo(f"Winning Trades: {result.winning_trades}")
    typer.echo(f"Losing Trades: {result.losing_trades}")
