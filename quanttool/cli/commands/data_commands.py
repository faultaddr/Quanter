"""Data commands for QuantTool CLI."""

import typer
from typing import List, Optional
from datetime import datetime
from ...application.data_service import DataService
from ...infrastructure.data_providers.tushare_provider import TuShareProvider
from ...infrastructure.data_providers.ashare_provider import AShareProvider
from ...infrastructure.data_providers.csv_provider import CSVProvider
from ...infrastructure.stores.parquet_store import ParquetStore


app = typer.Typer()


@app.command()
def pull(
    provider: str = typer.Option(
        "tushare", "--provider", "-p", help="Data provider to use"
    ),
    symbols: List[str] = typer.Option(..., "--symbol", "-s", help="Symbols to pull"),
    start_date: datetime = typer.Option(
        ..., "--start", "-sd", formats=["%Y-%m-%d"], help="Start date (YYYY-MM-DD)"
    ),
    end_date: datetime = typer.Option(
        ..., "--end", "-ed", formats=["%Y-%m-%d"], help="End date (YYYY-MM-DD)"
    ),
    timeframe: str = typer.Option(
        "1d", "--timeframe", "-tf", help="Timeframe (1m, 5m, 10m, 1d, etc.)"
    ),
    output_dir: str = typer.Option("./data", "--output", "-o", help="Output directory"),
):
    """Pull data from various providers."""
    # Initialize service and store
    data_service = DataService()

    # Set up the appropriate provider based on the option
    if provider == "tushare":
        provider_instance = TuShareProvider()
    elif provider == "ashare":
        provider_instance = AShareProvider()
    elif provider == "csv_mock":
        provider_instance = CSVProvider()
    else:
        typer.echo(f"Unknown provider: {provider}")
        raise typer.Exit(code=1)

    # Initialize provider
    provider_instance.initialize()

    # Set up store
    store = ParquetStore(output_dir)
    data_service.set_store(store)

    # Pull data
    result = data_service.pull_data(
        provider_name=provider,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        save_to_store=True,
    )

    typer.echo(f"Pulled data for {len(result['data'])} symbols")
    typer.echo(f"Period: {start_date} to {end_date}")
    typer.echo(f"Timeframe: {timeframe}")
    typer.echo(f"Provider: {provider}")


@app.command()
def search(
    provider: str = typer.Option(
        "tushare", "--provider", "-p", help="Data provider to use"
    ),
    query: str = typer.Argument(..., help="Search query"),
):
    """Search for symbols."""
    # Initialize provider
    if provider == "tushare":
        provider_instance = TuShareProvider()
    elif provider == "ashare":
        provider_instance = AShareProvider()
    elif provider == "csv_mock":
        provider_instance = CSVProvider()
    else:
        typer.echo(f"Unknown provider: {provider}")
        raise typer.Exit(code=1)

    provider_instance.initialize()

    # Search symbols
    results = provider_instance.search_symbols(query)

    if results:
        typer.echo(f"Found {len(results)} symbols:")
        for result in results:
            typer.echo(f"  {result['symbol']}: {result.get('name', 'N/A')}")
    else:
        typer.echo("No symbols found.")
