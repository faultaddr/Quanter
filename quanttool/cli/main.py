"""Main CLI for QuantTool."""

import typer
from .commands.data_commands import app as data_app
from .commands.backtest_commands import app as backtest_app
from .commands.analysis_commands import app as analysis_app


app = typer.Typer()

# Add subcommands
app.add_typer(data_app, name="data", help="Data-related commands")
app.add_typer(backtest_app, name="backtest", help="Backtesting commands")
app.add_typer(analysis_app, name="analyze", help="Analysis commands")


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
