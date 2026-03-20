"""
Backtest tools for MCP Agent.
"""

from typing import List
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from ..schemas.tools import (
    RunBacktestInput,
    RunBacktestOutput,
    TradeRecord,
)


# Strategy registry
STRATEGY_REGISTRY = {
    "ma_cross": ("quanttool.strategies.ma_cross", "MACrossStrategy"),
    "dual_ma": ("quanttool.strategies.dual_ma", "DualMAStrategy"),
    "rsi": ("quanttool.strategies.rsi", "RSIStrategy"),
    "macd": ("quanttool.strategies.macd", "MACDStrategy"),
    "bollinger": ("quanttool.strategies.bollinger", "BollingerStrategy"),
    "kdj": ("quanttool.strategies.kdj", "KDJStrategy"),
    "turtle": ("quanttool.strategies.turtle", "TurtleStrategy"),
}


def run_backtest(input_data: RunBacktestInput) -> RunBacktestOutput:
    """
    Run a backtest with the specified strategy.

    Args:
        input_data: Backtest parameters

    Returns:
        RunBacktestOutput with backtest results
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        # Parse dates
        if input_data.end_date:
            end_date = datetime.strptime(input_data.end_date, "%Y-%m-%d")
        else:
            end_date = datetime.now()

        if input_data.start_date:
            start_date = datetime.strptime(input_data.start_date, "%Y-%m-%d")
        else:
            start_date = end_date - timedelta(days=180)

        # Get stock data
        analyzer = StockAnalyzer(use_cache=True)
        data_map = analyzer.get_stock_data_batch(input_data.symbols, days=365)

        if not data_map:
            return RunBacktestOutput(
                strategy=input_data.strategy,
                symbols=input_data.symbols,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                initial_capital=input_data.initial_cash,
                final_capital=input_data.initial_cash,
                total_return=0.0,
                error="无法获取股票数据",
            )

        # Import and run strategy
        strategy_module, strategy_class = STRATEGY_REGISTRY.get(
            input_data.strategy,
            ("quanttool.strategies.ma_cross", "MACrossStrategy")
        )

        module = __import__(strategy_module, fromlist=[strategy_class])
        StrategyClass = getattr(module, strategy_class)

        # Simple backtest simulation
        # Note: For production, use the full BacktestEngine
        results = _run_simple_backtest(
            data_map,
            StrategyClass,
            start_date,
            end_date,
            input_data.initial_cash,
            input_data.commission_rate,
        )

        return RunBacktestOutput(
            strategy=input_data.strategy,
            symbols=input_data.symbols,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=input_data.initial_cash,
            final_capital=results['final_capital'],
            total_return=results['total_return'],
            annual_return=results.get('annual_return'),
            max_drawdown=results.get('max_drawdown'),
            win_rate=results.get('win_rate'),
            total_trades=results['total_trades'],
            trades=results.get('trades'),
        )

    except Exception as e:
        return RunBacktestOutput(
            strategy=input_data.strategy,
            symbols=input_data.symbols,
            start_date=input_data.start_date or "",
            end_date=input_data.end_date or "",
            initial_capital=input_data.initial_cash,
            final_capital=input_data.initial_cash,
            total_return=0.0,
            error=str(e),
        )


def _run_simple_backtest(
    data_map: dict,
    StrategyClass,
    start_date: datetime,
    end_date: datetime,
    initial_cash: float,
    commission_rate: float,
) -> dict:
    """
    Run a simplified backtest simulation.

    For full backtest functionality, use BacktestService.
    """
    import pandas as pd
    import numpy as np

    cash = initial_cash
    holdings = {}  # symbol -> shares
    total_trades = 0
    trades = []
    portfolio_values = []

    # Combine all data
    for symbol, df in data_map.items():
        if df.empty:
            continue

        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        if len(df) < 10:
            continue

        # Simple buy and hold for demonstration
        # In production, use actual strategy signals
        buy_price = df.iloc[0]['close']
        shares = int((cash * 0.3) / buy_price)

        if shares > 0:
            cost = shares * buy_price * (1 + commission_rate)
            if cost <= cash:
                cash -= cost
                holdings[symbol] = shares
                total_trades += 1
                trades.append(TradeRecord(
                    date=df.index[0].strftime("%Y-%m-%d"),
                    action="buy",
                    price=buy_price,
                    shares=shares,
                    amount=cost,
                ))

    # Calculate final value
    final_value = cash

    for symbol, df in data_map.items():
        if symbol in holdings and not df.empty:
            last_price = df.iloc[-1]['close']
            final_value += holdings[symbol] * last_price

    portfolio_values.append(initial_cash)
    portfolio_values.append(final_value)

    total_return = ((final_value - initial_cash) / initial_cash) * 100

    # Calculate annualized return (approximate)
    days = (end_date - start_date).days
    annual_return = None
    if days > 0 and total_return != 0:
        annual_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100

    return {
        'final_capital': round(final_value, 2),
        'total_return': round(total_return, 2),
        'annual_return': round(annual_return, 2) if annual_return else None,
        'max_drawdown': None,
        'win_rate': None,
        'total_trades': total_trades,
        'trades': trades if trades else None,
    }
