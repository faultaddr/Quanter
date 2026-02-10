"""
A-Share Market Analysis Package
Provides unified interface for stock screening, strategy analysis, and signal generation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .screeners.stock_screener import StockScreener
from .strategies.strategy_tools import StrategyManager
from .data.data_fetcher import DataFetcher


def get_data(symbol, start_date='2020-01-01', end_date=None, source='eastmoney'):
    """
    Get stock data for a given symbol
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        source (str): Data source ('eastmoney', 'tushare', 'baostock', 'yahoo')
    
    Returns:
        pandas.DataFrame: Stock data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Use DataFetcher to get data
    fetcher = DataFetcher()
    data = fetcher.fetch(symbol, start_date, end_date, source=source)
    
    return data


def run_backtest(strategy, start_date, end_date, initial_capital=10000, symbol='000001', data_source='eastmoney'):
    """
    Run backtest for a given strategy and symbol
    
    Args:
        strategy (Strategy): Strategy object with generate_signals method
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        initial_capital (float): Initial capital for backtest
        symbol (str): Stock symbol
        data_source (str): Data source to use
    
    Returns:
        dict: Backtest results
    """
    # Get data
    data = get_data(symbol, start_date, end_date, source=data_source)
    
    # Generate signals using the strategy
    signals = strategy.generate_signals(data)
    
    # Perform backtest calculation
    # Calculate positions based on signals
    positions = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    # Calculate daily returns
    daily_returns = data['close'].pct_change()
    strategy_returns = daily_returns * positions.shift(1)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
    equity_curve = initial_capital * cumulative_returns
    
    # Calculate performance metrics
    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
    annualized_return = (cumulative_returns.iloc[-1]) ** (252 / len(cumulative_returns)) - 1 if len(cumulative_returns) > 0 else 0
    
    # Calculate max drawdown
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() if not drawdowns.empty else 0

    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    excess_returns = strategy_returns.dropna()
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
    
    # Prepare results
    results = {
        'equity_curve': equity_curve,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'signals': signals,
        'positions': positions,
        'daily_returns': strategy_returns
    }
    
    return results


def analyze_performance(data, benchmark_symbol=None, start_date=None, end_date=None):
    """
    Analyze performance of trading strategy
    
    Args:
        data (DataFrame): Stock data with signals
        benchmark_symbol (str): Benchmark symbol for comparison
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        dict: Performance analysis results
    """
    # Placeholder for performance analysis
    # In a real implementation, this would calculate various performance metrics
    if data is not None and not data.empty:
        if 'close' in data.columns:
            # Calculate basic performance metrics
            total_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1 if len(data) > 0 else 0
            
            # Calculate volatility (risk measure)
            daily_returns = data['close'].pct_change().dropna()
            volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility
            
            # Calculate max drawdown
            cumulative = (1 + daily_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': total_return / volatility if volatility != 0 else 0
            }
    
    return {'total_return': 0, 'volatility': 0, 'max_drawdown': 0, 'sharpe_ratio': 0}


__all__ = ['get_data', 'run_backtest', 'analyze_performance']