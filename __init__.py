"""
Main entry point for the quantitative trading tool
"""
import os
from .data.data_fetcher import DataFetcher
from .backtest.backtester import Backtester
from .analysis.performance_analyzer import PerformanceAnalyzer

__version__ = "0.1.0"
__author__ = "Quant Trader"

def get_data(symbol, start_date, end_date, source="eastmoney"):
    """
    Fetch historical data for a given symbol

    Args:
        symbol (str): Stock symbol (e.g., '000001')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        source (str): Data source ('eastmoney', 'tushare', 'baostock', 'yahoo')

    Returns:
        pandas.DataFrame: Historical price data
    """
    fetcher = DataFetcher()
    return fetcher.fetch(symbol, start_date, end_date, source)

def run_backtest(strategy, start_date, end_date, initial_capital=100000):
    """
    Run a backtest for a given strategy
    
    Args:
        strategy: Trading strategy object
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        initial_capital (float): Initial capital for backtest
    
    Returns:
        dict: Backtest results
    """
    backtester = Backtester()
    return backtester.run(strategy, start_date, end_date, initial_capital)

def analyze_performance(results):
    """
    Analyze the performance of a backtest
    
    Args:
        results (dict): Backtest results
    
    Returns:
        dict: Performance metrics
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze(results)