"""
MCP Agent tools for QuantTool.

This module provides tool implementations that wrap QuantTool's
core functionality for MCP integration.
"""

from .analysis import analyze_stock, get_stock_score
from .backtest import run_backtest
from .qlib import run_qlib_backtest
from .screening import screen_stocks

__all__ = [
    "analyze_stock",
    "get_stock_score",
    "run_backtest",
    "run_qlib_backtest",
    "screen_stocks",
]
