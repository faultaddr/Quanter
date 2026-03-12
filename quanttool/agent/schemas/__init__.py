"""
Schema definitions for MCP Agent tools.
"""

from .tools import (
    AnalyzeStockInput,
    AnalyzeStockOutput,
    RunBacktestInput,
    RunBacktestOutput,
    QlibBacktestInput,
    QlibBacktestOutput,
    ScreenStocksInput,
    ScreenStocksOutput,
    GetStockScoreInput,
    GetStockScoreOutput,
)

__all__ = [
    "AnalyzeStockInput",
    "AnalyzeStockOutput",
    "RunBacktestInput",
    "RunBacktestOutput",
    "QlibBacktestInput",
    "QlibBacktestOutput",
    "ScreenStocksInput",
    "ScreenStocksOutput",
    "GetStockScoreInput",
    "GetStockScoreOutput",
]
