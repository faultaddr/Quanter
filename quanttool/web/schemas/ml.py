"""ML strategy API schemas."""

from typing import List

from pydantic import BaseModel


class MLBacktestRequest(BaseModel):
    """ML model backtest request."""

    model_path: str = ""
    symbols: List[str] = []
    start_date: str = ""
    end_date: str = ""
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003
    buy_threshold: float = 0.50
    sell_threshold: float = 0.50


class MLScanRequest(BaseModel):
    """ML model scan request."""

    model_path: str = ""
    symbols: List[str] = []
    top_n: int = 20
    min_probability: float = 0.50


class MLMonitorRequest(BaseModel):
    """ML monitor request."""

    model_path: str = ""
    symbols: List[str] = []
    interval_seconds: int = 60
