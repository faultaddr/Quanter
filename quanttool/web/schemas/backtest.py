"""Backtest API schemas."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, List, Optional


class BacktestRequest(BaseModel):
    """Request schema for running a backtest."""

    strategy_name: str = Field(..., description="Name of the strategy to use")
    symbols: List[str] = Field(..., description="List of symbols to trade")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    timeframe: str = Field(default="10m", description="Timeframe (1m, 5m, 10m, 1d)")
    initial_cash: float = Field(default=100000.0, description="Initial capital")
    commission_rate: float = Field(default=0.0003, description="Commission rate per trade")
    data_provider: str = Field(default="tushare", description="Data provider to use")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")


class MetricSchema(BaseModel):
    """Schema for performance metrics."""

    name: str
    value: float
    description: Optional[str] = None


class TradeSchema(BaseModel):
    """Schema for trade data."""

    id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    fee: float = 0.0
    pnl: Optional[float] = None


class BacktestResponse(BaseModel):
    """Response schema for backtest results."""

    run_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    metrics: List[MetricSchema]
    trades: List[TradeSchema]
