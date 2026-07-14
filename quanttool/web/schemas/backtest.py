"""Backtest API schemas."""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    """Request schema for running a backtest."""

    strategy_name: str = Field(default="ma_cross", description="Name of the strategy to use")
    symbols: List[str] = Field(default_factory=list, description="List of symbols to trade")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    initial_cash: float = Field(default=100000.0, description="Initial capital")
    commission_rate: float = Field(default=0.0003, description="Commission rate per trade")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")

    def get_start_date(self) -> str:
        """Return the start date, defaulting to one year ago."""
        if self.start_date:
            return self.start_date
        return (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    def get_end_date(self) -> str:
        """Return the end date, defaulting to today."""
        if self.end_date:
            return self.end_date
        return datetime.now().strftime("%Y-%m-%d")


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
