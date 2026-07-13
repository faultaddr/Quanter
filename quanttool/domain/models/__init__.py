"""Data models for QuantTool."""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
import pandas as pd
from pydantic import BaseModel, Field
from typing import Union


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class Timeframe(str, Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_10 = "10m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class Bar(BaseModel):
    """Represents a single candlestick bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: Optional[float] = None
    timeframe: Timeframe
    symbol: str
    is_filled: bool = False  # True if this is a gap-filled bar


class Trade(BaseModel):
    """Represents a single trade."""

    id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    fee: float = 0.0
    pnl: Optional[float] = None


class Order(BaseModel):
    """Represents a trading order."""

    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    timestamp: datetime
    filled_quantity: float = 0.0
    filled_avg_price: Optional[float] = None
    status: str = "pending"  # pending, partially_filled, filled, cancelled
    parent_strategy: Optional[str] = None


class Position(BaseModel):
    """Represents a position in a symbol."""

    symbol: str
    side: PositionSide
    quantity: float
    avg_price: float
    timestamp: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    # T+1 规则支持：记录可卖日期（A股当天买入次日才可卖）
    sellable_date: Optional[datetime] = None  # 可以卖出的日期
    # 止损止盈支持
    stop_loss_price: Optional[float] = None  # 止损价格
    take_profit_price: Optional[float] = None  # 止盈价格
    trailing_stop_enabled: bool = False  # 是否启用移动止损
    trailing_stop_percent: float = 0.0  # 移动止损比例
    highest_price_since_entry: Optional[float] = None  # 入场后最高价（用于移动止损）


class Portfolio(BaseModel):
    """Represents a portfolio of positions."""

    cash: float
    positions: List[Position]
    total_value: float
    timestamp: datetime


class Signal(BaseModel):
    """Represents a trading signal."""

    symbol: str
    timestamp: datetime
    direction: OrderSide  # buy or sell
    strength: float = 1.0  # Signal strength (1.0 = normal, >1.0 = strong)
    reason: Optional[str] = None  # Reason for the signal
    predicted_return: Optional[float] = None  # Predicted return if available
    confidence: Optional[float] = None  # Confidence in signal (0.0-1.0)


class ExperimentRun(BaseModel):
    """Represents a single experiment run."""

    id: str
    type: str  # backtest, factor_mining, prediction, etc.
    parameters: Dict[str, Any]
    git_commit: Optional[str] = None
    data_version: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str  # pending, running, completed, failed
    results: Optional[Dict[str, Any]] = None
    artifacts: List[str] = []  # Paths to output files


class Metric(BaseModel):
    """Represents a performance metric."""

    name: str
    value: float
    description: Optional[str] = None


class BacktestResult(BaseModel):
    """Represents backtest results."""

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
    trades: List[Trade]
    orders: List[Order]
    metrics: List[Metric]
    equity_curve: List[Dict[str, Any]]  # Store as list of dictionaries with timestamp and value


class FactorEvaluationResult(BaseModel):
    """Represents factor evaluation results."""

    factor_name: str
    ic: float  # Information coefficient
    rank_ic: float  # Rank information coefficient
    ic_ir: float  # Information ratio
    win_rate: float
    avg_return: float
    volatility: float
    sharpe_ratio: float
    turnover: float
    max_exposure: float
    data: Any  # Can be DataFrame, string, or other data representation


from .serenity import (  # noqa: E402
    EvidenceStrength,
    ResearchTimingQuadrant,
    ResearchVerdict,
    SerenityEvidence,
    SerenityEvidenceSummary,
    SerenityFactors,
    SerenityPenalties,
    SerenityScoreDetail,
    SerenityScoreResult,
    SerenityScorecard,
)
