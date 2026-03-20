"""
Tool input/output schemas for MCP Agent.

All schemas use Pydantic for validation and serialization.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import date


# ==================== Analyze Stock ====================

class AnalyzeStockInput(BaseModel):
    """Input schema for analyze_stock tool."""
    symbol: str = Field(
        ...,
        description="Stock symbol (e.g., '600519', '000001.SZ')"
    )
    days: int = Field(
        default=360,
        ge=30,
        le=720,
        description="Number of trading days to analyze"
    )


class AnalyzeStockOutput(BaseModel):
    """Output schema for analyze_stock tool."""
    symbol: str = Field(description="Analyzed stock symbol")
    name: Optional[str] = Field(default=None, description="Stock name")
    analysis_report: str = Field(description="Full analysis report text")
    recommendation: str = Field(description="Buy/Sell/Hold recommendation")
    score: Optional[int] = Field(default=None, description="Overall score (0-100)")
    price: Optional[float] = Field(default=None, description="Latest price")
    change_pct: Optional[float] = Field(default=None, description="Price change percentage")
    indicators: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Technical indicators summary"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")


# ==================== Run Backtest ====================

class RunBacktestInput(BaseModel):
    """Input schema for run_backtest tool."""
    symbols: List[str] = Field(
        ...,
        min_items=1,
        description="List of stock symbols to backtest"
    )
    strategy: Literal["ma_cross", "rsi", "macd", "bollinger", "dual_ma", "turtle", "kdj"] = Field(
        default="ma_cross",
        description="Trading strategy to use"
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date (YYYY-MM-DD), defaults to 180 days ago"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date (YYYY-MM-DD), defaults to today"
    )
    initial_cash: float = Field(
        default=100000.0,
        gt=0,
        description="Initial capital"
    )
    commission_rate: float = Field(
        default=0.0003,
        ge=0,
        le=0.01,
        description="Commission rate per trade"
    )


class TradeRecord(BaseModel):
    """Single trade record."""
    date: str
    action: str  # buy/sell
    price: float
    shares: int
    amount: float


class RunBacktestOutput(BaseModel):
    """Output schema for run_backtest tool."""
    strategy: str = Field(description="Strategy used")
    symbols: List[str] = Field(description="Symbols backtested")
    start_date: str = Field(description="Backtest start date")
    end_date: str = Field(description="Backtest end date")
    initial_capital: float = Field(description="Initial capital")
    final_capital: float = Field(description="Final capital")
    total_return: float = Field(description="Total return percentage")
    annual_return: Optional[float] = Field(default=None, description="Annualized return")
    max_drawdown: Optional[float] = Field(default=None, description="Maximum drawdown")
    sharpe_ratio: Optional[float] = Field(default=None, description="Sharpe ratio")
    win_rate: Optional[float] = Field(default=None, description="Win rate percentage")
    total_trades: int = Field(default=0, description="Total number of trades")
    trades: Optional[List[TradeRecord]] = Field(default=None, description="Trade records")
    error: Optional[str] = Field(default=None, description="Error message if any")


# ==================== Qlib Backtest ====================

class QlibBacktestInput(BaseModel):
    """Input schema for qlib_backtest tool."""
    symbols: List[str] = Field(
        ...,
        min_items=1,
        description="List of stock symbols for ML backtest"
    )
    model: Literal[
        "lgb", "xgboost", "catboost", "mlp", "gru", "lstm",
        "gats", "tabnet", "transformer", "double_gru", "double_lstm",
        "linear", "ridge", "lasso", "elastic_net", "svr",
        "random_forest", "extra_trees", "adaboost", "gbdt",
        "tabnet2", "tabtransformer", "deepfm"
    ] = Field(
        default="lgb",
        description="ML model to use (23 options available)"
    )
    days: int = Field(
        default=180,
        ge=60,
        le=720,
        description="Number of days of training data"
    )
    epochs: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Training epochs for neural network models"
    )
    initial_cash: float = Field(
        default=100000.0,
        gt=0,
        description="Initial capital"
    )


class QlibBacktestOutput(BaseModel):
    """Output schema for qlib_backtest tool."""
    model: str = Field(description="ML model used")
    symbols: List[str] = Field(description="Symbols analyzed")
    training_days: int = Field(description="Days of training data")
    initial_capital: float = Field(description="Initial capital")
    final_capital: float = Field(description="Final capital")
    total_return: float = Field(description="Total return percentage")
    annual_return: Optional[float] = Field(default=None, description="Annualized return")
    information_ratio: Optional[float] = Field(default=None, description="Information ratio")
    max_drawdown: Optional[float] = Field(default=None, description="Maximum drawdown")
    ic: Optional[float] = Field(default=None, description="Information coefficient")
    rank_ic: Optional[float] = Field(default=None, description="Rank IC")
    selected_stocks: Optional[List[str]] = Field(default=None, description="Top selected stocks")
    error: Optional[str] = Field(default=None, description="Error message if any")


# ==================== Screen Stocks ====================

class FilterCondition(BaseModel):
    """Filter condition for stock screening."""
    indicator: str = Field(description="Technical indicator name")
    operator: Literal[">", "<", ">=", "<=", "==", "between"] = Field(description="Comparison operator")
    value: Any = Field(description="Value to compare against")


class ScreenStocksInput(BaseModel):
    """Input schema for screen_stocks tool."""
    index: Literal["hs300", "zz500", "sz50", "all"] = Field(
        default="hs300",
        description="Index to screen stocks from"
    )
    filters: Optional[List[FilterCondition]] = Field(
        default=None,
        description="Filter conditions"
    )
    min_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Minimum score threshold"
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of results"
    )


class ScreenedStock(BaseModel):
    """Screened stock result."""
    symbol: str
    name: Optional[str] = None
    score: Optional[int] = None
    price: Optional[float] = None
    change_pct: Optional[float] = None
    reason: Optional[str] = None


class ScreenStocksOutput(BaseModel):
    """Output schema for screen_stocks tool."""
    index: str = Field(description="Index screened")
    total_stocks: int = Field(description="Total stocks in index")
    filtered_count: int = Field(description="Number of stocks passing filters")
    stocks: List[ScreenedStock] = Field(description="Screened stock results")
    filters_applied: Optional[List[str]] = Field(default=None, description="Filters applied")
    error: Optional[str] = Field(default=None, description="Error message if any")


# ==================== Get Stock Score ====================

class GetStockScoreInput(BaseModel):
    """Input schema for get_stock_score tool."""
    symbol: str = Field(
        ...,
        description="Stock symbol to get score for"
    )


class ScoreBreakdown(BaseModel):
    """Score breakdown by dimension."""
    trend_score: Optional[int] = Field(default=None, description="Trend factor score")
    momentum_score: Optional[int] = Field(default=None, description="Momentum factor score")
    capital_flow_score: Optional[int] = Field(default=None, description="Capital flow score")
    position_modifier: Optional[float] = Field(default=None, description="Position modifier")


class GetStockScoreOutput(BaseModel):
    """Output schema for get_stock_score tool."""
    symbol: str = Field(description="Stock symbol")
    name: Optional[str] = Field(default=None, description="Stock name")
    total_score: int = Field(description="Total weighted score (0-100)")
    score_breakdown: Optional[ScoreBreakdown] = Field(
        default=None,
        description="Detailed score breakdown"
    )
    recommendation: str = Field(description="Buy/Sell/Hold recommendation")
    confidence: Optional[float] = Field(default=None, description="Confidence level (0-1)")
    price: Optional[float] = Field(default=None, description="Latest price")
    error: Optional[str] = Field(default=None, description="Error message if any")
