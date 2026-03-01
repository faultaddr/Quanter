"""Backtest service for QuantTool."""

from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
from ..domain.interfaces.strategy import IStrategy
from ..domain.interfaces.data_provider import IDataProvider
from ..domain.models import BacktestResult, Trade, Order, Metric
from ..backtest.engine import BacktestEngine
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger


logger = get_logger(__name__)


class BacktestService:
    """Service class for running backtests."""

    def __init__(self):
        """Initialize backtest service."""
        self.engine = BacktestEngine()

    def run_backtest(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "10m",
        initial_cash: float = 100000.0,
        commission_rate: float = 0.0003,
        data_provider: str = "tushare",
    ) -> BacktestResult:
        """
        Run a backtest with the specified parameters.

        Args:
            strategy_name: Name of the strategy to use
            strategy_params: Parameters for the strategy
            symbols: List of symbols to trade
            start_date: Start date for the backtest
            end_date: End date for the backtest
            timeframe: Timeframe for the backtest
            initial_cash: Initial capital for the backtest
            commission_rate: Commission rate per trade
            data_provider: Name of the data provider to use

        Returns:
            Backtest result object
        """
        logger.info(
            f"Starting backtest for strategy: {strategy_name}, symbols: {symbols}, "
            f"timeframe: {timeframe}, period: {start_date} to {end_date}"
        )

        # Get strategy from registry
        strategy_class = registry.get(ComponentType.STRATEGY, strategy_name)
        strategy = strategy_class()
        strategy.initialize(strategy_params)

        # Get data provider
        provider_class = registry.get(ComponentType.DATA_PROVIDER, data_provider)
        data_provider_instance = provider_class()
        if hasattr(data_provider_instance, "initialize"):
            data_provider_instance.initialize()

        # Get data
        data = data_provider_instance.get_bars(symbols, start_date, end_date, timeframe)

        # Configure engine with parameters
        self.engine.set_initial_cash(initial_cash)
        self.engine.set_commission_rate(commission_rate)

        # Run the backtest
        result = self.engine.run_backtest(strategy, data, start_date, end_date)

        logger.info(
            f"Backtest completed. Final value: {result.final_capital}, Total return: {result.total_return:.2%}"
        )

        return result

    def calculate_metrics(
        self, trades: List[Trade], initial_capital: float
    ) -> List[Metric]:
        """
        Calculate performance metrics from trade history.

        Args:
            trades: List of trades from the backtest
            initial_capital: Initial capital for the backtest

        Returns:
            List of calculated metrics
        """
        return self.engine.calculate_metrics(trades, initial_capital)
