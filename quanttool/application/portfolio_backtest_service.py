"""Portfolio backtest service for validating scan results."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

import pandas as pd
import numpy as np

from quanttool.infrastructure.stores.meta_db import MetaDB
from quanttool.infrastructure.data_providers.data_fetcher import EnhancedDataFetcher
from quanttool.infrastructure.data_providers.incremental_data_manager import IncrementalDataManager, DataType
from quanttool.core.logging import get_logger


logger = get_logger(__name__)


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_loss_ratio: float


class PortfolioBacktestService:
    """Service for creating and managing portfolio backtests."""

    def __init__(
        self,
        db_path: str = "./quanttool.db",
        data_fetcher: Optional[EnhancedDataFetcher] = None,
        use_incremental: bool = True,
    ):
        self.db = MetaDB(db_path)
        self.data_fetcher = data_fetcher or EnhancedDataFetcher()
        self.use_incremental = use_incremental

        # 增量数据管理器
        self._incremental_manager: Optional[IncrementalDataManager] = None
        if use_incremental:
            try:
                self._incremental_manager = IncrementalDataManager()
            except Exception as e:
                logger.warning(f"增量数据管理器初始化失败: {e}")
                self._incremental_manager = None

    def _get_stock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        获取股票数据（优先使用增量数据管理器）

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame 或 None
        """
        # 优先使用增量数据管理器
        if self._incremental_manager:
            try:
                df = self._incremental_manager.get_data(
                    symbol,
                    start_date,
                    end_date,
                    self.data_fetcher,
                    data_type=DataType.STOCK_BAR,
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"增量获取失败 {symbol}: {e}，回退到直接获取")

        # 回退到直接获取
        try:
            df_dict = self.data_fetcher.get_bars(
                [symbol],
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                timeframe="1d"
            )
            return df_dict.get(symbol, pd.DataFrame())
        except Exception as e:
            logger.error(f"获取数据失败 {symbol}: {e}")
            return None

    def create_portfolio_from_scan(
        self,
        scan_id: str,
        initial_capital: float = 500000,
        top_n: int = 5,
        holding_period: int = 20,
    ) -> Optional[str]:
        """
        Create a portfolio backtest from scan results.

        Args:
            scan_id: The scan record ID
            initial_capital: Initial capital amount (default 500,000)
            top_n: Number of top stocks to include (default 5)
            holding_period: Holding period in trading days (default 20)

        Returns:
            backtest_id: The ID of the created backtest, or None if failed
        """
        # Get scan record
        scan_record = self.db.get_scan_record(scan_id)
        if not scan_record:
            logger.error(f"Scan record not found: {scan_id}")
            return None

        results = scan_record.get("results", [])
        if len(results) < top_n:
            logger.warning(f"Scan only has {len(results)} results, requested top {top_n}")
            top_n = len(results)

        if top_n == 0:
            logger.error("No stocks in scan results")
            return None

        # Create backtest record
        scan_date = scan_record.get("scan_date", datetime.now().isoformat())
        backtest_data = {
            "scan_id": scan_id,
            "portfolio_name": f"Portfolio_{scan_date[:10]}",
            "initial_capital": initial_capital,
            "start_date": scan_date,
            "status": "active",
        }

        backtest_id = self.db.create_portfolio_backtest(backtest_data)
        logger.info(f"Created portfolio backtest: {backtest_id}")

        # Calculate capital per stock (equal weight)
        capital_per_stock = initial_capital / top_n

        # Create holdings for top N stocks
        for i, stock in enumerate(results[:top_n]):
            symbol = stock.get("symbol", "")
            name = stock.get("name", "")
            entry_price = stock.get("close", 0)

            if entry_price <= 0:
                logger.warning(f"Invalid price for {symbol}: {entry_price}")
                continue

            # Calculate shares (100 share lots for A-shares)
            shares = int(capital_per_stock / entry_price / 100) * 100
            weight = 1.0 / top_n

            holding_data = {
                "backtest_id": backtest_id,
                "symbol": symbol,
                "name": name,
                "entry_date": scan_date,
                "entry_price": entry_price,
                "shares": shares,
                "weight": weight,
                "status": "holding",
            }

            self.db.add_portfolio_holding(holding_data)
            logger.info(f"Added holding: {symbol} x {shares} @ {entry_price}")

        return backtest_id

    def update_portfolio_value(self, backtest_id: str, date: Optional[datetime] = None) -> bool:
        """
        Update portfolio value for a specific date.

        Args:
            backtest_id: The backtest ID
            date: Date to update (default: today)

        Returns:
            success: True if update was successful
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime("%Y-%m-%d")

        # Get backtest with holdings
        backtest = self.db.get_portfolio_backtest(backtest_id)
        if not backtest:
            logger.error(f"Backtest not found: {backtest_id}")
            return False

        if backtest.get("status") != "active":
            logger.info(f"Backtest {backtest_id} is not active, skipping update")
            return False

        holdings = backtest.get("holdings", [])
        if not holdings:
            logger.warning(f"No holdings in backtest {backtest_id}")
            return False

        # Initialize data fetcher
        if not self.data_fetcher._initialized:
            self.data_fetcher.initialize()

        # Calculate market value
        market_value = 0.0
        total_cost = 0.0

        for holding in holdings:
            if holding.get("status") != "holding":
                continue

            symbol = holding.get("symbol", "")
            shares = holding.get("shares", 0)
            entry_price = holding.get("entry_price", 0)

            # Get current price
            try:
                # Get data up to the target date
                end_date_dt = date
                start_date_dt = date - timedelta(days=30)

                df = self._get_stock_data(symbol, start_date_dt, end_date_dt)

                if df is None or df.empty:
                    logger.warning(f"No data for {symbol} on {date_str}")
                    continue

                # Get the last available price up to target date
                mask = df.index <= pd.Timestamp(date)
                if not mask.any():
                    logger.warning(f"No price data for {symbol} on or before {date_str}")
                    continue

                current_price = df[mask].iloc[-1]["close"]
                market_value += shares * current_price
                total_cost += shares * entry_price

            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
                continue

        # Calculate cash (remaining uninvested capital)
        initial_capital = backtest.get("initial_capital", 500000)
        invested_capital = sum(
            h.get("shares", 0) * h.get("entry_price", 0)
            for h in holdings
            if h.get("status") == "holding"
        )
        cash_value = initial_capital - invested_capital

        # Total value
        total_value = cash_value + market_value

        # Calculate daily return
        # Get previous day's value for return calculation
        daily_values = backtest.get("daily_values", [])
        if daily_values:
            prev_value = daily_values[-1].get("total_value", total_value)
            daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0
        else:
            daily_return = 0

        # Record daily value
        value_data = {
            "backtest_id": backtest_id,
            "date": date_str,
            "total_value": total_value,
            "cash_value": cash_value,
            "market_value": market_value,
            "daily_return": daily_return,
        }

        self.db.record_daily_value(value_data)
        logger.info(f"Updated portfolio {backtest_id} value for {date_str}: {total_value:,.2f}")

        return True

    def update_all_active_portfolios(self, date: Optional[datetime] = None) -> Dict[str, bool]:
        """
        Update all active portfolio values.

        Args:
            date: Date to update (default: today)

        Returns:
            results: Dict mapping backtest_id to success status
        """
        if date is None:
            date = datetime.now()

        active_portfolios = self.db.get_active_portfolios()
        results = {}

        for portfolio in active_portfolios:
            backtest_id = portfolio.get("id")
            success = self.update_portfolio_value(backtest_id, date)
            results[backtest_id] = success

        return results

    def close_portfolio(
        self,
        backtest_id: str,
        exit_date: Optional[datetime] = None,
    ) -> Optional[PortfolioMetrics]:
        """
        Close a portfolio and calculate final metrics.

        Args:
            backtest_id: The backtest ID
            exit_date: Exit date (default: today)

        Returns:
            metrics: Portfolio performance metrics, or None if failed
        """
        if exit_date is None:
            exit_date = datetime.now()

        exit_date_str = exit_date.strftime("%Y-%m-%d")

        # Get backtest
        backtest = self.db.get_portfolio_backtest(backtest_id)
        if not backtest:
            logger.error(f"Backtest not found: {backtest_id}")
            return None

        holdings = backtest.get("holdings", [])
        if not holdings:
            logger.warning(f"No holdings in backtest {backtest_id}")
            return None

        # Initialize data fetcher
        if not self.data_fetcher._initialized:
            self.data_fetcher.initialize()

        # Close each holding
        total_profit = 0
        total_cost = 0
        winners = 0
        losers = 0
        total_gain = 0
        total_loss = 0

        for holding in holdings:
            if holding.get("status") != "holding":
                continue

            symbol = holding.get("symbol", "")
            shares = holding.get("shares", 0)
            entry_price = holding.get("entry_price", 0)
            entry_date = holding.get("entry_date", "")

            try:
                # Get exit price
                end_date_dt = exit_date
                start_date_dt = exit_date - timedelta(days=30)

                df = self._get_stock_data(symbol, start_date_dt, end_date_dt)

                if df is None or df.empty:
                    logger.warning(f"No data for {symbol} on {exit_date_str}")
                    continue

                mask = df.index <= pd.Timestamp(exit_date)
                if not mask.any():
                    continue

                exit_price = df[mask].iloc[-1]["close"]
                realized_return = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
                profit = shares * (exit_price - entry_price)

                # Update holding
                self.db.update_holding_exit(
                    holding.get("id"),
                    {
                        "exit_date": exit_date_str,
                        "exit_price": exit_price,
                        "realized_return": realized_return,
                    },
                )

                # Track statistics
                total_profit += profit
                total_cost += shares * entry_price

                if realized_return > 0:
                    winners += 1
                    total_gain += realized_return
                elif realized_return < 0:
                    losers += 1
                    total_loss += abs(realized_return)

            except Exception as e:
                logger.error(f"Error closing holding for {symbol}: {e}")
                continue

        # Calculate metrics
        initial_capital = backtest.get("initial_capital", 500000)
        total_return = total_profit / initial_capital if initial_capital > 0 else 0

        # Get daily values for volatility and sharpe calculation
        daily_values = backtest.get("daily_values", [])
        if len(daily_values) >= 2:
            returns = [v.get("daily_return", 0) for v in daily_values if v.get("daily_return") is not None]

            if returns:
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                avg_return = np.mean(returns) * 252  # Annualized
                sharpe_ratio = avg_return / volatility if volatility > 0 else 0

                # Max drawdown
                values = [v.get("total_value", initial_capital) for v in daily_values]
                peak = values[0]
                max_drawdown = 0
                for value in values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak if peak > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0

        # Win rate and profit/loss ratio
        total_trades = winners + losers
        win_rate = winners / total_trades if total_trades > 0 else 0
        profit_loss_ratio = (total_gain / winners) / (total_loss / losers) if winners > 0 and losers > 0 else 0

        # Holding period
        start_date = datetime.fromisoformat(backtest.get("start_date", exit_date_str))
        days_held = (exit_date - start_date).days
        annualized_return = (1 + total_return) ** (365 / days_held) - 1 if days_held > 0 and total_return > -1 else 0

        metrics = PortfolioMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
        )

        # Update backtest record
        self.db.close_portfolio_backtest(
            backtest_id,
            {
                "end_date": exit_date_str,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
            },
        )

        logger.info(f"Closed portfolio {backtest_id} with return: {total_return:.2%}")

        return metrics

    def check_and_close_expired_portfolios(
        self,
        holding_period: int = 20,
        date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Check and close portfolios that have exceeded holding period.

        Args:
            holding_period: Holding period in trading days (default 20)
            date: Current date (default: today)

        Returns:
            closed_ids: List of closed backtest IDs
        """
        if date is None:
            date = datetime.now()

        active_portfolios = self.db.get_active_portfolios()
        closed_ids = []

        for portfolio in active_portfolios:
            backtest_id = portfolio.get("id")
            start_date_str = portfolio.get("start_date", "")

            try:
                start_date = datetime.fromisoformat(start_date_str)
                # Simple day count (can be improved to count only trading days)
                days_held = (date - start_date).days

                if days_held >= holding_period:
                    metrics = self.close_portfolio(backtest_id, date)
                    if metrics:
                        closed_ids.append(backtest_id)
                        logger.info(f"Auto-closed expired portfolio {backtest_id} after {days_held} days")
            except Exception as e:
                logger.error(f"Error checking portfolio {backtest_id}: {e}")

        return closed_ids

    def get_portfolio_summary(self, backtest_id: str) -> Optional[Dict]:
        """
        Get a summary of portfolio performance.

        Args:
            backtest_id: The backtest ID

        Returns:
            summary: Dict with summary information
        """
        backtest = self.db.get_portfolio_backtest(backtest_id)
        if not backtest:
            return None

        holdings = backtest.get("holdings", [])
        daily_values = backtest.get("daily_values", [])

        current_value = daily_values[-1].get("total_value", backtest.get("initial_capital")) if daily_values else backtest.get("initial_capital", 500000)
        initial_capital = backtest.get("initial_capital", 500000)

        summary = {
            "backtest_id": backtest_id,
            "portfolio_name": backtest.get("portfolio_name"),
            "status": backtest.get("status"),
            "start_date": backtest.get("start_date"),
            "end_date": backtest.get("end_date"),
            "initial_capital": initial_capital,
            "current_value": current_value,
            "total_return": backtest.get("total_return") or (current_value - initial_capital) / initial_capital,
            "sharpe_ratio": backtest.get("sharpe_ratio"),
            "max_drawdown": backtest.get("max_drawdown"),
            "holdings_count": len(holdings),
            "active_holdings": sum(1 for h in holdings if h.get("status") == "holding"),
            "daily_values_count": len(daily_values),
        }

        return summary