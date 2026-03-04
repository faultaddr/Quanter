"""Backtest engine for QuantTool."""

from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
from ..domain.interfaces.strategy import IStrategy
from ..domain.models import Trade, Order, Position, Portfolio, Metric, BacktestResult
from ..core.timeutils import get_next_trading_bar_timestamp
from ..core.logging import get_logger


logger = get_logger(__name__)


class BacktestEngine:
    """Event-driven backtest engine with T+1 support for A-shares."""

    def __init__(self):
        """Initialize backtest engine."""
        self.initial_cash = 100000.0
        self.commission_rate = 0.0003
        self.min_commission = 5.0
        self.slippage_rate = 0.0001
        self.max_position_size = 0.1
        self.max_positions = 10
        self.enable_t_plus_1 = True  # A股 T+1 规则开关

        # Runtime state
        self.current_portfolio = None
        self.trades = []
        self.orders = []
        self.positions = {}
        self.equity_curve = []

    def set_initial_cash(self, cash: float):
        """Set initial cash for backtest."""
        self.initial_cash = cash

    def set_commission_rate(self, rate: float):
        """Set commission rate for trades."""
        self.commission_rate = rate

    def set_min_commission(self, min_comm: float):
        """Set minimum commission per trade."""
        self.min_commission = min_comm

    def set_slippage_rate(self, rate: float):
        """Set slippage rate for trades."""
        self.slippage_rate = rate

    def set_max_position_size(self, size: float):
        """Set maximum position size as fraction of portfolio."""
        self.max_position_size = size

    def set_max_positions(self, num: int):
        """Set maximum number of simultaneous positions."""
        self.max_positions = num

    def set_t_plus_1(self, enabled: bool):
        """Enable or disable T+1 rule for A-shares."""
        self.enable_t_plus_1 = enabled

    def _get_next_trading_day(self, current_date: datetime) -> datetime:
        """
        Get the next trading day from current date.
        Simplified implementation: add 1 day, skip weekends.
        For production, use a trading calendar.
        """
        from datetime import timedelta
        next_day = current_date + timedelta(days=1)
        # Skip weekends (Saturday=5, Sunday=6)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day

    def run_backtest(
        self,
        strategy: IStrategy,
        data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResult:
        """
        Run the backtest with the given strategy and data.

        Args:
            strategy: Trading strategy to test
            data: Historical data for backtest
            start_date: Start date of backtest
            end_date: End date of backtest

        Returns:
            Backtest result object
        """
        # Initialize portfolio
        self.current_portfolio = Portfolio(
            cash=self.initial_cash,
            positions=[],
            total_value=self.initial_cash,
            timestamp=start_date,
        )

        # Combine all timestamps across all symbols to create a unified timeline
        all_timestamps = set()
        for symbol, df in data.items():
            all_timestamps.update(df["timestamp"].tolist())

        all_timestamps = sorted(list(all_timestamps))

        # Initialize runtime state with latest prices tracking
        self.trades = []
        self.orders = []
        self.positions = {}
        self.equity_curve = []
        self.latest_market_prices = {}  # Track latest market prices for each symbol

        # Process each timestamp in chronological order
        for timestamp in all_timestamps:
            if timestamp < start_date or timestamp > end_date:
                continue

            # Process this timestamp for each symbol
            for symbol, df in data.items():
                # Get the bar at this timestamp for this symbol
                symbol_data = df[df["timestamp"] == timestamp]

                if not symbol_data.empty:
                    current_bar = symbol_data.iloc[0]  # Get the row

                    # Update latest market price for this symbol
                    self.latest_market_prices[symbol] = current_bar["close"]

                    # Get historical data up to this point for the strategy
                    hist_data = df[df["timestamp"] <= timestamp].copy()

                    # Get signal from strategy
                    signal = strategy.get_signal(current_bar, hist_data)

                    if signal and signal.get("direction"):
                        # Execute the signal at the next bar's close (next_close execution model)
                        next_timestamp = get_next_trading_bar_timestamp(timestamp)

                        # Find the next available bar for this symbol
                        # Convert next_timestamp to the same timezone-naive format as the dataframe
                        if hasattr(next_timestamp, 'tz') and next_timestamp.tz is not None:
                            # Convert timezone-aware timestamp to naive but keep the same time
                            next_timestamp_naive = next_timestamp.replace(tzinfo=None)
                        else:
                            next_timestamp_naive = next_timestamp

                        next_bar_data = df[df["timestamp"] >= next_timestamp_naive]
                        if not next_bar_data.empty:
                            execution_bar = next_bar_data.iloc[
                                0
                            ]  # First bar at or after next_timestamp
                            execution_price = execution_bar["close"]

                            # Apply slippage
                            if signal["direction"] == "buy":
                                execution_price *= 1 + self.slippage_rate
                            else:  # sell
                                execution_price *= 1 - self.slippage_rate

                            # Execute the order
                            self._execute_signal(
                                symbol,
                                signal,
                                execution_price,
                                timestamp,
                                execution_bar["timestamp"],
                            )

            # Record portfolio value at this timestamp
            portfolio_value = self._calculate_portfolio_value(timestamp)
            self.equity_curve.append(
                {"timestamp": timestamp, "portfolio_value": portfolio_value}
            )

        # Calculate final metrics
        backtest_result = self._generate_backtest_result(start_date, end_date)

        return backtest_result

    def _execute_signal(
        self,
        symbol: str,
        signal: Dict[str, Any],
        price: float,
        signal_time: datetime,
        exec_time: datetime,
    ):
        """Execute a trading signal."""
        direction = signal["direction"]

        # Calculate position size based on risk management rules
        max_position_value = self.current_portfolio.cash * self.max_position_size
        position_value = min(
            max_position_value,
            self.current_portfolio.total_value * self.max_position_size,
        )

        # Calculate quantity based on available capital
        if direction == "buy":
            # Check if we're already at max positions
            if (
                len([p for p in self.current_portfolio.positions if p.side == "long"])
                >= self.max_positions
            ):
                # Find the weakest long position to potentially close
                # For simplicity, we'll just skip if we're at max positions
                return

            quantity = position_value / price
            if quantity * price > self.current_portfolio.cash:
                quantity = self.current_portfolio.cash / price

            if quantity * price > 0:  # Only execute if we have enough cash
                # Place buy order
                order = Order(
                    id=f"order_{len(self.orders)+1}",
                    symbol=symbol,
                    side=direction,
                    order_type="market",
                    quantity=quantity,
                    price=price,
                    timestamp=exec_time,
                    parent_strategy=signal.get("strategy_name", "unknown"),
                )
                self.orders.append(order)

                # Execute the trade
                commission = max(
                    self.min_commission, quantity * price * self.commission_rate
                )
                total_cost = quantity * price + commission

                if total_cost <= self.current_portfolio.cash:
                    # Update portfolio
                    self.current_portfolio.cash -= total_cost

                    # Update or create position
                    if symbol in self.positions:
                        # Average into existing position
                        existing_pos = self.positions[symbol]
                        total_qty = existing_pos.quantity + quantity
                        avg_price = (
                            existing_pos.avg_price * existing_pos.quantity
                            + price * quantity
                        ) / total_qty

                        existing_pos.quantity = total_qty
                        existing_pos.avg_price = avg_price
                        existing_pos.timestamp = exec_time
                        # T+1: 新买入部分次日才能卖，但原有部分可以卖
                        # 简化处理：混合持仓的可卖日期取最早的
                        if existing_pos.sellable_date is None:
                            existing_pos.sellable_date = self._get_next_trading_day(exec_time)
                    else:
                        # Create new position with T+1 restriction
                        from datetime import timedelta
                        sellable_date = self._get_next_trading_day(exec_time) if self.enable_t_plus_1 else exec_time
                        pos = Position(
                            symbol=symbol,
                            side="long",
                            quantity=quantity,
                            avg_price=price,
                            timestamp=exec_time,
                            sellable_date=sellable_date,
                        )
                        self.positions[symbol] = pos
                        self.current_portfolio.positions.append(pos)

                    # Record the trade
                    trade = Trade(
                        id=f"trade_{len(self.trades)+1}",
                        symbol=symbol,
                        side=direction,
                        quantity=quantity,
                        price=price,
                        timestamp=exec_time,
                        fee=commission,
                    )
                    self.trades.append(trade)

        elif direction == "sell":
            # Check if we have a position in this symbol
            if symbol in self.positions and self.positions[symbol].side == "long":
                pos = self.positions[symbol]

                # T+1 检查：当天买入的股票不能卖出
                if self.enable_t_plus_1 and pos.sellable_date is not None:
                    if exec_time < pos.sellable_date:
                        # 跳过卖出信号，因为还在 T+1 限制期内
                        return

                # Determine how much to sell (could be partial)
                sell_quantity = min(pos.quantity, signal.get("quantity", pos.quantity))

                # Place sell order
                order = Order(
                    id=f"order_{len(self.orders)+1}",
                    symbol=symbol,
                    side=direction,
                    order_type="market",
                    quantity=sell_quantity,
                    price=price,
                    timestamp=exec_time,
                    parent_strategy=signal.get("strategy_name", "unknown"),
                )
                self.orders.append(order)

                # Execute the trade
                gross_proceeds = sell_quantity * price
                commission = max(
                    self.min_commission, sell_quantity * price * self.commission_rate
                )
                net_proceeds = gross_proceeds - commission

                # Update portfolio
                self.current_portfolio.cash += net_proceeds

                # Update position
                pos.quantity -= sell_quantity
                if pos.quantity <= 0:
                    # Close position completely
                    self.current_portfolio.positions = [
                        p
                        for p in self.current_portfolio.positions
                        if p.symbol != symbol
                    ]
                    del self.positions[symbol]
                else:
                    # Partial sell, keep position
                    pos.timestamp = exec_time

                # Calculate PnL
                cost_basis = sell_quantity * pos.avg_price
                pnl = gross_proceeds - cost_basis - commission

                # Record the trade
                trade = Trade(
                    id=f"trade_{len(self.trades)+1}",
                    symbol=symbol,
                    side=direction,
                    quantity=sell_quantity,
                    price=price,
                    timestamp=exec_time,
                    fee=commission,
                    pnl=pnl,
                )
                self.trades.append(trade)

    def _calculate_portfolio_value(self, timestamp: datetime) -> float:
        """Calculate total portfolio value at a given timestamp."""
        # Get current positions value based on latest available prices
        positions_value = 0

        for symbol, pos in self.positions.items():
            # Get the latest market price for this symbol
            if symbol in self.latest_market_prices:
                current_price = self.latest_market_prices[symbol]
                positions_value += pos.quantity * current_price
            else:
                # Fallback to average price if no current market price is available
                positions_value += pos.quantity * pos.avg_price

        return self.current_portfolio.cash + positions_value

    def _generate_backtest_result(
        self, start_date: datetime, end_date: datetime
    ) -> BacktestResult:
        """Generate backtest result object from backtest data."""
        final_capital = self._calculate_portfolio_value(end_date)
        total_return = (final_capital - self.initial_cash) / self.initial_cash

        # Calculate other metrics
        metrics, volatility, sharpe_ratio, sortino_ratio, max_drawdown, profit_factor = self.calculate_metrics(self.trades, self.initial_cash, self.equity_curve)

        # Convert equity curve to list of dicts format
        equity_curve_list = [{"timestamp": item["timestamp"], "portfolio_value": item["portfolio_value"]} for item in self.equity_curve]

        # Annual return calculation (simple approach)
        days_diff = (end_date - start_date).days
        annual_return = (
            ((final_capital / self.initial_cash) ** (365.0 / days_diff) - 1)
            if days_diff > 0
            else 0.0
        )

        # Determine winning/losing trades
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]

        # Create result object
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_cash,
            final_capital=final_capital,
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=len(winning_trades) / len(self.trades) if self.trades else 0.0,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            trades=self.trades,
            orders=self.orders,
            metrics=metrics,
            equity_curve=equity_curve_list,
        )

        return result

    def calculate_metrics(
        self, trades: List[Trade], initial_capital: float, equity_curve: List[Dict[str, float]]
    ) -> tuple:
        """
        Calculate performance metrics from trade history and equity curve.

        Args:
            trades: List of trades from the backtest
            initial_capital: Initial capital for the backtest
            equity_curve: Equity curve data

        Returns:
            Tuple of (metrics_list, volatility, sharpe_ratio, sortino_ratio, max_drawdown, profit_factor)
        """
        if not trades:
            return [], 0.0, 0.0, 0.0, 0.0, 0.0

        # Calculate basic trade statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]

        total_pnl = sum((t.pnl or 0) for t in trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        gross_profit = (
            sum(t.pnl for t in winning_trades if t.pnl) if winning_trades else 0
        )
        gross_loss = sum(t.pnl for t in losing_trades if t.pnl) if losing_trades else 0

        profit_factor = (
            abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")
        )

        # Calculate metrics from equity curve
        if len(equity_curve) > 1:
            # Extract portfolio values
            portfolio_values = [item['portfolio_value'] for item in equity_curve]

            # Calculate returns
            returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                      for i in range(1, len(portfolio_values))]

            if len(returns) > 0:
                # Calculate volatility (standard deviation of returns, annualized)
                import numpy as np
                returns_array = np.array(returns)
                volatility = np.std(returns_array) * (252 ** 0.5)  # Annualized volatility (assuming 252 trading days)

                # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
                avg_return = np.mean(returns_array) * 252  # Annualized return
                sharpe_ratio = avg_return / volatility if volatility != 0 else 0.0

                # Calculate Sortino ratio (using downside deviation)
                negative_returns = returns_array[returns_array < 0]
                if len(negative_returns) > 0:
                    downside_deviation = np.std(negative_returns) * (252 ** 0.5)
                else:
                    downside_deviation = 0.0
                sortino_ratio = avg_return / downside_deviation if downside_deviation != 0 else 0.0

                # Calculate maximum drawdown
                peak = portfolio_values[0]
                max_dd = 0.0
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    dd = (peak - value) / peak
                    if dd > max_dd:
                        max_dd = dd
                max_drawdown = max_dd
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
                max_drawdown = 0.0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown = 0.0

        # Create metrics list
        metrics = [
            Metric(
                name="total_return",
                value=(initial_capital + total_pnl) / initial_capital - 1,
                description="Total return over the backtest period",
            ),
            Metric(
                name="total_trades",
                value=total_trades,
                description="Total number of trades executed",
            ),
            Metric(
                name="win_rate",
                value=win_rate,
                description="Percentage of winning trades",
            ),
            Metric(
                name="profit_factor",
                value=profit_factor,
                description="Gross profit divided by gross loss",
            ),
            Metric(
                name="volatility",
                value=volatility,
                description="Annualized volatility of returns",
            ),
            Metric(
                name="sharpe_ratio",
                value=sharpe_ratio,
                description="Risk-adjusted return (annualized return / annualized volatility)",
            ),
            Metric(
                name="sortino_ratio",
                value=sortino_ratio,
                description="Downside risk-adjusted return (annualized return / annualized downside deviation)",
            ),
            Metric(
                name="max_drawdown",
                value=max_drawdown,
                description="Maximum peak-to-trough drawdown",
            ),
            Metric(
                name="avg_win_trade",
                value=gross_profit / len(winning_trades) if winning_trades else 0,
                description="Average profit per winning trade",
            ),
            Metric(
                name="avg_loss_trade",
                value=gross_loss / len(losing_trades) if losing_trades else 0,
                description="Average loss per losing trade",
            ),
        ]

        return metrics, volatility, sharpe_ratio, sortino_ratio, max_drawdown, profit_factor
