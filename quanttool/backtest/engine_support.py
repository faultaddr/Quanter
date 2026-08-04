"""Focused constraint, stop and metric helpers for the backtest engine."""

from datetime import datetime
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..core.errors import BacktestError
from ..domain.models import Metric, Trade
from .ashare_constraints import LimitStatus, StockStatus


class BacktestEngineSupport:
    """Mixin for cohesive engine concerns that do not own the event loop."""

    def _constraint_result(
        self,
        symbol: str,
        direction: str,
        price: float,
        exec_time: datetime,
        current_bar: pd.Series,
    ) -> Tuple[bool, Optional[str], Optional[str], float]:
        if not self.use_ashare_constraints or self.ashare_constraints is None:
            return True, None, None, self.slippage_rate

        stock_info = self._stock_info_cache[symbol]
        prev_close = stock_info.get("prev_close")
        if prev_close is None:
            raise BacktestError(
                f"No preceding close available for {symbol} at {exec_time}"
            )
        is_suspended = stock_info.get("is_suspended", False)
        bar_suspended = current_bar.get("is_suspended")
        if bar_suspended is not None and not pd.isna(bar_suspended):
            is_suspended = bool(bar_suspended)
        stock_name = stock_info.get("stock_name")
        bar_name = current_bar.get("stock_name")
        if bar_name is not None and not pd.isna(bar_name):
            stock_name = str(bar_name)
        listing_session = current_bar.get("listing_session")
        if listing_session is not None and pd.isna(listing_session):
            listing_session = None

        method = (
            self.ashare_constraints.can_buy
            if direction == "buy"
            else self.ashare_constraints.can_sell
        )
        constraint = method(
            symbol,
            price,
            float(prev_close),
            bool(is_suspended),
            stock_name,
            trade_date=exec_time,
            listing_session=(
                int(listing_session)
                if listing_session is not None
                else None
            ),
        )
        if constraint.can_trade:
            return True, None, None, constraint.slippage_rate

        if constraint.limit_status is LimitStatus.LIMIT_UP:
            code = "limit_up"
            self.limit_up_skips += 1
        elif constraint.limit_status is LimitStatus.LIMIT_DOWN:
            code = "limit_down"
            self.limit_down_skips += 1
        elif constraint.limit_status is LimitStatus.SUSPENDED:
            code = "suspended"
            self.suspended_skips += 1
        elif constraint.stock_status is StockStatus.ST:
            code = "st_restricted"
            self.stock_restriction_skips += 1
        else:
            code = "constraint"
        return False, code, constraint.reason, constraint.slippage_rate

    def _check_stop_loss_take_profit(
        self,
        symbol: str,
        current_bar: pd.Series,
        timestamp: datetime,
    ) -> None:
        position = self.positions.get(symbol)
        if position is None:
            return
        if (
            self.enable_t_plus_1
            and position.sellable_date is not None
            and timestamp < position.sellable_date
        ):
            return

        current_high = float(current_bar["high"])
        current_low = float(current_bar["low"])
        if position.trailing_stop_enabled:
            prior_high = position.highest_price_since_entry or current_high
            if current_high > prior_high:
                position.highest_price_since_entry = current_high
                trailing = current_high * (1 - position.trailing_stop_percent)
                if (
                    position.stop_loss_price is None
                    or trailing > position.stop_loss_price
                ):
                    position.stop_loss_price = trailing

        trigger_price: Optional[float] = None
        reason: Optional[str] = None
        if (
            self.enable_stop_loss
            and position.stop_loss_price is not None
            and current_low <= position.stop_loss_price
        ):
            trigger_price = position.stop_loss_price
            reason = "stop_loss"
        elif (
            self.enable_take_profit
            and position.take_profit_price is not None
            and current_high >= position.take_profit_price
        ):
            trigger_price = position.take_profit_price
            reason = "take_profit"
        if trigger_price is None or reason is None:
            return

        trade_count = len(self.trades)
        self._execute_signal(
            symbol=symbol,
            signal={
                "direction": "sell",
                "quantity": position.quantity,
                "strategy_name": f"{reason}_trigger",
            },
            price=float(trigger_price),
            signal_time=timestamp,
            exec_time=timestamp,
            current_bar=current_bar,
            next_execution_time=None,
        )
        if len(self.trades) > trade_count:
            if reason == "stop_loss":
                self.stop_loss_triggers += 1
            else:
                self.take_profit_triggers += 1

    def calculate_metrics(
        self,
        trades: List[Trade],
        initial_capital: float,
        equity_curve: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple:
        """Calculate trade and equity-curve metrics."""
        if not trades:
            return [], 0.0, 0.0, 0.0, 0.0, 0.0
        curve = self.equity_curve if equity_curve is None else equity_curve
        total_trades = len(trades)
        winning = [trade for trade in trades if trade.pnl and trade.pnl > 0]
        losing = [trade for trade in trades if trade.pnl and trade.pnl < 0]
        total_pnl = sum((trade.pnl or 0.0) for trade in trades)
        win_rate = len(winning) / total_trades if total_trades else 0.0
        gross_profit = sum(trade.pnl or 0.0 for trade in winning)
        gross_loss = sum(trade.pnl or 0.0 for trade in losing)
        profit_factor = (
            abs(gross_profit / gross_loss)
            if gross_loss
            else float("inf")
        )

        volatility = sharpe = sortino = max_drawdown = 0.0
        values = [float(item["portfolio_value"]) for item in curve]
        if len(values) > 1:
            import numpy as np

            returns = np.array(
                [
                    (values[index] - values[index - 1]) / values[index - 1]
                    for index in range(1, len(values))
                    if values[index - 1] != 0
                ]
            )
            if returns.size:
                volatility = float(np.std(returns) * math.sqrt(252))
                annualized = float(np.mean(returns) * 252)
                sharpe = annualized / volatility if volatility else 0.0
                downside = returns[returns < 0]
                downside_deviation = (
                    float(np.std(downside) * math.sqrt(252))
                    if downside.size
                    else 0.0
                )
                sortino = (
                    annualized / downside_deviation
                    if downside_deviation
                    else 0.0
                )
                peak = values[0]
                for value in values:
                    peak = max(peak, value)
                    if peak:
                        max_drawdown = max(
                            max_drawdown,
                            (peak - value) / peak,
                        )

        metrics = [
            Metric(
                name="total_return",
                value=(initial_capital + total_pnl) / initial_capital - 1,
            ),
            Metric(name="total_trades", value=total_trades),
            Metric(name="win_rate", value=win_rate),
            Metric(name="profit_factor", value=profit_factor),
            Metric(name="volatility", value=volatility),
            Metric(name="sharpe_ratio", value=sharpe),
            Metric(name="sortino_ratio", value=sortino),
            Metric(name="max_drawdown", value=max_drawdown),
            Metric(
                name="avg_win_trade",
                value=gross_profit / len(winning) if winning else 0.0,
            ),
            Metric(
                name="avg_loss_trade",
                value=gross_loss / len(losing) if losing else 0.0,
            ),
            Metric(name="stop_loss_triggers", value=self.stop_loss_triggers),
            Metric(name="take_profit_triggers", value=self.take_profit_triggers),
        ]
        return (
            metrics,
            volatility,
            sharpe,
            sortino,
            max_drawdown,
            profit_factor,
        )
