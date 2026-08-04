"""Event-ordered A-share backtest engine."""

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.errors import BacktestError
from ..core.logging import get_logger
from ..domain.interfaces.strategy import IStrategy
from ..domain.models import (
    BacktestResult,
    Metric,
    Order,
    OrderSide,
    Portfolio,
    Position,
    Trade,
)
from .a_share_rules import resolve_trading_rule, round_buy_quantity
from .ashare_constraints import ASShareConstraints, create_constraints
from .engine_support import BacktestEngineSupport
from .fee_schedule import calculate_transaction_cost


logger = get_logger(__name__)


@dataclass(frozen=True)
class PendingSignal:
    """A signal scheduled for the next supplied bar."""

    symbol: str
    signal: Dict[str, Any]
    signal_time: datetime
    execution_time: datetime


class BacktestEngine(BacktestEngineSupport):
    """Run bar-complete signals with next-bar execution and net costs."""

    def __init__(
        self,
        use_ashare_constraints: bool = True,
        enable_st_restriction: bool = True,
        enable_limit_check: bool = True,
        commission_rate: Optional[float] = None,
        initial_cash: float = 100000.0,
    ) -> None:
        if initial_cash <= 0:
            raise BacktestError("initial_cash must be positive")
        self.initial_cash = float(initial_cash)
        self.commission_rate = (
            0.0003 if commission_rate is None else commission_rate
        )
        self.min_commission = 5.0
        self.slippage_rate = 0.0001
        self.max_position_size = 0.1
        self.max_positions = 10
        self.enable_t_plus_1 = True
        self.enable_stop_loss = True
        self.enable_take_profit = True
        self.enable_trailing_stop = False
        self.trailing_stop_percent = 0.05

        self.use_ashare_constraints = use_ashare_constraints
        self.ashare_constraints: Optional[ASShareConstraints] = None
        if use_ashare_constraints:
            self.ashare_constraints = create_constraints(
                commission_rate=self.commission_rate,
                enable_st=enable_st_restriction,
                enable_limit=enable_limit_check,
            )

        self._stock_info_cache: Dict[str, Dict[str, Any]] = {}
        self.current_portfolio: Optional[Portfolio] = None
        self.trades: List[Trade] = []
        self.orders: List[Order] = []
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[Dict[str, Any]] = []
        self.latest_market_prices: Dict[str, float] = {}
        self._next_bar_time: Dict[str, Dict[pd.Timestamp, pd.Timestamp]] = {}
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.stop_loss_triggers = 0
        self.take_profit_triggers = 0
        self.limit_up_skips = 0
        self.limit_down_skips = 0
        self.stock_restriction_skips = 0
        self.suspended_skips = 0

    def set_initial_cash(self, cash: float) -> None:
        if cash <= 0:
            raise BacktestError("initial cash must be positive")
        self.initial_cash = float(cash)

    def set_commission_rate(self, rate: float) -> None:
        if rate < 0:
            raise BacktestError("commission rate must be non-negative")
        self.commission_rate = rate
        if self.ashare_constraints is not None:
            self.ashare_constraints.commission_rate = rate

    def set_min_commission(self, min_comm: float) -> None:
        if min_comm < 0:
            raise BacktestError("minimum commission must be non-negative")
        self.min_commission = min_comm
        if self.ashare_constraints is not None:
            self.ashare_constraints.min_commission = min_comm

    def set_slippage_rate(self, rate: float) -> None:
        if rate < 0:
            raise BacktestError("slippage rate must be non-negative")
        self.slippage_rate = rate

    def set_max_position_size(self, size: float) -> None:
        if not 0 < size <= 1:
            raise BacktestError("max position size must be in (0, 1]")
        self.max_position_size = size

    def set_max_positions(self, num: int) -> None:
        if num <= 0:
            raise BacktestError("max positions must be positive")
        self.max_positions = num

    def set_t_plus_1(self, enabled: bool) -> None:
        self.enable_t_plus_1 = enabled

    def update_stock_info(
        self,
        symbol: str,
        prev_close: float,
        is_suspended: bool = False,
        stock_name: Optional[str] = None,
    ) -> None:
        """Set non-price metadata; run-time previous close comes from bars."""
        self._stock_info_cache[symbol] = {
            "prev_close": prev_close,
            "is_suspended": is_suspended,
            "stock_name": stock_name,
        }

    @staticmethod
    def _to_datetime(value: Any) -> datetime:
        return pd.Timestamp(value).to_pydatetime()

    @staticmethod
    def _signal_direction(signal: Dict[str, Any]) -> Optional[str]:
        value = signal.get("direction")
        if value is None:
            return None
        if isinstance(value, OrderSide):
            return value.value
        direction = str(value).lower()
        if direction not in {"buy", "sell", "hold"}:
            raise BacktestError(f"Unsupported signal direction: {value}")
        return direction

    def _prepare_data(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        prepared: Dict[str, pd.DataFrame] = {}
        self._next_bar_time = {}
        for symbol, source in data.items():
            missing = required - set(source.columns)
            if source.empty or missing:
                raise BacktestError(
                    f"Invalid backtest bars for {symbol}: missing={sorted(missing)}"
                )
            frame = source.copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            frame = frame.sort_values("timestamp", kind="stable").reset_index(
                drop=True
            )
            if frame["timestamp"].duplicated().any():
                raise BacktestError(
                    f"Duplicate backtest timestamps for {symbol}"
                )
            prepared[symbol] = frame
            timestamps = frame["timestamp"].tolist()
            self._next_bar_time[symbol] = {
                timestamps[index]: timestamps[index + 1]
                for index in range(len(timestamps) - 1)
            }
        if not prepared:
            raise BacktestError("Backtest data is empty")
        return prepared

    def _initialize_run(
        self,
        symbols: List[str],
        start_date: datetime,
    ) -> None:
        if self.initial_cash <= 0:
            raise BacktestError("initial cash must be positive")
        self.current_portfolio = Portfolio(
            cash=self.initial_cash,
            positions=[],
            total_value=self.initial_cash,
            timestamp=start_date,
        )
        self.trades = []
        self.orders = []
        self.positions = {}
        self.equity_curve = []
        self.latest_market_prices = {}
        self._reset_counters()

        configured = self._stock_info_cache
        self._stock_info_cache = {}
        for symbol in symbols:
            metadata = configured.get(symbol, {})
            self._stock_info_cache[symbol] = {
                "prev_close": None,
                "is_suspended": metadata.get("is_suspended", False),
                "stock_name": metadata.get("stock_name"),
            }

    def run_backtest(
        self,
        strategy: IStrategy,
        data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResult:
        """Run signals after each completed bar and fill on the next open."""
        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            raise BacktestError("start_date must not be after end_date")
        frames = self._prepare_data(data)
        self._initialize_run(list(frames), start_date)

        timeline = sorted(
            {
                timestamp
                for frame in frames.values()
                for timestamp in frame["timestamp"].tolist()
                if pd.Timestamp(start_date)
                <= timestamp
                <= pd.Timestamp(end_date)
            }
        )
        pending_by_timestamp: Dict[pd.Timestamp, List[PendingSignal]] = {}

        for timestamp in timeline:
            pending_now = pending_by_timestamp.pop(timestamp, [])
            pending_by_symbol: Dict[str, List[PendingSignal]] = {}
            for pending in pending_now:
                pending_by_symbol.setdefault(pending.symbol, []).append(pending)

            for symbol, frame in frames.items():
                matching = frame.loc[frame["timestamp"] == timestamp]
                if matching.empty:
                    continue
                current_bar = matching.iloc[0]
                execution_time = self._to_datetime(timestamp)
                next_timestamp = self._next_bar_time[symbol].get(timestamp)

                for pending in pending_by_symbol.get(symbol, []):
                    self._execute_signal(
                        symbol=symbol,
                        signal=pending.signal,
                        price=float(current_bar["open"]),
                        signal_time=pending.signal_time,
                        exec_time=execution_time,
                        current_bar=current_bar,
                        next_execution_time=(
                            self._to_datetime(next_timestamp)
                            if next_timestamp is not None
                            else None
                        ),
                    )

                self.latest_market_prices[symbol] = float(current_bar["close"])
                if symbol in self.positions:
                    self._check_stop_loss_take_profit(
                        symbol,
                        current_bar,
                        execution_time,
                    )

                historical = frame.loc[
                    frame["timestamp"] <= timestamp
                ].copy()
                signal = strategy.get_signal(current_bar, historical)
                if signal:
                    direction = self._signal_direction(signal)
                    if direction not in {None, "hold"} and next_timestamp is not None:
                        pending = PendingSignal(
                            symbol=symbol,
                            signal=dict(signal),
                            signal_time=execution_time,
                            execution_time=self._to_datetime(next_timestamp),
                        )
                        pending_by_timestamp.setdefault(
                            next_timestamp,
                            [],
                        ).append(pending)

                stock_info = self._stock_info_cache[symbol]
                stock_info["prev_close"] = float(current_bar["close"])
                bar_name = current_bar.get("stock_name")
                if bar_name is not None and not pd.isna(bar_name):
                    stock_info["stock_name"] = str(bar_name)
                bar_suspended = current_bar.get("is_suspended")
                if bar_suspended is not None and not pd.isna(bar_suspended):
                    stock_info["is_suspended"] = bool(bar_suspended)

            portfolio_value = self._calculate_portfolio_value(execution_time)
            if self.current_portfolio is not None:
                self.current_portfolio.total_value = portfolio_value
                self.current_portfolio.timestamp = execution_time
            self.equity_curve.append(
                {
                    "timestamp": execution_time,
                    "portfolio_value": portfolio_value,
                }
            )

        return self._generate_backtest_result(start_date, end_date)

    def _reject_order(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        price: float,
        exec_time: datetime,
        signal: Dict[str, Any],
        code: str,
        reason: str,
    ) -> None:
        self.orders.append(
            Order(
                id=f"order_{len(self.orders) + 1}",
                symbol=symbol,
                side=direction,
                order_type="market",
                quantity=max(0.0, float(quantity)),
                price=price,
                timestamp=exec_time,
                status="rejected",
                parent_strategy=signal.get("strategy_name", "unknown"),
                rejection_code=code,
                rejection_reason=reason,
            )
        )

    def _filled_order(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        price: float,
        exec_time: datetime,
        signal: Dict[str, Any],
    ) -> None:
        self.orders.append(
            Order(
                id=f"order_{len(self.orders) + 1}",
                symbol=symbol,
                side=direction,
                order_type="market",
                quantity=quantity,
                price=price,
                timestamp=exec_time,
                filled_quantity=quantity,
                filled_avg_price=price,
                status="filled",
                parent_strategy=signal.get("strategy_name", "unknown"),
            )
        )

    def _execute_signal(
        self,
        symbol: str,
        signal: Dict[str, Any],
        price: float,
        signal_time: datetime,
        exec_time: datetime,
        current_bar: pd.Series,
        next_execution_time: Optional[datetime],
    ) -> None:
        del signal_time  # Retained in the interface for audit/event compatibility.
        direction = self._signal_direction(signal)
        if direction not in {"buy", "sell"}:
            return
        if self.current_portfolio is None:
            raise BacktestError("Backtest portfolio is not initialized")

        stock_info = self._stock_info_cache[symbol]
        stock_name = stock_info.get("stock_name")
        bar_name = current_bar.get("stock_name")
        if bar_name is not None and not pd.isna(bar_name):
            stock_name = str(bar_name)
        listing_session = current_bar.get("listing_session")
        if listing_session is not None and pd.isna(listing_session):
            listing_session = None
        rule = resolve_trading_rule(
            symbol,
            exec_time.date(),
            stock_name=stock_name,
            listing_session=(
                int(listing_session)
                if listing_session is not None
                else None
            ),
        )

        requested = signal.get("quantity", 0.0)
        try:
            requested_quantity = float(requested or 0.0)
        except (TypeError, ValueError) as exc:
            raise BacktestError("Signal quantity must be numeric") from exc

        allowed, code, reason, slippage_rate = self._constraint_result(
            symbol,
            direction,
            price,
            exec_time,
            current_bar,
        )
        if not allowed:
            self._reject_order(
                symbol,
                direction,
                requested_quantity,
                price,
                exec_time,
                signal,
                code or "constraint",
                reason or "Order rejected by A-share constraints",
            )
            return

        if direction == "buy":
            self._execute_buy(
                symbol,
                signal,
                price,
                exec_time,
                next_execution_time,
                rule,
                requested_quantity,
                slippage_rate,
            )
        else:
            self._execute_sell(
                symbol,
                signal,
                price,
                exec_time,
                rule,
                requested_quantity,
                slippage_rate,
            )

    def _execute_buy(
        self,
        symbol: str,
        signal: Dict[str, Any],
        price: float,
        exec_time: datetime,
        next_execution_time: Optional[datetime],
        rule: Any,
        requested_quantity: float,
        slippage_rate: float,
    ) -> None:
        assert self.current_portfolio is not None
        if symbol in self.positions:
            self._reject_order(
                symbol,
                "buy",
                requested_quantity,
                price,
                exec_time,
                signal,
                "already_positioned",
                "A position in this symbol already exists",
            )
            return
        if len(self.positions) >= self.max_positions:
            self._reject_order(
                symbol,
                "buy",
                requested_quantity,
                price,
                exec_time,
                signal,
                "max_positions",
                "Maximum simultaneous positions reached",
            )
            return

        portfolio_value = self._calculate_portfolio_value(exec_time)
        budget = min(
            self.current_portfolio.cash * self.max_position_size,
            portfolio_value * self.max_position_size,
        )
        desired = requested_quantity if requested_quantity > 0 else budget / price
        quantity = round_buy_quantity(desired, rule)
        if quantity <= 0:
            self._reject_order(
                symbol,
                "buy",
                desired,
                price,
                exec_time,
                signal,
                "invalid_lot",
                "Requested quantity is below the board minimum",
            )
            return

        breakdown = None
        slippage_cost = 0.0
        while quantity > 0:
            breakdown = calculate_transaction_cost(
                price,
                quantity,
                "buy",
                exec_time.date(),
                commission_rate=self.commission_rate,
                min_commission=self.min_commission,
            )
            slippage_cost = breakdown.gross_amount * slippage_rate
            if breakdown.net_amount + slippage_cost <= self.current_portfolio.cash:
                break
            quantity = round_buy_quantity(
                quantity - rule.buy_increment,
                rule,
            )
        if quantity <= 0 or breakdown is None:
            self._reject_order(
                symbol,
                "buy",
                desired,
                price,
                exec_time,
                signal,
                "insufficient_cash",
                "Cash is insufficient after transaction costs",
            )
            return

        cash_cost = breakdown.net_amount + slippage_cost
        self.current_portfolio.cash -= cash_cost
        average_cost = cash_cost / quantity
        sellable_date = (
            next_execution_time
            if self.enable_t_plus_1
            else exec_time
        )
        position = Position(
            symbol=symbol,
            side="long",
            quantity=quantity,
            avg_price=average_cost,
            timestamp=exec_time,
            sellable_date=sellable_date,
            stop_loss_price=signal.get("stop_loss"),
            take_profit_price=signal.get("take_profit"),
            trailing_stop_enabled=self.enable_trailing_stop,
            trailing_stop_percent=self.trailing_stop_percent,
            highest_price_since_entry=price,
        )
        self.positions[symbol] = position
        self.current_portfolio.positions.append(position)
        self._filled_order(symbol, "buy", quantity, price, exec_time, signal)
        self.trades.append(
            Trade(
                id=f"trade_{len(self.trades) + 1}",
                symbol=symbol,
                side="buy",
                quantity=quantity,
                price=price,
                timestamp=exec_time,
                fee=breakdown.total_fee,
                gross_amount=breakdown.gross_amount,
                commission=breakdown.commission,
                stamp_tax=breakdown.stamp_tax,
                transfer_fee=breakdown.transfer_fee,
                slippage_cost=slippage_cost,
            )
        )

    def _execute_sell(
        self,
        symbol: str,
        signal: Dict[str, Any],
        price: float,
        exec_time: datetime,
        rule: Any,
        requested_quantity: float,
        slippage_rate: float,
    ) -> None:
        assert self.current_portfolio is not None
        position = self.positions.get(symbol)
        if position is None:
            self._reject_order(
                symbol,
                "sell",
                requested_quantity,
                price,
                exec_time,
                signal,
                "no_position",
                "No long position is available to sell",
            )
            return
        if (
            self.enable_t_plus_1
            and position.sellable_date is not None
            and exec_time < position.sellable_date
        ):
            self._reject_order(
                symbol,
                "sell",
                requested_quantity or position.quantity,
                price,
                exec_time,
                signal,
                "t_plus_one",
                "Position is not sellable until the next supplied bar",
            )
            return

        held_quantity = int(round(position.quantity))
        desired = requested_quantity if requested_quantity > 0 else held_quantity
        if desired >= held_quantity:
            quantity = held_quantity
        elif rule.board == "main":
            quantity = int(math.floor(desired))
            quantity = (quantity // rule.buy_increment) * rule.buy_increment
        else:
            quantity = int(math.floor(desired))
        if quantity <= 0:
            self._reject_order(
                symbol,
                "sell",
                desired,
                price,
                exec_time,
                signal,
                "invalid_lot",
                "Partial sell quantity violates the board increment",
            )
            return

        breakdown = calculate_transaction_cost(
            price,
            quantity,
            "sell",
            exec_time.date(),
            commission_rate=self.commission_rate,
            min_commission=self.min_commission,
        )
        slippage_cost = breakdown.gross_amount * slippage_rate
        net_proceeds = breakdown.net_amount - slippage_cost
        cost_basis = quantity * position.avg_price
        pnl = net_proceeds - cost_basis
        self.current_portfolio.cash += net_proceeds
        position.quantity -= quantity
        position.realized_pnl += pnl
        position.timestamp = exec_time
        if position.quantity <= 0:
            self.current_portfolio.positions = [
                item
                for item in self.current_portfolio.positions
                if item.symbol != symbol
            ]
            del self.positions[symbol]

        self._filled_order(symbol, "sell", quantity, price, exec_time, signal)
        self.trades.append(
            Trade(
                id=f"trade_{len(self.trades) + 1}",
                symbol=symbol,
                side="sell",
                quantity=quantity,
                price=price,
                timestamp=exec_time,
                fee=breakdown.total_fee,
                pnl=pnl,
                gross_amount=breakdown.gross_amount,
                commission=breakdown.commission,
                stamp_tax=breakdown.stamp_tax,
                transfer_fee=breakdown.transfer_fee,
                slippage_cost=slippage_cost,
            )
        )

    def _calculate_portfolio_value(self, timestamp: datetime) -> float:
        del timestamp
        if self.current_portfolio is None:
            return self.initial_cash
        positions_value = sum(
            position.quantity
            * self.latest_market_prices.get(symbol, position.avg_price)
            for symbol, position in self.positions.items()
        )
        return self.current_portfolio.cash + positions_value

    def _generate_backtest_result(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResult:
        final_capital = self._calculate_portfolio_value(end_date)
        total_return = (final_capital - self.initial_cash) / self.initial_cash
        (
            metrics,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            profit_factor,
        ) = self.calculate_metrics(
            self.trades,
            self.initial_cash,
            self.equity_curve,
        )
        if self.use_ashare_constraints:
            metrics.extend(
                [
                    Metric(name="limit_up_skips", value=self.limit_up_skips),
                    Metric(name="limit_down_skips", value=self.limit_down_skips),
                    Metric(
                        name="stock_restriction_skips",
                        value=self.stock_restriction_skips,
                    ),
                    Metric(name="suspended_skips", value=self.suspended_skips),
                ]
            )
        days = (end_date - start_date).days
        annual_return = (
            (final_capital / self.initial_cash) ** (365.0 / days) - 1
            if days > 0
            else 0.0
        )
        closed = [
            trade
            for trade in self.trades
            if trade.side == OrderSide.SELL and trade.pnl is not None
        ]
        winning = [trade for trade in closed if trade.pnl > 0]
        losing = [trade for trade in closed if trade.pnl < 0]
        return BacktestResult(
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
            win_rate=len(winning) / len(closed) if closed else 0.0,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            trades=self.trades,
            orders=self.orders,
            metrics=metrics,
            equity_curve=self.equity_curve,
        )
