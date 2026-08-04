"""Deterministic integrity tests for A-share backtest execution."""

from datetime import datetime
import unittest

import pandas as pd

from quanttool.backtest.engine import BacktestEngine
from quanttool.domain.interfaces.strategy import IStrategy


class SequenceStrategy(IStrategy):
    """Return signals keyed by the number of bars observed."""

    def __init__(self, signals):
        self.signals = signals

    def initialize(self, parameters):
        return None

    def calculate_signals(self, bars):
        return bars.copy()

    def get_signal(self, current_bar, historical_bars):
        return self.signals.get(
            len(historical_bars),
            {"direction": "hold"},
        )

    def get_name(self):
        return "sequence"

    def get_parameters(self):
        return {}

    def get_description(self):
        return "Deterministic test strategy"


def make_bars(rows=5):
    """Build deterministic daily bars with distinct open prices."""
    timestamps = pd.bdate_range("2026-07-27", periods=rows)
    closes = [10.0, 10.2, 10.4, 10.3, 10.5][:rows]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [value + 0.2 for value in closes],
            "low": [value - 0.2 for value in closes],
            "close": closes,
            "volume": [1_000_000.0] * rows,
            "amount": [value * 1_000_000 for value in closes],
        }
    )


class BacktestExecutionIntegrityTests(unittest.TestCase):
    """Prove event ordering, market constraints and net cash accounting."""

    def test_signal_fills_at_next_bar_open(self):
        bars = make_bars(5)
        strategy = SequenceStrategy(
            {1: {"direction": "buy"}, 3: {"direction": "sell"}}
        )
        engine = BacktestEngine(initial_cash=100_000)
        result = engine.run_backtest(
            strategy,
            {"600000.SH": bars},
            bars.timestamp.iloc[0].to_pydatetime(),
            bars.timestamp.iloc[-1].to_pydatetime(),
        )
        self.assertEqual(
            result.trades[0].timestamp,
            bars.timestamp.iloc[1].to_pydatetime(),
        )
        self.assertEqual(result.trades[0].price, bars.open.iloc[1])
        self.assertEqual(
            result.trades[1].timestamp,
            bars.timestamp.iloc[3].to_pydatetime(),
        )

    def test_main_board_buy_is_integer_hundred_share_lot(self):
        bars = make_bars(3)
        result = BacktestEngine(initial_cash=100_000).run_backtest(
            SequenceStrategy({1: {"direction": "buy"}}),
            {"600000.SH": bars},
            bars.timestamp.iloc[0].to_pydatetime(),
            bars.timestamp.iloc[-1].to_pydatetime(),
        )
        quantity = result.trades[0].quantity
        self.assertIsInstance(quantity, int)
        self.assertEqual(quantity % 100, 0)

    def test_trade_fee_contains_all_transaction_costs(self):
        bars = make_bars(5)
        result = BacktestEngine(initial_cash=100_000).run_backtest(
            SequenceStrategy(
                {1: {"direction": "buy"}, 3: {"direction": "sell"}}
            ),
            {"600000.SH": bars},
            bars.timestamp.iloc[0].to_pydatetime(),
            bars.timestamp.iloc[-1].to_pydatetime(),
        )
        buy, sell = result.trades
        self.assertEqual(
            buy.fee,
            buy.commission + buy.transfer_fee + buy.stamp_tax,
        )
        self.assertEqual(
            sell.fee,
            sell.commission + sell.transfer_fee + sell.stamp_tax,
        )
        self.assertEqual(buy.stamp_tax, 0.0)
        self.assertGreater(sell.stamp_tax, 0.0)

    def test_appending_future_bars_does_not_change_prior_fills(self):
        short = make_bars(4)
        long = make_bars(5)
        strategy = SequenceStrategy(
            {1: {"direction": "buy"}, 3: {"direction": "sell"}}
        )
        short_result = BacktestEngine(initial_cash=100_000).run_backtest(
            strategy,
            {"600000.SH": short},
            short.timestamp.iloc[0].to_pydatetime(),
            short.timestamp.iloc[-1].to_pydatetime(),
        )
        long_result = BacktestEngine(initial_cash=100_000).run_backtest(
            strategy,
            {"600000.SH": long},
            long.timestamp.iloc[0].to_pydatetime(),
            long.timestamp.iloc[-1].to_pydatetime(),
        )
        short_fills = [
            (trade.side, trade.quantity, trade.price, trade.timestamp)
            for trade in short_result.trades
        ]
        comparable = [
            (trade.side, trade.quantity, trade.price, trade.timestamp)
            for trade in long_result.trades
            if trade.timestamp <= short.timestamp.iloc[-1]
        ]
        self.assertEqual(short_fills, comparable)

    def test_limit_up_rejection_uses_preceding_close(self):
        bars = make_bars(3)
        bars.loc[1, ["open", "high", "low", "close"]] = [
            11.0,
            11.0,
            11.0,
            11.0,
        ]
        engine = BacktestEngine(initial_cash=100_000)
        result = engine.run_backtest(
            SequenceStrategy({1: {"direction": "buy"}}),
            {"600000.SH": bars},
            bars.timestamp.iloc[0].to_pydatetime(),
            bars.timestamp.iloc[-1].to_pydatetime(),
        )
        self.assertEqual(result.trades, [])
        rejected = [
            order for order in result.orders if order.status == "rejected"
        ]
        self.assertEqual(rejected[0].rejection_code, "limit_up")

    def test_t_plus_one_uses_next_supplied_bar(self):
        bars = make_bars(5)
        bars.loc[2, "timestamp"] = pd.Timestamp("2026-08-03")
        bars.loc[3, "timestamp"] = pd.Timestamp("2026-08-04")
        bars.loc[4, "timestamp"] = pd.Timestamp("2026-08-05")
        result = BacktestEngine(initial_cash=100_000).run_backtest(
            SequenceStrategy(
                {1: {"direction": "buy"}, 2: {"direction": "sell"}}
            ),
            {"600000.SH": bars},
            bars.timestamp.iloc[0].to_pydatetime(),
            bars.timestamp.iloc[-1].to_pydatetime(),
        )
        sell_trades = [
            trade for trade in result.trades if trade.side == "sell"
        ]
        self.assertEqual(
            sell_trades[0].timestamp,
            pd.Timestamp("2026-08-03").to_pydatetime(),
        )


if __name__ == "__main__":
    unittest.main()
