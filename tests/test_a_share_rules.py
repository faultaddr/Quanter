"""Golden cases for dated A-share trading rules and fees."""

from datetime import date
import unittest

from quanttool.core.errors import BacktestError


class AShareRuleTests(unittest.TestCase):
    """Lock official-date, board and lot-size boundaries."""

    def test_symbol_forms_normalize_identically(self):
        from quanttool.backtest.a_share_rules import normalize_symbol

        expected = normalize_symbol("600000")
        self.assertEqual(normalize_symbol("SH600000"), expected)
        self.assertEqual(normalize_symbol("600000.SH"), expected)

    def test_board_classification_and_lots(self):
        from quanttool.backtest.a_share_rules import resolve_trading_rule

        main = resolve_trading_rule("002415.SZ", date(2026, 8, 4))
        chinext = resolve_trading_rule("300750.SZ", date(2026, 8, 4))
        star = resolve_trading_rule("688981.SH", date(2026, 8, 4))
        bse = resolve_trading_rule("920001.BJ", date(2026, 8, 4))
        self.assertEqual(
            (
                main.board,
                main.price_limit,
                main.min_buy_quantity,
                main.buy_increment,
            ),
            ("main", 0.10, 100, 100),
        )
        self.assertEqual(
            (chinext.board, chinext.price_limit),
            ("chinext", 0.20),
        )
        self.assertEqual(
            (
                star.board,
                star.price_limit,
                star.min_buy_quantity,
                star.buy_increment,
            ),
            ("star", 0.20, 200, 1),
        )
        self.assertEqual(
            (
                bse.board,
                bse.price_limit,
                bse.min_buy_quantity,
                bse.buy_increment,
            ),
            ("bse", 0.30, 100, 1),
        )

    def test_chinext_and_main_st_limits_are_dated(self):
        from quanttool.backtest.a_share_rules import resolve_trading_rule

        self.assertEqual(
            resolve_trading_rule(
                "300750.SZ",
                date(2020, 8, 23),
            ).price_limit,
            0.10,
        )
        self.assertEqual(
            resolve_trading_rule(
                "300750.SZ",
                date(2020, 8, 24),
            ).price_limit,
            0.20,
        )
        self.assertEqual(
            resolve_trading_rule(
                "600000.SH",
                date(2026, 7, 5),
                stock_name="ST浦发",
            ).price_limit,
            0.05,
        )
        self.assertEqual(
            resolve_trading_rule(
                "600000.SH",
                date(2026, 7, 6),
                stock_name="ST浦发",
            ).price_limit,
            0.10,
        )

    def test_first_five_registration_sessions_have_no_limit(self):
        from quanttool.backtest.a_share_rules import resolve_trading_rule

        self.assertIsNone(
            resolve_trading_rule(
                "688981.SH",
                date(2026, 8, 4),
                listing_session=5,
            ).price_limit
        )
        self.assertEqual(
            resolve_trading_rule(
                "688981.SH",
                date(2026, 8, 4),
                listing_session=6,
            ).price_limit,
            0.20,
        )

    def test_buy_quantities_obey_board_rules(self):
        from quanttool.backtest.a_share_rules import (
            resolve_trading_rule,
            round_buy_quantity,
        )

        main = resolve_trading_rule("600000.SH", date(2026, 8, 4))
        star = resolve_trading_rule("688981.SH", date(2026, 8, 4))
        self.assertEqual(round_buy_quantity(299.9, main), 200)
        self.assertEqual(round_buy_quantity(199.9, star), 0)
        self.assertEqual(round_buy_quantity(245.9, star), 245)

    def test_unknown_symbol_and_pre_2017_date_fail_closed(self):
        from quanttool.backtest.a_share_rules import resolve_trading_rule

        with self.assertRaises(BacktestError):
            resolve_trading_rule("123456", date(2026, 8, 4))
        with self.assertRaises(BacktestError):
            resolve_trading_rule("600000.SH", date(2016, 12, 31))

    def test_board_rules_do_not_apply_before_board_launch(self):
        from quanttool.backtest.a_share_rules import resolve_trading_rule

        with self.assertRaises(BacktestError):
            resolve_trading_rule("688981.SH", date(2019, 7, 21))
        with self.assertRaises(BacktestError):
            resolve_trading_rule("920001.BJ", date(2021, 11, 14))

    def test_fee_schedule_changes_on_official_dates(self):
        from quanttool.backtest.fee_schedule import resolve_fee_rates

        self.assertEqual(
            resolve_fee_rates(date(2022, 4, 28)).transfer_fee_rate,
            0.00002,
        )
        self.assertEqual(
            resolve_fee_rates(date(2022, 4, 29)).transfer_fee_rate,
            0.00001,
        )
        self.assertEqual(
            resolve_fee_rates(date(2023, 8, 27)).stamp_tax_rate,
            0.001,
        )
        self.assertEqual(
            resolve_fee_rates(date(2023, 8, 28)).stamp_tax_rate,
            0.0005,
        )

    def test_stamp_tax_applies_to_sell_only(self):
        from quanttool.backtest.fee_schedule import calculate_transaction_cost

        buy = calculate_transaction_cost(
            10.0,
            1000,
            "buy",
            date(2026, 8, 4),
        )
        sell = calculate_transaction_cost(
            10.0,
            1000,
            "sell",
            date(2026, 8, 4),
        )
        self.assertEqual(buy.stamp_tax, 0.0)
        self.assertEqual(sell.stamp_tax, 5.0)
        self.assertGreater(sell.total_fee, buy.total_fee)

    def test_constraint_facade_delegates_to_dated_boundaries(self):
        from quanttool.backtest.ashare_constraints import ASShareConstraints

        constraints = ASShareConstraints()
        self.assertEqual(constraints.get_market_type("002415.SZ"), "main")
        self.assertEqual(
            constraints.calculate_limit_price(
                "600000.SH",
                10.05,
                date(2026, 8, 4),
            ),
            (11.06, 9.05),
        )
        costs = constraints.apply_transaction_costs(
            10.0,
            1000,
            "sell",
            date(2026, 8, 4),
        )
        self.assertEqual(costs.stamp_tax, 5.0)

    def test_constraint_facade_requires_explicit_trade_date(self):
        from quanttool.backtest.ashare_constraints import ASShareConstraints

        constraints = ASShareConstraints()
        with self.assertRaises(BacktestError):
            constraints.calculate_limit_price("600000.SH", 10.0)
        with self.assertRaises(BacktestError):
            constraints.apply_transaction_costs(10.0, 100, "buy")


if __name__ == "__main__":
    unittest.main()
