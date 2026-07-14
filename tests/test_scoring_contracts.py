import unittest

from quanttool.factors.scoring import UnifiedScoringSystem
from quanttool.factors.scoring.base import ScoreResult
from quanttool.factors.scoring.strategies.trend import TrendScoringStrategy
from quanttool.factors.trend_scoring_system import TrendScoringSystem
from tests.fixtures.algorithm_data import make_indicator_ready_ohlcv, make_trending_ohlcv


class UnifiedScoringContractTests(unittest.TestCase):
    def test_default_scorer_loads_all_context_strategies(self):
        scorer = UnifiedScoringSystem()

        self.assertEqual(
            {strategy.name for strategy in scorer.strategies},
            {"multi_dimension", "trend", "breakout"},
        )

    def test_default_scorer_returns_context_score_keys(self):
        scorer = UnifiedScoringSystem()
        scores = scorer.calculate_context_scores(
            make_indicator_ready_ohlcv(rows=260),
            symbol="000001.SZ",
            trade_date="2024-12-31",
        )

        self.assertEqual(set(scores), {"classic", "trend", "breakout"})
        for key, result in scores.items():
            with self.subTest(key=key):
                self.assertIsInstance(result, ScoreResult)
                self.assertEqual(result.strategy_name, key)
                self.assertGreaterEqual(result.final_score, 0)
                self.assertLessEqual(result.final_score, 100)
                self.assertNotIn("评分策略缺失", result.filter_reason)
                self.assertIsInstance(result.to_dict(), dict)

        self.assertIn("is_low_position", scores["breakout"].details)
        self.assertIn("low_position", scores["breakout"].details)
        self.assertIn("consolidation", scores["breakout"].details)
        self.assertIn("breakout", scores["breakout"].details)

    def test_multi_dimension_strategy_uses_legacy_calculate_all_scores(self):
        scorer = UnifiedScoringSystem()
        result = scorer.calculate_context_scores(
            make_indicator_ready_ohlcv(rows=260),
            symbol="000001.SZ",
            trade_date="2024-12-31",
        )["classic"]

        self.assertIsInstance(result.details, dict)
        self.assertIn("legacy_result", result.details)
        self.assertIn("factors_raw", result.details)
        self.assertIn("execution", result.details)


class TrendScoringIndicatorTests(unittest.TestCase):
    def test_rsi_treats_zero_average_loss_as_extreme_strength(self):
        df = make_trending_ohlcv(rows=260)
        df["volume"] = df["volume"] * 10
        df["amount"] = df["close"] * df["volume"]

        result = TrendScoringSystem().calculate_score(df)

        self.assertTrue(result.passed_hard_filter, result.hard_filter_reason)
        self.assertGreaterEqual(result.details["timing"]["rsi"], 99)

    def test_strategy_rsi_treats_zero_average_loss_as_extreme_strength(self):
        df = make_trending_ohlcv(rows=260)
        df["volume"] = df["volume"] * 10
        df["amount"] = df["close"] * df["volume"]

        result = TrendScoringStrategy().calculate_score(df)

        self.assertTrue(result.passed_filter, result.filter_reason)
        self.assertGreaterEqual(result.details["timing"]["rsi"], 99)


if __name__ == "__main__":
    unittest.main()
