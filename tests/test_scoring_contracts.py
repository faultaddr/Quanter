import unittest

from quanttool.factors.scoring import UnifiedScoringSystem
from quanttool.factors.scoring.base import ScoreResult
from tests.fixtures.algorithm_data import make_indicator_ready_ohlcv


class UnifiedScoringContractTests(unittest.TestCase):
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
                self.assertIsInstance(result.to_dict(), dict)

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


if __name__ == "__main__":
    unittest.main()
