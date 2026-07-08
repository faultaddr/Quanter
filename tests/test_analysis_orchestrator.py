import unittest

from quanttool.factors.analysis_context import (
    ActionType,
    AnalysisContext,
    FinalRecommendation,
    FundamentalData,
    ScoringSystemType,
    StopLossConfig,
    StopLossType,
    UnifiedMarketState,
)
from quanttool.factors.scoring.base import ScoreResult
from tests.fixtures.algorithm_data import make_indicator_ready_ohlcv


class FakeScoringSystem:
    def calculate_context_scores(self, df, symbol="", trade_date=""):
        return {
            "classic": ScoreResult(
                final_score=66.0,
                passed_filter=True,
                strategy_name="classic",
                details={
                    "trend_score": 61.0,
                    "position_modifier": 1.0,
                    "score_grade": "良好",
                    "factors_score": {"trend_strength": 66},
                    "factors_raw": {"aux_factors": {"bias20": 0.01}},
                    "execution": {"action_guide": "测试"},
                    "warnings": ["测试警告"],
                },
            ),
            "trend": ScoreResult(
                final_score=72.0,
                passed_filter=True,
                strategy_name="trend",
                timing_coefficient=1.1,
                details={
                    "trend_total_score": 65.0,
                    "timing_type": "测试时机",
                    "ma_structure_score": 70.0,
                    "price_momentum_score": 68.0,
                    "volume_score": 60.0,
                    "relative_strength_score": 55.0,
                },
            ),
            "breakout": ScoreResult(
                final_score=58.0,
                passed_filter=True,
                strategy_name="breakout",
                stop_loss_price=9.5,
                take_profit_price=11.0,
                details={
                    "is_low_position": True,
                    "is_consolidating": True,
                    "has_breakout": False,
                    "quality_score": 60.0,
                    "growth_score": 55.0,
                    "value_score": 52.0,
                    "momentum_score": 57.0,
                    "flow_score": 59.0,
                    "risk_score": 54.0,
                    "consolidation_days": 24,
                    "price_range": 0.12,
                    "volume_ratio": 1.3,
                    "breakout_strength": 0.0,
                },
            ),
        }


class FakeRecommendationEngine:
    def generate_recommendation(self, context):
        return FinalRecommendation(
            action=ActionType.BUY,
            primary_system=ScoringSystemType.CLASSIC,
            final_score=context.classic_score.score,
            score_grade="良好",
            entry_low=context.current_price * 0.99,
            entry_high=context.current_price * 1.01,
            position_size="30%",
            reasons=["测试推荐"],
            warnings=context.classic_score.warnings,
            confidence="高",
        )


class FakeStopLossCalculator:
    def calculate(self, df, context):
        return StopLossConfig(
            stop_price=context.current_price * 0.95,
            stop_type=StopLossType.ATR,
            distance_percent=0.05,
            confidence=0.8,
        )


class AnalysisOrchestratorTests(unittest.TestCase):
    def test_build_context_uses_injected_components(self):
        from quanttool.factors.analysis_orchestrator import AnalysisOrchestrator

        df = make_indicator_ready_ohlcv(rows=260)
        orchestrator = AnalysisOrchestrator(
            scoring_system=FakeScoringSystem(),
            recommendation_engine=FakeRecommendationEngine(),
            stop_loss_calculator=FakeStopLossCalculator(),
            market_state_builder=lambda data: UnifiedMarketState(confidence=0.75),
            fundamental_provider=lambda symbol: FundamentalData(data_source="fake"),
        )

        context = orchestrator.build_context(
            df,
            "000001.SZ",
            current_price=12.34,
        )

        self.assertIsInstance(context, AnalysisContext)
        self.assertEqual(context.symbol, "000001.SZ")
        self.assertEqual(context.current_price, 12.34)
        self.assertEqual(context.classic_score.score, 66.0)
        self.assertEqual(context.trend_score.final_score, 72.0)
        self.assertEqual(context.breakout_score.consolidation_days, 24)
        self.assertEqual(context.market_state.confidence, 0.75)
        self.assertEqual(context.fundamental_data.data_source, "fake")
        self.assertEqual(context.final_recommendation.action, ActionType.BUY)

    def test_build_context_handles_empty_data(self):
        from quanttool.factors.analysis_orchestrator import AnalysisOrchestrator

        orchestrator = AnalysisOrchestrator(scoring_system=FakeScoringSystem())
        context = orchestrator.build_context(
            make_indicator_ready_ohlcv(rows=260).iloc[0:0],
            "000001.SZ",
        )

        self.assertEqual(context.symbol, "000001.SZ")
        self.assertEqual(context.current_price, 0)

    def test_default_classic_score_preserves_position_modifier(self):
        from quanttool.factors.analysis_orchestrator import AnalysisOrchestrator
        from quanttool.factors.stock_analyzer import StockAnalyzer

        df = make_indicator_ready_ohlcv(rows=260)
        expected = StockAnalyzer.__new__(StockAnalyzer)._run_classic_scoring(
            df,
            "000001.SZ",
        )
        context = AnalysisOrchestrator(
            recommendation_engine=FakeRecommendationEngine(),
            stop_loss_calculator=FakeStopLossCalculator(),
            market_state_builder=lambda data: UnifiedMarketState(confidence=0.75),
            fundamental_provider=lambda symbol: FundamentalData(data_source="fake"),
        ).build_context(df, "000001.SZ")

        self.assertEqual(
            context.classic_score.position_modifier,
            expected.position_modifier,
        )
        self.assertEqual(context.classic_score.warnings, expected.warnings)
