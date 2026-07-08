"""Analysis context orchestration for single-stock analysis."""

from datetime import datetime
from typing import Callable, Optional

import pandas as pd

from .analysis_context import (
    AnalysisContext,
    BreakoutScore,
    ClassicScore,
    FundamentalData,
    MarketState,
    PositionAssessment,
    ScoringSystemType,
    TrendScore,
    UnifiedMarketState,
)
from .recommendation_engine import RecommendationEngine
from .scoring import UnifiedScoringSystem
from .scoring.base import ScoreResult
from .screening import StockScreener
from .talib_patterns import recognize_talib_patterns
from .trading_strategies import TradingStrategies
from .unified_stop_loss import UnifiedStopLossCalculator


class AnalysisOrchestrator:
    """Build AnalysisContext from prepared market data."""

    def __init__(
        self,
        scoring_system: Optional[UnifiedScoringSystem] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
        stop_loss_calculator: Optional[UnifiedStopLossCalculator] = None,
        market_state_builder: Optional[Callable[[pd.DataFrame], UnifiedMarketState]] = None,
        fundamental_provider: Optional[Callable[[str], FundamentalData]] = None,
    ) -> None:
        self.scoring_system = scoring_system or UnifiedScoringSystem()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
        self.stop_loss_calculator = stop_loss_calculator or UnifiedStopLossCalculator()
        self.market_state_builder = market_state_builder or self._build_default_market_state
        self.fundamental_provider = fundamental_provider

    def build_context(
        self,
        df: pd.DataFrame,
        symbol: str,
        primary_system: ScoringSystemType = ScoringSystemType.AUTO,
        current_price: Optional[float] = None,
    ) -> AnalysisContext:
        if df.empty:
            return AnalysisContext(
                symbol=symbol,
                current_price=0,
                analysis_date=datetime.now(),
            )

        latest = df.iloc[-1]
        close = float(current_price if current_price is not None else latest.get("close", 0))

        context = AnalysisContext(
            symbol=symbol,
            current_price=close,
            analysis_date=datetime.now(),
        )

        date_col = "trade_date" if "trade_date" in df.columns else ("timestamp" if "timestamp" in df.columns else "date")
        trade_date = ""
        if date_col in df.columns:
            value = df[date_col].iloc[-1]
            trade_date = value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)

        scores = self.scoring_system.calculate_context_scores(
            df,
            symbol=symbol,
            trade_date=trade_date,
        )
        context.classic_score = self._to_classic_score(scores["classic"])
        context.trend_score = self._to_trend_score(scores["trend"])
        context.breakout_score = self._to_breakout_score(scores["breakout"])
        context.market_state = self.market_state_builder(df)
        context.position_assessment = self._build_position_assessment(df, context.classic_score)
        context.stop_loss_config = self.stop_loss_calculator.calculate(df, context)
        context.candlestick_patterns = recognize_talib_patterns(df, lookback=5).get("patterns", [])
        context.screening_result = self._run_screening(df, context.classic_score)
        context.strategy_signals = self._run_strategy_signals(df)
        context.fundamental_data = self._load_fundamental_data(symbol)
        context.df_last_row = self._build_last_row(df)
        context.final_recommendation = self.recommendation_engine.generate_recommendation(context)
        return context

    def _to_classic_score(self, result: ScoreResult) -> ClassicScore:
        details = result.details or {}
        legacy = details.get("legacy_result", {})
        factors_raw = details.get("factors_raw", legacy.get("factors_raw", {}))
        return ClassicScore(
            score=result.final_score,
            trend_score=details.get("trend_score", legacy.get("trend_score", 50)),
            position_modifier=details.get("position_modifier", 1.0),
            score_grade=details.get("score_grade", legacy.get("score_grade", "一般")),
            factors_score=details.get("factors_score", legacy.get("factors_score", {})),
            factors_raw=factors_raw,
            execution=details.get("execution", legacy.get("execution", {})),
            warnings=details.get("warnings", legacy.get("warnings", [])),
        )

    def _to_trend_score(self, result: ScoreResult) -> TrendScore:
        details = result.details or {}
        timing_coefficient = result.timing_coefficient
        if timing_coefficient is None:
            timing_coefficient = details.get("timing_coefficient")
        if timing_coefficient is None:
            timing_coefficient = 0 if not result.passed_filter else 1.0

        timing_type = details.get("timing_type")
        if timing_type is None:
            timing_type = "standard" if not result.passed_filter else "标准"

        return TrendScore(
            final_score=result.final_score,
            trend_total_score=details.get("trend_total_score", 0),
            timing_coefficient=timing_coefficient,
            timing_type=timing_type,
            passed_hard_filter=result.passed_filter,
            hard_filter_reason=result.filter_reason,
            ma_structure_score=details.get("ma_structure_score", 0),
            price_momentum_score=details.get("price_momentum_score", 0),
            volume_score=details.get("volume_score", 0),
            relative_strength_score=details.get("relative_strength_score", 0),
            details=details,
        )

    def _to_breakout_score(self, result: ScoreResult) -> BreakoutScore:
        details = result.details or {}
        flat_detail_keys = {
            "is_low_position",
            "is_consolidating",
            "has_breakout",
            "quality_score",
            "growth_score",
            "value_score",
            "momentum_score",
            "flow_score",
            "risk_score",
            "consolidation_days",
            "price_range",
            "volume_ratio",
            "breakout_strength",
        }
        legacy_details = {
            key: value for key, value in details.items()
            if key not in flat_detail_keys
        }
        return BreakoutScore(
            final_score=result.final_score,
            is_low_position=details.get("is_low_position", False),
            is_consolidating=details.get("is_consolidating", False),
            has_breakout=details.get("has_breakout", False),
            passed_filter=result.passed_filter,
            filter_reason=result.filter_reason,
            quality_score=details.get("quality_score", 50.0),
            growth_score=details.get("growth_score", 50.0),
            value_score=details.get("value_score", 50.0),
            momentum_score=details.get("momentum_score", 50.0),
            flow_score=details.get("flow_score", 50.0),
            risk_score=details.get("risk_score", 50.0),
            consolidation_days=details.get("consolidation_days", 0),
            price_range=details.get("price_range", 0.0),
            volume_ratio=details.get("volume_ratio", 1.0),
            breakout_strength=details.get("breakout_strength", 0.0),
            stop_loss_price=result.stop_loss_price or 0.0,
            take_profit_price=result.take_profit_price or 0.0,
            details=legacy_details or details,
        )

    def _build_default_market_state(self, df: pd.DataFrame) -> UnifiedMarketState:
        try:
            from quanttool.strategies.adaptive_threshold import IndexMarketDetector, MarketRegime

            dual_state = IndexMarketDetector(default_index="hs300").get_dual_market_state(df)
            regime_map = {
                MarketRegime.BULL: MarketState.BULL,
                MarketRegime.BEAR: MarketState.BEAR,
                MarketRegime.SIDEWAY: MarketState.SIDEWAY,
                MarketRegime.VOLATILE: MarketState.VOLATILE,
            }
            return UnifiedMarketState(
                index_regime=regime_map.get(dual_state.index_regime, MarketState.SIDEWAY),
                stock_regime=regime_map.get(dual_state.stock_regime, MarketState.SIDEWAY),
                combined_regime=regime_map.get(dual_state.index_regime, MarketState.SIDEWAY),
                confidence=dual_state.confidence,
                combined_signal=dual_state.combined_signal.value,
                index_code=dual_state.index_code,
                index_name=dual_state.index_name,
            )
        except Exception:
            return UnifiedMarketState()

    def _build_position_assessment(self, df: pd.DataFrame, classic_score: ClassicScore) -> PositionAssessment:
        """Build position assessment using the current StockAnalyzer rules."""
        if df.empty:
            return PositionAssessment()

        latest = df.iloc[-1]
        close = latest.get("close", 0)
        ma20 = latest.get("ma_20", close)
        ma50 = latest.get("ma_50", close)
        ma200 = latest.get("ma_200", 0)

        position_modifier = classic_score.position_modifier

        wr = latest.get("wr", 50)
        cci = latest.get("cci", 0)
        rsi = latest.get("rsi_24", 50)
        boll_upper = latest.get("boll_upper", close)
        boll_lower = latest.get("boll_lower", close)
        boll_pctb = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper > boll_lower else 0.5
        is_extreme_overbought = wr < 10 or cci > 200 or rsi > 80 or boll_pctb > 0.95
        is_extreme_oversold = wr > 90 or cci < -200 or rsi < 20 or boll_pctb < 0.05
        is_overbought = (wr < 20 or cci > 100 or rsi > 70 or boll_pctb > 0.85) and not is_extreme_overbought
        is_oversold = (wr > 80 or cci < -100 or rsi < 30 or boll_pctb < 0.15) and not is_extreme_oversold

        avg_ma = (ma20 + ma50) / 2 if ma20 > 0 and ma50 > 0 else close
        price_ratio = close / avg_ma if avg_ma > 0 else 1.0
        bias20 = (close / ma20 - 1) if ma20 > 0 else 0

        long_term_position = "mid"
        short_term_position = "mid"

        if len(df) >= 60:
            high_60 = df["high"].iloc[-60:].max()
            low_60 = df["low"].iloc[-60:].min()
            if high_60 > low_60:
                position_ratio = (close - low_60) / (high_60 - low_60)
                if position_ratio < 0.35:
                    long_term_position = "low"
                elif position_ratio > 0.65:
                    long_term_position = "high"

        if is_extreme_oversold or is_oversold:
            short_term_position = "low"
        elif is_extreme_overbought or is_overbought:
            short_term_position = "high"

        if is_extreme_oversold or is_oversold:
            position = "low"
            reason = f"技术指标超卖（WR={wr:.1f}, RSI={rsi:.1f}）"
        elif is_extreme_overbought or is_overbought:
            position = "high"
            reason = f"技术指标超买（WR={wr:.1f}, RSI={rsi:.1f}）"
        elif boll_pctb < 0.25:
            position = "low"
            reason = f"布林带下轨附近（位置={boll_pctb*100:.0f}%）"
        elif boll_pctb > 0.75:
            position = "high"
            reason = f"布林带上轨附近（位置={boll_pctb*100:.0f}%）"
        elif price_ratio < 0.95:
            position = "low"
            reason = f"均线位置偏低（价格/均价={price_ratio:.2f}）"
        elif price_ratio > 1.05:
            position = "high"
            reason = f"均线位置偏高（价格/均价={price_ratio:.2f}）"
        else:
            position = "middle"
            reason = f"位置适中（修正系数={position_modifier:.2f}）"

        return PositionAssessment(
            position=position,
            long_term_position=long_term_position,
            short_term_position=short_term_position,
            is_overbought=is_overbought or is_extreme_overbought,
            is_oversold=is_oversold or is_extreme_oversold,
            is_extreme_overbought=is_extreme_overbought,
            is_extreme_oversold=is_extreme_oversold,
            position_modifier=position_modifier,
            price_ratio=price_ratio,
            boll_pctb=boll_pctb,
            bias20=bias20,
            close=close,
            ma20=ma20,
            ma50=ma50,
            ma200=ma200 if not pd.isna(ma200) else 0,
            reason=reason,
        )

    def _run_screening(self, df: pd.DataFrame, classic_score: ClassicScore) -> dict:
        try:
            outcome = StockScreener().screen(df, classic_score.factors_raw)
            return {
                "result": outcome.result.value,
                "score_modifier": outcome.score_modifier,
                "reasons": outcome.reasons,
                "details": outcome.details,
            }
        except Exception as exc:
            return {"result": "error", "score_modifier": 1.0, "reasons": [str(exc)], "details": {}}

    def _run_strategy_signals(self, df: pd.DataFrame) -> dict:
        strategies = TradingStrategies()
        signals = {}
        try:
            signals["rsi"] = strategies.evaluate_current_signal(strategies.rsi_strategy(df), "RSI策略")
            signals["macd"] = strategies.evaluate_current_signal(strategies.macd_strategy(df), "MACD策略")
            signals["ma"] = strategies.evaluate_current_signal(strategies.ma_crossover_strategy(df), "均线交叉策略")
            signals["boll"] = strategies.evaluate_current_signal(strategies.bollinger_bands_strategy(df), "布林带策略")
        except Exception as exc:
            signals["error"] = str(exc)
        return signals

    def _load_fundamental_data(self, symbol: str) -> FundamentalData:
        if self.fundamental_provider is not None:
            try:
                return self.fundamental_provider(symbol)
            except Exception:
                return FundamentalData()

        try:
            from quanttool.infrastructure.data_providers.fundamental_provider import FundamentalDataProvider

            fd_dict = FundamentalDataProvider().get_fundamental_summary(symbol)
            fd = FundamentalData()
            for key, value in fd_dict.items():
                if hasattr(fd, key):
                    setattr(fd, key, value)
            print(f"基本面数据: PE={fd.pe_ttm or 'N/A'}, ROE={fd.roe or 'N/A'}%, 数据源={fd.data_source}")
            return fd
        except Exception as exc:
            print(f"基本面数据获取失败: {exc}")
            return FundamentalData()

    def _build_last_row(self, df: pd.DataFrame) -> dict:
        latest = df.iloc[-1]
        last_row = latest.to_dict() if hasattr(latest, "to_dict") else dict(latest)
        amt_col = "amount" if "amount" in df.columns else ("amt" if "amt" in df.columns else None)
        if amt_col:
            last_row["amt_ma20"] = df[amt_col].tail(20).mean()
        return last_row
