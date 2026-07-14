# QuantTool Algorithm Core Refactor Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe first wave for algorithm-layer cleanup by adding deterministic algorithm fixtures, unifying scoring contracts, extracting analysis-context orchestration, and delegating context report generation out of `StockAnalyzer`.

**Architecture:** Keep public legacy entry points stable while moving new behavior behind focused modules. `UnifiedScoringSystem` becomes the scoring contract facade, `AnalysisOrchestrator` builds `AnalysisContext` from prepared DataFrames, and `StockReportGenerator` renders context reports without data access.

**Tech Stack:** Python >=3.8, pandas, numpy, unittest, existing QuantTool `AnalysisContext` dataclasses, existing `quanttool.factors.scoring` strategy package.

## Global Constraints

- Do not delete legacy public imports for `quanttool.factors.stock_analyzer.StockAnalyzer`, `quanttool.factors.scoring_system.ScoringSystem`, `quanttool.factors.trend_scoring_system.TrendScoringSystem`, or `quanttool.factors.breakout_scoring_system.BreakoutScoringSystem`.
- Do not change scoring thresholds, factor weights, recommendation rules, API paths, CLI commands, or report section text.
- When extracting orchestration code, preserve the current `StockAnalyzer._build_position_assessment` behavior exactly; do not replace it with a simplified heuristic.
- Preserve the current default fundamental-data loading behavior in `StockAnalyzer.build_analysis_context`; dependency injection may override it only for deterministic tests.
- When moving report helpers into a subpackage, update imports to keep existing dependencies valid without changing rendered report text.
- Do not touch `quanttool/backtest/engine.py`, `quanttool/strategies/qlib_strategy.py`, `quanttool/strategies/gbm_strategy.py`, or `quanttool/factors/ml_feature_engineer.py` in this phase.
- Do not add runtime dependencies.
- Tests must use deterministic local OHLCV fixtures and must not require qlib, databases, network data providers, or realtime行情.
- Keep unrelated user files such as `AGENTS.md` out of commits.

---

## File Structure

- Create: `tests/fixtures/__init__.py` to make algorithm test fixtures importable.
- Create: `tests/fixtures/algorithm_data.py` for deterministic OHLCV sample builders.
- Create: `tests/test_algorithm_fixtures.py` for fixture shape and determinism tests.
- Modify: `quanttool/factors/scoring/strategies/multi_dimension.py` to call the existing `ScoringSystem.calculate_all_scores` API correctly.
- Modify: `quanttool/factors/scoring/unified_scoring_system.py` to add `calculate_context_scores`.
- Create: `tests/test_scoring_contracts.py` for scoring facade behavior.
- Create: `quanttool/factors/analysis_orchestrator.py` for context assembly.
- Create: `tests/test_analysis_orchestrator.py` for dependency-injected context assembly and facade delegation.
- Create: `quanttool/factors/reports/__init__.py`.
- Create: `quanttool/factors/reports/stock_report.py` for Markdown report generation from `AnalysisContext`.
- Modify: `quanttool/factors/stock_analyzer.py` to delegate `build_analysis_context` and `generate_report_from_context`.

---

### Task 1: Add Deterministic Algorithm Fixtures

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/algorithm_data.py`
- Create: `tests/test_algorithm_fixtures.py`

**Interfaces:**
- Produces: `make_trending_ohlcv(rows: int = 260) -> pd.DataFrame`
- Produces: `make_sideways_ohlcv(rows: int = 260) -> pd.DataFrame`
- Produces: `make_breakout_ohlcv(rows: int = 260) -> pd.DataFrame`
- Produces: `make_indicator_ready_ohlcv(rows: int = 260) -> pd.DataFrame`

- [ ] **Step 1: Create fixture package marker**

Create `tests/fixtures/__init__.py`:

```python
"""Shared deterministic fixtures for tests."""
```

- [ ] **Step 2: Write failing fixture tests**

Create `tests/test_algorithm_fixtures.py`:

```python
import unittest

import pandas as pd

from tests.fixtures.algorithm_data import (
    make_breakout_ohlcv,
    make_indicator_ready_ohlcv,
    make_sideways_ohlcv,
    make_trending_ohlcv,
)


class AlgorithmFixtureTests(unittest.TestCase):
    def test_ohlcv_fixtures_have_required_columns(self):
        required = {
            "timestamp",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        }

        for builder in [
            make_trending_ohlcv,
            make_sideways_ohlcv,
            make_breakout_ohlcv,
        ]:
            with self.subTest(builder=builder.__name__):
                df = builder(rows=260)
                self.assertEqual(len(df), 260)
                self.assertTrue(required.issubset(df.columns))
                self.assertFalse(df[list(required - {"date"})].isna().any().any())
                self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))

    def test_fixtures_are_deterministic(self):
        first = make_trending_ohlcv(rows=260)
        second = make_trending_ohlcv(rows=260)
        pd.testing.assert_frame_equal(first, second)

    def test_indicator_ready_fixture_contains_legacy_columns(self):
        df = make_indicator_ready_ohlcv(rows=260)
        for column in [
            "ma_5",
            "ma_10",
            "ma_20",
            "ma_50",
            "ma_200",
            "atr_14",
            "boll_upper",
            "boll_mid",
            "boll_lower",
            "rsi_24",
            "wr",
            "cci",
        ]:
            self.assertIn(column, df.columns)
        self.assertGreater(df["ma_20"].iloc[-1], 0)
        self.assertGreater(df["atr_14"].iloc[-1], 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Verify red**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_algorithm_fixtures -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tests.fixtures.algorithm_data'`.

- [ ] **Step 4: Implement deterministic fixture builders**

Create `tests/fixtures/algorithm_data.py`:

```python
"""Deterministic OHLCV fixtures for algorithm tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dates(rows: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=rows, freq="B")


def _frame_from_close(close: np.ndarray) -> pd.DataFrame:
    rows = len(close)
    dates = _dates(rows)
    idx = np.arange(rows, dtype=float)
    open_ = close * (1 + 0.002 * np.sin(idx / 7.0))
    high = np.maximum(open_, close) * (1 + 0.008 + 0.001 * np.cos(idx / 5.0))
    low = np.minimum(open_, close) * (1 - 0.008 - 0.001 * np.sin(idx / 6.0))
    volume = 1_200_000 + (idx % 21) * 25_000

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "date": dates.strftime("%Y-%m-%d"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df["amount"] = df["close"] * df["volume"]
    return df


def _add_indicator_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]

    for period in [5, 10, 20, 50, 200]:
        result[f"ma_{period}"] = close.rolling(period, min_periods=1).mean()

    mid = close.rolling(20, min_periods=1).mean()
    std = close.rolling(20, min_periods=1).std(ddof=0).fillna(0)
    result["boll_mid"] = mid
    result["boll_upper"] = mid + 2 * std
    result["boll_lower"] = mid - 2 * std

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr_14"] = tr.rolling(14, min_periods=1).mean()

    delta = close.diff().fillna(0)
    gain = delta.where(delta > 0, 0).rolling(24, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(24, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    result["rsi_24"] = 100 - (100 / (1 + rs))

    highest = high.rolling(14, min_periods=1).max()
    lowest = low.rolling(14, min_periods=1).min()
    result["wr"] = (highest - close) / (highest - lowest + 1e-10) * 100

    typical = (high + low + close) / 3
    typical_ma = typical.rolling(14, min_periods=1).mean()
    mean_dev = typical.rolling(14, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(),
        raw=True,
    )
    result["cci"] = (typical - typical_ma) / (0.015 * mean_dev.replace(0, np.nan))
    result["cci"] = result["cci"].fillna(0)
    return result


def make_trending_ohlcv(rows: int = 260) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 10.0 + idx * 0.035 + 0.18 * np.sin(idx / 8.0)
    return _frame_from_close(close)


def make_sideways_ohlcv(rows: int = 260) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 12.0 + 0.28 * np.sin(idx / 6.0) + 0.08 * np.cos(idx / 13.0)
    return _frame_from_close(close)


def make_breakout_ohlcv(rows: int = 260) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 9.0 + 0.10 * np.sin(idx / 5.0)
    close[-40:-5] = 9.2 + 0.05 * np.sin(idx[-40:-5] / 3.0)
    close[-5:] = np.linspace(9.45, 10.25, 5)
    return _frame_from_close(close)


def make_indicator_ready_ohlcv(rows: int = 260) -> pd.DataFrame:
    return _add_indicator_columns(make_trending_ohlcv(rows=rows))
```

- [ ] **Step 5: Verify green**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_algorithm_fixtures -v
```

Expected: PASS, 3 tests OK.

- [ ] **Step 6: Commit fixture baseline**

Run:

```bash
git add tests/fixtures/__init__.py tests/fixtures/algorithm_data.py tests/test_algorithm_fixtures.py
git commit -m "test: add deterministic algorithm fixtures"
```

---

### Task 2: Add Unified Scoring Contract

**Files:**
- Modify: `quanttool/factors/scoring/strategies/multi_dimension.py`
- Modify: `quanttool/factors/scoring/unified_scoring_system.py`
- Create: `tests/test_scoring_contracts.py`

**Interfaces:**
- Consumes: `ScoreResult` and `ScoringStrategy`.
- Produces: `UnifiedScoringSystem.calculate_context_scores(df: pd.DataFrame, symbol: str = "", trade_date: str = "") -> Dict[str, ScoreResult]`.

- [ ] **Step 1: Write failing scoring contract tests**

Create `tests/test_scoring_contracts.py`:

```python
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
```

- [ ] **Step 2: Verify red**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_scoring_contracts -v
```

Expected: FAIL with `AttributeError: 'UnifiedScoringSystem' object has no attribute 'calculate_context_scores'`.

- [ ] **Step 3: Fix multi-dimension legacy adapter**

Replace `MultiDimensionScoringStrategy.calculate_score` in `quanttool/factors/scoring/strategies/multi_dimension.py` with:

```python
    def calculate_score(self, df: pd.DataFrame, **kwargs) -> ScoreResult:
        """计算多维度评分"""
        result = self._legacy_system.calculate_all_scores(
            df=df,
            stock_code=kwargs.get("symbol", ""),
            trade_date_T=kwargs.get("trade_date", ""),
            trade_date_T1=kwargs.get("trade_date_t1"),
            open_T1=kwargs.get("open_t1"),
        )

        if "error" in result:
            return ScoreResult(
                final_score=0,
                passed_filter=False,
                filter_reason=result["error"],
                strategy_name=self.name,
                details={"legacy_result": result},
            )

        return ScoreResult(
            final_score=result.get("score", result.get("final_score", 0)),
            passed_filter=result.get("bias_passed", True),
            filter_reason=result.get("filter_reason", ""),
            strategy_name=self.name,
            details={
                "legacy_result": result,
                "trend_score": result.get("trend_score", 0),
                "momentum_score": result.get("momentum_score", 0),
                "money_score": result.get("money_score", 0),
                "trend_bonus": result.get("trend_bonus", 0),
                "volume_bonus": result.get("volume_bonus", 0),
                "trigger_type": result.get("trigger_type", "none"),
                "trigger_detail": result.get("trigger_detail", ""),
                "factors_raw": result.get("factors_raw", {}),
                "factors_score": result.get("factors_score", {}),
                "execution": result.get("execution", {}),
                "warnings": result.get("warnings", []),
                "score_grade": result.get("score_grade", "一般"),
            },
        )
```

- [ ] **Step 4: Add context scoring facade method**

Add this method to `UnifiedScoringSystem` in `quanttool/factors/scoring/unified_scoring_system.py` after `calculate_scores`:

```python
    def calculate_context_scores(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        trade_date: str = "",
    ) -> Dict[str, ScoreResult]:
        """Calculate scores using context-facing names.

        Returns keys matching AnalysisContext score families:
        classic, trend, and breakout.
        """
        raw_scores = self.calculate_scores(
            df,
            symbol=symbol,
            trade_date=trade_date,
        )
        aliases = {
            "classic": "multi_dimension",
            "trend": "trend",
            "breakout": "breakout",
        }

        context_scores: Dict[str, ScoreResult] = {}
        for context_key, strategy_name in aliases.items():
            result = raw_scores.get(strategy_name)
            if result is None:
                result = ScoreResult(
                    final_score=0,
                    passed_filter=False,
                    filter_reason=f"评分策略缺失: {strategy_name}",
                    strategy_name=context_key,
                )
            else:
                result.strategy_name = context_key
            context_scores[context_key] = result

        return context_scores
```

- [ ] **Step 5: Verify green**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_scoring_contracts -v
```

Expected: PASS, 2 tests OK.

- [ ] **Step 6: Run fixtures plus scoring tests**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_algorithm_fixtures tests.test_scoring_contracts -v
```

Expected: PASS.

- [ ] **Step 7: Commit scoring contract**

Run:

```bash
git add quanttool/factors/scoring/strategies/multi_dimension.py quanttool/factors/scoring/unified_scoring_system.py tests/test_scoring_contracts.py
git commit -m "refactor: add unified scoring context contract"
```

---

### Task 3: Extract Analysis Context Orchestrator

**Files:**
- Create: `quanttool/factors/analysis_orchestrator.py`
- Create or modify: `tests/test_analysis_orchestrator.py`

**Interfaces:**
- Consumes: `UnifiedScoringSystem.calculate_context_scores(...)`.
- Produces: `AnalysisOrchestrator.build_context(df: pd.DataFrame, symbol: str, primary_system: ScoringSystemType = ScoringSystemType.AUTO, current_price: Optional[float] = None) -> AnalysisContext`.

- [ ] **Step 1: Write failing orchestrator tests**

Create `tests/test_analysis_orchestrator.py` with the imports and fakes below:

```python
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
```

Append these test cases:

```python
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
```

- [ ] **Step 2: Verify red**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_analysis_orchestrator.AnalysisOrchestratorTests -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'quanttool.factors.analysis_orchestrator'`.

- [ ] **Step 3: Implement analysis orchestrator**

Create `quanttool/factors/analysis_orchestrator.py`:

```python
"""Analysis context orchestration for single-stock analysis."""

from datetime import datetime
from typing import Callable, Optional

import pandas as pd

from .analysis_context import (
    AnalysisContext,
    BreakoutScore,
    ClassicScore,
    FundamentalData,
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
```

Continue the same file with helper methods:

```python
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
        return TrendScore(
            final_score=result.final_score,
            trend_total_score=details.get("trend_total_score", 0),
            timing_coefficient=result.timing_coefficient or details.get("timing_coefficient", 1.0),
            timing_type=details.get("timing_type", "标准"),
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
            details=details,
        )
```

Finish the file with the remaining helpers:

```python
    def _build_default_market_state(self, df: pd.DataFrame) -> UnifiedMarketState:
        try:
            from quanttool.strategies.adaptive_threshold import IndexMarketDetector, MarketRegime
            from .analysis_context import MarketState

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
```

- [ ] **Step 4: Verify orchestrator tests green**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_analysis_orchestrator.AnalysisOrchestratorTests -v
```

Expected: PASS, 2 tests OK.

- [ ] **Step 5: Run scoring and orchestrator tests together**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_algorithm_fixtures tests.test_scoring_contracts tests.test_analysis_orchestrator -v
```

Expected: PASS.

- [ ] **Step 6: Commit orchestrator**

Run:

```bash
git add quanttool/factors/analysis_orchestrator.py tests/test_analysis_orchestrator.py
git commit -m "refactor: extract analysis context orchestrator"
```

---

### Task 4: Extract Context Report Generator

**Files:**
- Create: `quanttool/factors/reports/__init__.py`
- Create: `quanttool/factors/reports/stock_report.py`
- Modify: `tests/test_analysis_orchestrator.py`

**Interfaces:**
- Produces: `StockReportGenerator.generate(df: pd.DataFrame, context: AnalysisContext, symbol: str) -> str`.

- [ ] **Step 1: Add failing report generator test**

Append to `tests/test_analysis_orchestrator.py`:

```python
class StockReportGeneratorTests(unittest.TestCase):
    def test_report_generator_renders_context_sections(self):
        from quanttool.factors.analysis_orchestrator import AnalysisOrchestrator
        from quanttool.factors.reports.stock_report import StockReportGenerator

        df = make_indicator_ready_ohlcv(rows=260)
        context = AnalysisOrchestrator(
            scoring_system=FakeScoringSystem(),
            recommendation_engine=FakeRecommendationEngine(),
            stop_loss_calculator=FakeStopLossCalculator(),
            market_state_builder=lambda data: UnifiedMarketState(confidence=0.75),
            fundamental_provider=lambda symbol: FundamentalData(data_source="fake"),
        ).build_context(df, "000001.SZ", current_price=12.34)

        report = StockReportGenerator().generate(df, context, "000001.SZ")

        self.assertIn("# 股票技术分析报告：000001.SZ", report)
        self.assertIn("## 第一部分：核心结论", report)
        self.assertIn("## 第二部分：三系统评分对比", report)
        self.assertIn("## 第四部分：交易执行计划", report)
        self.assertIn("测试推荐", report)
```

- [ ] **Step 2: Verify red**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_analysis_orchestrator.StockReportGeneratorTests -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'quanttool.factors.reports'`.

- [ ] **Step 3: Create report package marker**

Create `quanttool/factors/reports/__init__.py`:

```python
"""Report generators for factor analysis."""

from .stock_report import StockReportGenerator

__all__ = ["StockReportGenerator"]
```

- [ ] **Step 4: Move context report methods into generator**

Create `quanttool/factors/reports/stock_report.py` with this header:

```python
"""Markdown stock report generation from AnalysisContext."""

from datetime import datetime
from typing import List

import pandas as pd

from quanttool.factors.analysis_context import (
    ActionType,
    AnalysisContext,
    MarketState,
    ScoringSystemType,
    StopLossType as UnifiedStopLossType,
)
from quanttool.factors.fundamental_rating import FundamentalRating


class StockReportGenerator:
    """Generate Markdown reports from a prepared AnalysisContext."""
```

Move the following methods from `quanttool/factors/stock_analyzer.py` into `StockReportGenerator` without changing their report text:

- `generate_report_from_context` body from lines starting at `def generate_report_from_context`.
- `_generate_core_conclusion_v2`.
- `_generate_three_system_analysis`.
- `_generate_market_risk_section`.
- `_generate_fundamental_section`.
- `_generate_trading_plan_v2`.
- `_generate_technical_indicators_table`.

Because `StockReportGenerator` lives in `quanttool.factors.reports`, remove the local import `from .fundamental_rating import FundamentalRating` from `_generate_fundamental_section`; use the module-level `from quanttool.factors.fundamental_rating import FundamentalRating` import instead. Do not change the generated Markdown text.

Rename the public moved method to `generate`:

```python
    def generate(
        self,
        df: pd.DataFrame,
        context: AnalysisContext,
        symbol: str,
    ) -> str:
        """Generate Markdown report from a prepared analysis context."""
```

Inside the moved body, keep all calls such as `self._generate_core_conclusion_v2(context)` unchanged because the helper methods now live on `StockReportGenerator`.

- [ ] **Step 5: Verify report generator test green**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_analysis_orchestrator.StockReportGeneratorTests -v
```

Expected: PASS.

- [ ] **Step 6: Run all algorithm tests**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_algorithm_fixtures tests.test_scoring_contracts tests.test_analysis_orchestrator -v
```

Expected: PASS.

- [ ] **Step 7: Commit report generator**

Run:

```bash
git add quanttool/factors/reports/__init__.py quanttool/factors/reports/stock_report.py tests/test_analysis_orchestrator.py
git commit -m "refactor: extract stock context report generator"
```

---

### Task 5: Delegate StockAnalyzer Context Methods

**Files:**
- Modify: `quanttool/factors/stock_analyzer.py`
- Modify: `tests/test_analysis_orchestrator.py`

**Interfaces:**
- Consumes: `AnalysisOrchestrator.build_context(...)`.
- Consumes: `StockReportGenerator.generate(...)`.
- Preserves: `StockAnalyzer.build_analysis_context(...) -> AnalysisContext`.
- Preserves: `StockAnalyzer.generate_report_from_context(...) -> str`.

- [ ] **Step 1: Add failing facade delegation tests**

Append to `tests/test_analysis_orchestrator.py`:

```python
class StockAnalyzerFacadeDelegationTests(unittest.TestCase):
    def test_stock_analyzer_build_context_delegates_to_orchestrator(self):
        from quanttool.factors.analysis_context import AnalysisContext
        from quanttool.factors.stock_analyzer import StockAnalyzer

        class FakeOrchestrator:
            def __init__(self):
                self.calls = []

            def build_context(self, df, symbol, primary_system=ScoringSystemType.AUTO, current_price=None):
                self.calls.append((symbol, current_price, primary_system))
                return AnalysisContext(
                    symbol=symbol,
                    current_price=current_price,
                    analysis_date=df["timestamp"].iloc[-1].to_pydatetime(),
                )

        df = make_indicator_ready_ohlcv(rows=260)
        analyzer = StockAnalyzer.__new__(StockAnalyzer)
        analyzer.fetcher = None
        analyzer._realtime_price_cache = {}
        analyzer.analysis_orchestrator = FakeOrchestrator()

        context = analyzer.build_analysis_context(df, "000001.SZ")

        self.assertEqual(context.symbol, "000001.SZ")
        self.assertEqual(context.current_price, df["close"].iloc[-1])
        self.assertEqual(len(analyzer.analysis_orchestrator.calls), 1)

    def test_stock_analyzer_report_delegates_to_report_generator(self):
        from quanttool.factors.analysis_context import AnalysisContext
        from quanttool.factors.stock_analyzer import StockAnalyzer

        class FakeReportGenerator:
            def generate(self, df, context, symbol):
                return f"report:{symbol}:{context.current_price:.2f}:{len(df)}"

        df = make_indicator_ready_ohlcv(rows=260)
        analyzer = StockAnalyzer.__new__(StockAnalyzer)
        analyzer.stock_report_generator = FakeReportGenerator()
        context = AnalysisContext(
            symbol="000001.SZ",
            current_price=12.34,
            analysis_date=df["timestamp"].iloc[-1].to_pydatetime(),
        )

        report = analyzer.generate_report_from_context(df, context, "000001.SZ")

        self.assertEqual(report, "report:000001.SZ:12.34:260")
```

- [ ] **Step 2: Verify red**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_analysis_orchestrator.StockAnalyzerFacadeDelegationTests -v
```

Expected: FAIL because `StockAnalyzer.build_analysis_context` still runs its own implementation and does not use `analysis_orchestrator`.

- [ ] **Step 3: Add new imports to StockAnalyzer**

Add to `quanttool/factors/stock_analyzer.py` near the other `quanttool.factors` imports:

```python
from quanttool.factors.analysis_orchestrator import AnalysisOrchestrator
from quanttool.factors.reports import StockReportGenerator
```

- [ ] **Step 4: Initialize delegates in StockAnalyzer.__init__**

Add near the end of `StockAnalyzer.__init__`, after cache fields are initialized:

```python
        self.analysis_orchestrator = AnalysisOrchestrator()
        self.stock_report_generator = StockReportGenerator()
```

- [ ] **Step 5: Replace build_analysis_context body with delegation**

Replace the body of `StockAnalyzer.build_analysis_context` with:

```python
        if df.empty:
            return AnalysisContext(
                symbol=symbol,
                current_price=0,
                analysis_date=datetime.now()
            )

        latest = df.iloc[-1]
        close = latest.get('close', 0)

        if self.fetcher:
            try:
                normalized = self._normalize_symbol(symbol)
                if normalized in self._realtime_price_cache:
                    cached_price = self._realtime_price_cache[normalized]
                    if cached_price and cached_price > 0:
                        close = cached_price
                else:
                    realtime_price = self.fetcher.get_realtime_price(normalized)
                    if realtime_price and realtime_price > 0:
                        close = realtime_price
                        self._realtime_price_cache[normalized] = realtime_price
            except Exception:
                pass

        return self.analysis_orchestrator.build_context(
            df,
            symbol,
            primary_system=primary_system,
            current_price=close,
        )
```

- [ ] **Step 6: Replace generate_report_from_context body with delegation**

Replace the body of `StockAnalyzer.generate_report_from_context` with:

```python
        return self.stock_report_generator.generate(df, context, symbol)
```

- [ ] **Step 7: Verify facade tests green**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_analysis_orchestrator.StockAnalyzerFacadeDelegationTests -v
```

Expected: PASS.

- [ ] **Step 8: Run all algorithm tests**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_algorithm_fixtures tests.test_scoring_contracts tests.test_analysis_orchestrator -v
```

Expected: PASS.

- [ ] **Step 9: Commit facade delegation**

Run:

```bash
git add quanttool/factors/stock_analyzer.py tests/test_analysis_orchestrator.py
git commit -m "refactor: delegate stock analyzer context workflow"
```

---

### Task 6: Final Verification

**Files:**
- All files touched in Tasks 1 through 5.

**Interfaces:**
- Consumes: complete test suite, compile check, frontend lint.
- Produces: verified first algorithm refactor wave ready for review.

- [ ] **Step 1: Run algorithm-focused tests**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_algorithm_fixtures tests.test_scoring_contracts tests.test_analysis_orchestrator -v
```

Expected: all tests pass.

- [ ] **Step 2: Run existing smoke tests**

Run:

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Compile Python package**

Run:

```bash
.venv-mcp/bin/python -m compileall -q quanttool
```

Expected: command exits 0 with no output.

- [ ] **Step 4: Run frontend lint**

Run:

```bash
cd quanttool/web/frontend && npm run lint
```

Expected: lint completes with no warnings or errors.

- [ ] **Step 5: Confirm only intentional files changed**

Run:

```bash
git status --short
```

Expected: tracked changes from this plan are committed. `AGENTS.md` may remain as an untracked user file.

---

## Self-Review Notes

- Spec coverage: deterministic fixtures, unified scoring contract, analysis orchestration, report extraction, facade compatibility, and verification are covered by Tasks 1 through 6.
- Type consistency: all public signatures use Python 3.8-compatible `Optional[...]` typing.
- Dependency boundary: no task adds runtime dependencies or requires external market data.
- Scope boundary: backtest, Qlib, GBM, and ML feature engineering are explicitly untouched in this phase.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-09-algorithm-core-refactor-phase1.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
