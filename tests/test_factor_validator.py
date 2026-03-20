"""Tests for factor validator module."""

import pytest
import pandas as pd
import numpy as np

from quanttool.factors.factor_validator import (
    FactorValidator,
    validate_factor,
    ICResult,
    DecayResult,
    QuantileResult,
    FactorValidationReport,
)


class TestFactorValidator:
    """Test cases for FactorValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return FactorValidator(min_periods=20, ic_rolling_window=30)

    @pytest.fixture
    def sample_data(self):
        """Create sample factor and return data."""
        np.random.seed(42)
        n = 200

        # 创建因子值（带预测能力）
        factor = pd.Series(
            np.random.randn(n) + np.linspace(-0.5, 0.5, n),  # 趋势
            index=pd.date_range('2023-01-01', periods=n)
        )

        # 创建收益（与因子正相关）
        returns = pd.Series(
            factor.values * 0.5 + np.random.randn(n) * 0.5,
            index=factor.index
        )

        return factor, returns

    @pytest.fixture
    def weak_factor_data(self):
        """Create weak factor data (no predictive power)."""
        np.random.seed(123)
        n = 200

        factor = pd.Series(
            np.random.randn(n),
            index=pd.date_range('2023-01-01', periods=n)
        )

        returns = pd.Series(
            np.random.randn(n) * 0.1,
            index=factor.index
        )

        return factor, returns

    # ========== IC计算测试 ==========

    def test_calculate_ic(self, validator, sample_data):
        """Test IC calculation."""
        factor, returns = sample_data
        ic_series = validator.calculate_ic(factor, returns)

        assert len(ic_series) > 0
        assert ic_series.max() <= 1.0
        assert ic_series.min() >= -1.0

    def test_calculate_ic_with_nan(self, validator):
        """Test IC calculation handles NaN."""
        factor = pd.Series([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10] * 5)
        returns = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5)

        # 使用索引对齐
        factor.index = pd.date_range('2023-01-01', periods=50)
        returns.index = pd.date_range('2023-01-01', periods=50)

        ic_series = validator.calculate_ic(factor, returns)
        assert not ic_series.empty

    # ========== IC分析测试 ==========

    def test_analyze_ic_strong(self, validator, sample_data):
        """Test IC analysis for strong factor."""
        factor, returns = sample_data
        result = validator.analyze_ic(factor, returns, "test_factor")

        assert isinstance(result, ICResult)
        assert result.factor_name == "test_factor"
        assert result.mean_ic != 0
        assert result.ir != 0
        assert 0 <= result.positive_ic_ratio <= 1

    def test_analyze_ic_weak(self, validator, weak_factor_data):
        """Test IC analysis for weak factor."""
        factor, returns = weak_factor_data
        result = validator.analyze_ic(factor, returns, "weak_factor")

        # 弱因子的IC应该接近0
        assert abs(result.mean_ic) < 0.1

    # ========== IC衰减分析测试 ==========

    def test_analyze_decay(self, validator, sample_data):
        """Test IC decay analysis."""
        factor, returns = sample_data
        result = validator.analyze_decay(factor, returns, "test_factor", max_horizon=10)

        assert isinstance(result, DecayResult)
        assert result.factor_name == "test_factor"
        assert 1 <= result.optimal_horizon <= 10
        assert len(result.decay_ic) > 0

    def test_analyze_decay_empty_data(self, validator):
        """Test IC decay with insufficient data."""
        factor = pd.Series([1, 2, 3])
        returns = pd.Series([1, 2, 3])

        result = validator.analyze_decay(factor, returns, "empty_factor")
        assert result.decay_ic == {}

    # ========== 分层回测测试 ==========

    def test_run分层_backtest(self, validator, sample_data):
        """Test quantile backtest."""
        factor, returns = sample_data
        result = validator.run分层_backtest(factor, returns, "test_factor", num_groups=5)

        assert isinstance(result, QuantileResult)
        assert result.num_groups == 5
        assert len(result.group_returns) <= 5

    def test_run分层_backtest_long_short(self, validator, sample_data):
        """Test long-short portfolio in quantile backtest."""
        factor, returns = sample_data
        result = validator.run分层_backtest(factor, returns, "test_factor", num_groups=5)

        # 强因子应该有正的long_short_return
        assert isinstance(result.long_short_return, float)
        assert isinstance(result.long_short_sharpe, float)

    # ========== 完整验证测试 ==========

    def test_validate_effective_factor(self, validator, sample_data):
        """Test validation for effective factor."""
        factor, returns = sample_data
        report = validator.validate(factor, returns, "effective_factor")

        assert isinstance(report, FactorValidationReport)
        assert report.factor_name == "effective_factor"
        assert report.ic_result is not None
        assert report.decay_result is not None
        assert report.quantile_result is not None
        assert 0 <= report.overall_score <= 100

    def test_validate_weak_factor(self, validator, weak_factor_data):
        """Test validation for weak factor."""
        factor, returns = weak_factor_data
        report = validator.validate(factor, returns, "weak_factor")

        # 弱因子可能无效
        assert isinstance(report, FactorValidationReport)

    def test_validate_with_benchmark(self, validator, sample_data):
        """Test validation with benchmark."""
        factor, returns = sample_data
        benchmark = pd.Series(
            np.random.randn(len(returns)) * 0.01,
            index=returns.index
        )

        report = validator.validate(
            factor, returns, "test_factor",
            benchmark_return=benchmark
        )

        assert report.quantile_result is not None

    # ========== 便捷函数测试 ==========

    def test_validate_factor_helper(self, sample_data):
        """Test validate_factor helper function."""
        factor, returns = sample_data
        report = validate_factor(factor, returns, "helper_test")

        assert isinstance(report, FactorValidationReport)
        assert report.factor_name == "helper_test"


class TestValidatorEdgeCases:
    """Edge case tests for FactorValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return FactorValidator(min_periods=20, ic_rolling_window=30)

    def test_all_same_values(self, validator):
        """Test with all same factor values."""
        factor = pd.Series([5.0] * 100)
        factor.index = pd.date_range('2023-01-01', periods=100)
        returns = pd.Series(np.random.randn(100) * 0.1)
        returns.index = factor.index

        result = validator.analyze_ic(factor, returns, "constant_factor")
        # 常数因子的IC可能不稳定
        assert isinstance(result, ICResult)

    def test_perfect_correlation(self, validator):
        """Test with perfectly correlated data."""
        factor = pd.Series(list(range(100)))
        factor.index = pd.date_range('2023-01-01', periods=100)
        returns = pd.Series([x * 0.5 for x in range(100)])
        returns.index = factor.index

        result = validator.analyze_ic(factor, returns, "perfect_factor")
        assert result.mean_ic > 0.5  # 应该有很高的IC
